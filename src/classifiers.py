from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import (
    StandardScaler,
    label_binarize,
    LabelEncoder,
    MinMaxScaler,
)
import shap
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    PredefinedSplit,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    classification_report,
    roc_auc_score,
)
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
import joblib
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import label_binarize
from utils import convert_to_serializable
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    validation_size: float = 0.1  # Proportion of data to use for validation
    random_state: int = 42
    cv_folds: int = 5
    max_iter: int = 10000
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 10000
    patience: int = 20
    lr_scheduler_step_size: int = 100
    lr_scheduler_gamma: float = 0.1
    weight_decay: float = 1e-5
    n_iter_search: int = 20

    # Common parameters for both binary and multiclass
    probe_params = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    # Decision tree specific
    tree_params = {
        "max_depth": [3, 5, 7, 9],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }


class ModelTrainer:
    """Class to handle model training and evaluation with modern sklearn practices."""

    def __init__(self, config: TrainingConfig, label_encoder: LabelEncoder = None):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = label_encoder

    def binarize_features(self, data: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """Binarize features by clipping values based on a threshold."""
        print(f"Binarizing features with threshold: {threshold}")
        threshold = float(threshold)
        return np.where(data > threshold, 1, 0)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        classes: np.ndarray,
        class_labels: List[str],
    ) -> Dict[str, Any]:
        """Compute metrics with proper handling of binary and multiclass cases."""
        # Convert numerical predictions back to original labels for the classification report
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(
                y_true_labels, y_pred_labels, output_dict=True
            ),
        }

        n_classes = len(classes)
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics["roc_auc"] = auc(fpr, tpr)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=classes)
            metrics["roc_auc"] = {}
            metrics["roc_curve"] = {}

            # Class-specific metrics
            for i, (class_idx, class_label) in enumerate(zip(classes, class_labels)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                metrics["roc_auc"][class_label] = auc(fpr, tpr)
                metrics["roc_curve"][class_label] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }

            # Micro and macro averaging
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            metrics["roc_auc"]["micro"] = auc(fpr_micro, tpr_micro)
            metrics["roc_curve"]["micro"] = {
                "fpr": fpr_micro.tolist(),
                "tpr": tpr_micro.tolist(),
            }
            metrics["roc_auc"]["macro"] = roc_auc_score(
                y_true_bin, y_prob, average="macro", multi_class="ovr"
            )

        return metrics

    def train_linear_probe(
        self, df, hidden=True, binarize_value=None, compute_shap=False
    ) -> Dict[str, Any]:
        """Train an optimized linear probe classifier with k-fold cross-validation (PyTorch based),
        and integrate SHAP explanations for the final trained model.
        """
        logging.info("Training linear probe classifier...")

        # Prepare data
        X = df["hidden_states"] if hidden else df["features"]
        y = df["label"]

        # Apply binarization if required
        if binarize_value is not None:
            X = self.binarize_features(X, threshold=binarize_value)

        # Encode labels
        classes = np.unique(y)
        class_labels = self.label_encoder.classes_
        n_classes = len(classes)

        logging.info(f"Number of classes: {n_classes}")
        logging.info(f"Classes: {class_labels}")

        # Split data into training and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Further split training data into training and validation sets for final model training
        # (This validation set is used for the final model, not for cross-validation)
        validation_size = self.config.validation_size
        X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
            X_temp,
            y_temp,
            test_size=validation_size / (1 - self.config.test_size),
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ============================================================
        # K-fold cross-validation using PyTorch
        # We will perform K-fold CV on the (X_temp, y_temp) data which excludes the test set.
        # For each fold, we train a linear probe model and evaluate it on the validation fold.
        # ============================================================
        kfold = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        cv_accuracies = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_temp, y_temp)):
            # Get fold data
            X_train_fold = X_temp[train_idx]
            y_train_fold = y_temp[train_idx]
            X_val_fold = X_temp[val_idx]
            y_val_fold = y_temp[val_idx]

            # Compute class weights for this fold
            class_weights_fold = compute_class_weights(y_train_fold, classes)
            class_weights_tensor_fold = torch.tensor(
                class_weights_fold, dtype=torch.float32
            ).to(device)

            # Convert to tensors
            X_train_fold_tensor = torch.tensor(X_train_fold.astype("float32"))
            y_train_fold_tensor = torch.tensor(y_train_fold.astype("int64"))
            X_val_fold_tensor = torch.tensor(X_val_fold.astype("float32"))
            y_val_fold_tensor = torch.tensor(y_val_fold.astype("int64"))

            # Create datasets and loaders for this fold
            train_fold_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
            val_fold_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)

            train_fold_loader = DataLoader(
                train_fold_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            val_fold_loader = DataLoader(
                val_fold_dataset, batch_size=self.config.batch_size
            )

            # Define the linear probe model for this fold
            input_dim = X_train_fold.shape[1]
            fold_model = nn.Linear(input_dim, n_classes).to(device)

            # Loss and optimizer
            criterion_fold = nn.CrossEntropyLoss(weight=class_weights_tensor_fold)
            optimizer_fold = optim.Adam(
                fold_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            # Learning rate scheduler for fold
            scheduler_fold = optim.lr_scheduler.StepLR(
                optimizer_fold,
                step_size=self.config.lr_scheduler_step_size,
                gamma=self.config.lr_scheduler_gamma,
            )

            # Training loop with early stopping for this fold
            num_epochs = self.config.num_epochs
            patience = self.config.patience
            best_val_loss = float("inf")
            epochs_no_improve = 0
            best_model_state_fold = None

            for epoch in range(num_epochs):
                # Training phase
                fold_model.train()
                train_epoch_loss = 0.0
                for batch_features, batch_labels in train_fold_loader:
                    batch_features, batch_labels = batch_features.to(
                        device
                    ), batch_labels.to(device)

                    # Forward
                    outputs = fold_model(batch_features)
                    loss = criterion_fold(outputs, batch_labels)

                    # Backward
                    optimizer_fold.zero_grad()
                    loss.backward()
                    optimizer_fold.step()

                    train_epoch_loss += loss.item()

                # Validation phase
                fold_model.eval()
                val_epoch_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_features, batch_labels in val_fold_loader:
                        batch_features, batch_labels = batch_features.to(
                            device
                        ), batch_labels.to(device)
                        outputs = fold_model(batch_features)
                        loss = criterion_fold(outputs, batch_labels)
                        val_epoch_loss += loss.item()

                        # Accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()

                train_epoch_loss /= len(train_fold_loader)
                val_epoch_loss /= len(val_fold_loader)
                val_accuracy = val_correct / val_total

                scheduler_fold.step()

                # Early stopping
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    epochs_no_improve = 0
                    best_model_state_fold = fold_model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            # Load best fold model
            if best_model_state_fold is not None:
                fold_model.load_state_dict(best_model_state_fold)

            # Evaluate fold model on validation set (final metric for this fold)
            fold_model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_fold_loader:
                    batch_features, batch_labels = batch_features.to(
                        device
                    ), batch_labels.to(device)
                    outputs = fold_model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            fold_val_accuracy = val_correct / val_total
            cv_accuracies.append(fold_val_accuracy)
            logging.info(
                f"Fold {fold_idx+1}/{self.config.cv_folds} Accuracy: {fold_val_accuracy:.4f}"
            )

        # Aggregate CV results
        cv_mean = float(np.mean(cv_accuracies))
        cv_std = float(np.std(cv_accuracies))

        # Now train final model on the full training (X_train_full, y_train_full) + val split (X_val_full, y_val_full)
        class_weights = compute_class_weights(y_train_full, classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            device
        )

        X_train_tensor = torch.tensor(X_train_full.astype("float32"))
        y_train_tensor = torch.tensor(y_train_full.astype("int64"))
        X_val_tensor = torch.tensor(X_val_full.astype("float32"))
        y_val_tensor = torch.tensor(y_val_full.astype("int64"))
        X_test_tensor = torch.tensor(X_test.astype("float32"))
        y_test_tensor = torch.tensor(y_test.astype("int64"))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        # Final model training
        input_dim = X_train_full.shape[1]
        model = nn.Linear(input_dim, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_scheduler_step_size,
            gamma=self.config.lr_scheduler_gamma,
        )

        num_epochs = self.config.num_epochs
        patience = self.config.patience
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_epoch_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

            # Validation phase
            model.eval()
            val_epoch_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features, batch_labels = batch_features.to(
                        device
                    ), batch_labels.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_epoch_loss += loss.item()

            train_epoch_loss /= len(train_loader)
            val_epoch_loss /= len(val_loader)

            scheduler.step()

            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_epoch_loss:.4f}, "
                f"Val Loss: {val_epoch_loss:.4f}, "
                f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Early stopping
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info("Early stopping triggered.")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            logging.warning("No improvement during training.")

        # Evaluation on test set
        model.eval()
        y_pred = []
        y_prob = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(F.softmax(outputs, dim=1).cpu().numpy())

        y_test = y_test_tensor.cpu().numpy()
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Compute metrics on the held-out test set
        print("Computing metrics...")
        metrics = self.compute_metrics(y_test, y_pred, y_prob, classes, class_labels)

        # logging best accuracy
        logging.info(
            f"Linear Probe Best Accuracy on Test Set: {metrics['accuracy']:.4f}"
        )

        # Get feature importance (coefficients)
        if n_classes == 2:
            coefficients = [model.weight[0].cpu().detach().numpy().tolist()]
        else:
            coefficients = [
                model.weight[i].cpu().detach().numpy().tolist()
                for i in range(n_classes)
            ]

        # =========================
        # SHAP Integration
        # =========================

        # Initialize empty list as default
        shap_values_list = []

        # Add compute_shap parameter with default False
        if compute_shap:
            # We'll take a small subset of the training data as background
            num_background = min(100, X_train_tensor.shape[0])
            background_data = X_train_tensor[:num_background].cpu().numpy()

            # Define prediction function for SHAP
            def model_predict(x: np.ndarray) -> np.ndarray:
                model.eval()
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = model(x_tensor)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()
                return probs

            # Create SHAP explainer
            # Increase max_evals to a number >= 2 * num_features + 1
            num_features = X_train_full.shape[1]
            required_min_evals = 2 * num_features + 1
            explainer = shap.Explainer(
                model_predict,
                background_data,
                algorithm="permutation",
                max_evals=required_min_evals,
            )

            # Compute SHAP values for the test set
            X_test_np = X_test_tensor.cpu().numpy()
            shap_values = explainer(X_test_np)

            # Convert SHAP values to list
            shap_values_list = shap_values.values.tolist()

        return {
            "model": model,
            "accuracy": metrics["accuracy"],
            "classes": classes,
            "class_labels": class_labels,
            "n_classes": n_classes,
            "metrics": metrics,
            "cv_scores": {
                "mean": cv_mean,
                "std": cv_std,
                "scores": cv_accuracies,
            },
            "feature_importance": {"coefficients": coefficients},
            "label_encoder": self.label_encoder,
            "hidden": hidden,
            "shap_values": shap_values_list,
        }

    def train_decision_tree(
        self, df, hidden=True, binarize_value=None
    ) -> Dict[str, Any]:
        """Train an optimized decision tree classifier with validation set using GridSearchCV."""
        logging.info("Training decision tree classifier...")

        # Prepare data
        X = df["hidden_states"] if hidden else df["features"]
        y = df["label"]

        # Convert binarize_value to None if it's a string 'None' or 'none'
        if isinstance(binarize_value, str) and binarize_value.lower() == "none":
            binarize_value = None

        # Apply binarization if required
        if binarize_value is not None:
            # Ensure binarize_value is a float
            try:
                binarize_value = float(binarize_value)
                logging.debug(f"Binarize value converted to float: {binarize_value}")
            except ValueError:
                logging.error(
                    f"Invalid binarize_value: {binarize_value}. Must be a float or None."
                )
                raise ValueError(
                    f"Invalid binarize_value: {binarize_value}. Must be a float or None."
                )

            X = self.binarize_features(X, threshold=binarize_value)

        # Encode labels
        classes = np.unique(y)
        class_labels = self.label_encoder.classes_
        n_classes = len(classes)

        # Split data into training, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        validation_size = self.config.validation_size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=validation_size / (1 - self.config.test_size),
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        # Combine training and validation data
        X_train_val = np.concatenate([X_train, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)

        # Create a PredefinedSplit object
        test_fold = np.concatenate(
            [
                np.full(X_train.shape[0], -1),  # Training samples
                np.zeros(X_val.shape[0]),  # Validation samples
            ]
        )
        ps = PredefinedSplit(test_fold)

        # Compute class weights
        class_weights = compute_class_weights(y_train, classes)
        class_weight_dict = dict(zip(classes, class_weights))

        # Initialize classifier
        clf = DecisionTreeClassifier(
            random_state=self.config.random_state,
            class_weight=class_weight_dict,
        )

        # Use GridSearchCV with the predefined split
        grid_search = GridSearchCV(
            clf,
            param_grid=self.config.tree_params,
            cv=ps,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train_val, y_train_val)

        best_model = grid_search.best_estimator_
        best_val_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Evaluate the best model on the test set
        y_test_pred = best_model.predict(X_test)
        y_test_prob = best_model.predict_proba(X_test)

        metrics = self.compute_metrics(
            y_test, y_test_pred, y_test_prob, classes, class_labels
        )

        # Cross-validation scores on combined training and validation set
        stratified_kfold = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        cv_scores = cross_val_score(
            best_model, X_train_val, y_train_val, cv=stratified_kfold
        )

        results = {
            "model": best_model,
            "best_params": best_params,
            "cv_scores": {
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "scores": cv_scores.tolist(),
            },
            "metrics": metrics,
            "feature_importance": {
                "importance": best_model.feature_importances_.tolist()
            },
            "n_classes": n_classes,
            "classes": class_labels.tolist(),
            "label_encoder": self.label_encoder,
            "hidden": hidden,
        }

        logging.info(f"Decision Tree Best Validation Accuracy: {best_val_score:.4f}")
        logging.info(f"Decision Tree Test Accuracy: {metrics['accuracy']:.4f}")
        return results

    def cluster_and_save_embeddings(
        self,
        npz,
        output_dir,
        scaling_method="standard",  # 'standard' or 'minmax'
    ):
        """Perform UMAP and PCA on features and hidden states with feature-wise normalization and save embeddings."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data from npz
        features = npz["features"]
        labels = npz["label"]
        hidden_states = npz["hidden_states"]

        # Feature-wise normalization
        if scaling_method == "standard":
            scaler = StandardScaler()  # Standardize each feature to mean 0, variance 1
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()  # Scale each feature to range [0, 1]
        else:
            raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")

        features_scaled = scaler.fit_transform(features)
        hidden_states_scaled = scaler.fit_transform(hidden_states)

        # Dimensionality reduction using UMAP
        reducer = umap.UMAP(n_components=2, random_state=self.config.random_state)

        # Apply UMAP to scaled features and hidden states
        umap_features = reducer.fit_transform(features_scaled)
        umap_hidden_states = reducer.fit_transform(hidden_states_scaled)

        # Dimensionality reduction using PCA
        pca_features = PCA(n_components=2).fit_transform(features_scaled)
        pca_hidden_states = PCA(n_components=2).fit_transform(hidden_states_scaled)

        # Generate and save UMAP plots for features colored by true labels
        plt.figure(figsize=(8, 6))
        plt.scatter(
            umap_features[:, 0],
            umap_features[:, 1],
            c=labels,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        plt.title("UMAP on Features (Colored by True Labels)")
        plt.colorbar(label="True Label")
        plot_path_umap_features = output_dir / "umap_features_true_labels.png"
        plt.savefig(plot_path_umap_features)
        plt.close()

        # Generate and save UMAP plots for hidden states colored by true labels
        plt.figure(figsize=(8, 6))
        plt.scatter(
            umap_hidden_states[:, 0],
            umap_hidden_states[:, 1],
            c=labels,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        plt.title("UMAP on Hidden States (Colored by True Labels)")
        plt.colorbar(label="True Label")
        plot_path_umap_hidden_states = output_dir / "umap_hidden_states_true_labels.png"
        plt.savefig(plot_path_umap_hidden_states)
        plt.close()

        # Generate and save PCA plots for features colored by true labels
        plt.figure(figsize=(8, 6))
        plt.scatter(
            pca_features[:, 0],
            pca_features[:, 1],
            c=labels,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        plt.title("PCA on Features (Colored by True Labels)")
        plt.colorbar(label="True Label")
        plot_path_pca_features = output_dir / "pca_features_true_labels.png"
        plt.savefig(plot_path_pca_features)
        plt.close()

        # Generate and save PCA plots for hidden states colored by true labels
        plt.figure(figsize=(8, 6))
        plt.scatter(
            pca_hidden_states[:, 0],
            pca_hidden_states[:, 1],
            c=labels,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        plt.title("PCA on Hidden States (Colored by True Labels)")
        plt.colorbar(label="True Label")
        plot_path_pca_hidden_states = output_dir / "pca_hidden_states_true_labels.png"
        plt.savefig(plot_path_pca_hidden_states)
        plt.close()

        logging.info(f"Saved UMAP and PCA plots colored by true labels to {output_dir}")

        # save embeddings
        embeddings_path = output_dir / "embeddings_umap_pca.npz"
        np.savez(
            embeddings_path,
            umap_features=umap_features,
            pca_features=pca_features,
            umap_hidden_states=umap_hidden_states,
            pca_hidden_states=pca_hidden_states,
            labels=labels,
        )
        logging.info(f"Saved UMAP and PCA embeddings to {embeddings_path}")

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        model_name: str,
        layer: str,
        model_type: str,
        hidden: bool = True,
    ):
        """Save model results and artifacts."""

        # Create a timestamp for unique directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Append timestamp to the output directory
        output_dir = (
            Path(output_dir) / model_name / f"layer_{layer}" / model_type / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model and label encoder together
        model_artifacts = {
            "model": results["model"],
            "label_encoder": results["label_encoder"],
        }
        model_path = output_dir / f"{model_type}_{hidden}_model.joblib"
        joblib.dump(model_artifacts, model_path)

        # Remove non-serializable objects for JSON
        results_copy = results.copy()
        results_copy.pop("model")
        results_copy.pop("label_encoder")

        # Final check to convert -> serializable
        results_copy = convert_to_serializable(results_copy)

        # print keys
        print(results_copy.keys())

        # Save metrics in JSON format
        metrics_path = output_dir / f"{model_type}_{hidden}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results_copy, f, indent=2)

        logging.info(f"Saved model and metrics to {output_dir}")


def compute_class_weights(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Compute class weights inversely proportional to class frequencies."""
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return class_weights


# def run_training_pipeline(
#     df: pd.DataFrame,
#     config: TrainingConfig,
#     output_dir: Path,
#     model_name: str,
#     layer: str,
# ):
#     """Run the complete training pipeline."""
#     trainer = ModelTrainer(config)

#     linear_results = trainer.train_linear_probe(df)
#     trainer.save_results(linear_results, output_dir, model_name, layer, "linear_probe")

#     tree_results = trainer.train_decision_tree(df)
#     trainer.save_results(tree_results, output_dir, model_name, layer, "decision_tree")

#     return linear_results, tree_results


def train_bow_baseline(
    dataset_name: str,
    dataset_split: str,
    text_field: str,
    label_field: str,
    config_name: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    stop_words: Union[str, None] = "english",
    use_balanced_class_weight: bool = True,
):
    """
    Train and evaluate a baseline classifier using TF-IDF features on a dataset
    loaded from the Hugging Face Hub.

    Parameters
    ----------
    dataset_name : str
        Name of the Hugging Face dataset to load.
    dataset_split : str
        The split of the dataset to load (e.g., "train", "test").
    text_field : str
        The name of the column containing the text data.
    label_field : str
        The name of the column containing the label.
    config_name : Optional[str], default=None
        Specific configuration name of the dataset if needed.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=10000
        Maximum number of iterations for the Logistic Regression solver.
    ngram_range : Tuple[int, int], default=(1, 2)
        The lower and upper boundary of the n-grams considered in TF-IDF vectorization.
    max_features : int, default=20000
        Maximum number of features to consider in TF-IDF.
    stop_words : Union[str, None], default="english"
        Stop words to use for TF-IDF vectorization. Use "english", None, or a custom list.
    use_balanced_class_weight : bool, default=True
        Whether to use balanced class weights in the Logistic Regression.

    Returns
    -------
    dict
        {
            "model": Trained LogisticRegression model,
            "vectorizer": Fitted TfidfVectorizer,
            "metrics": dict with keys: "accuracy", "classification_report",
                      "classes", "n_classes", "roc_auc", "roc_curve"
        }

    Notes
    -----
    This function uses a simple logistic regression classifier on top of TF-IDF vectorized text.
    It's intended as a baseline or linear probe.
    """

    # Load dataset from Hugging Face
    ds = load_dataset(dataset_name, split=dataset_split, name=config_name)
    df = ds.to_pandas()

    # Validate columns
    if text_field not in df.columns:
        raise ValueError(
            f"The specified text_field '{text_field}' does not exist in the dataset."
        )
    if label_field not in df.columns:
        raise ValueError(
            f"The specified label_field '{label_field}' does not exist in the dataset."
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Extract text and labels
    texts = df[text_field].astype(str).tolist()
    labels = df[label_field].values

    # Ensure we have at least two samples and preferably multiple classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        warnings.warn(
            "Only one class present in the data. The model will not generalize."
        )
    if len(texts) < 2:
        raise ValueError("Not enough data samples.")

    # Attempt train/test split
    try:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as e:
        raise ValueError(f"Failed to split dataset: {e}")

    # Check for classes again after splitting (in rare cases stratify might fail)
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)
    if len(train_classes) < 2:
        warnings.warn("Training set contains only one class.")
    if len(test_classes) < 2:
        warnings.warn("Test set contains only one class. Metrics may be degenerate.")

    # Map classes to integers for binary tasks
    if len(unique_labels) == 2:
        class_to_num = {cls: i for i, cls in enumerate(unique_labels)}
        y_train = np.array([class_to_num[yt] for yt in y_train])
        y_test = np.array([class_to_num[yt] for yt in y_test])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range, max_features=max_features, stop_words=stop_words
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Logistic Regression setup
    class_weight = "balanced" if use_balanced_class_weight else None
    clf = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight,
        # Use 'saga' solver if L1 regularization or other advanced methods are needed
        # Otherwise 'lbfgs' is a good default.
        solver="lbfgs",
    )

    # Train model
    clf.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)

    # Convert back to original label space if binary
    if len(unique_labels) == 2:
        inv_class_map = {v: k for k, v in class_to_num.items()}
        y_true_labels = [inv_class_map[yt] for yt in y_test]
        y_pred_labels = [inv_class_map[yp] for yp in y_pred]
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        classes = unique_labels
    else:
        # Multiclass
        classes = unique_labels
        report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "classes": classes.tolist(),
        "n_classes": len(classes),
    }

    # Compute ROC AUC
    # For binary
    if len(classes) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            metrics["roc_auc"] = auc(fpr, tpr)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        except ValueError as e:
            # This can happen if only one class is present in y_test
            metrics["roc_auc"] = None
            metrics["roc_curve"] = None
            warnings.warn(f"ROC computation failed: {e}")
    else:
        # Multiclass case
        try:
            Y_test_bin = label_binarize(y_test, classes=range(len(classes)))
            roc_auc_dict = {}
            roc_curve_dict = {}
            for i, c in enumerate(classes):
                fpr_c, tpr_c, _ = roc_curve(Y_test_bin[:, i], y_prob[:, i])
                roc_auc_dict[c] = auc(fpr_c, tpr_c)
                roc_curve_dict[c] = {"fpr": fpr_c.tolist(), "tpr": tpr_c.tolist()}
            metrics["roc_auc"] = roc_auc_dict
            metrics["roc_curve"] = roc_curve_dict

            # Micro-average
            fpr_micro, tpr_micro, _ = roc_curve(Y_test_bin.ravel(), y_prob.ravel())
            metrics["roc_auc"]["micro"] = auc(fpr_micro, tpr_micro)
            metrics["roc_curve"]["micro"] = {
                "fpr": fpr_micro.tolist(),
                "tpr": tpr_micro.tolist(),
            }

            # Macro-average
            metrics["roc_auc"]["macro"] = roc_auc_score(
                Y_test_bin, y_prob, average="macro", multi_class="ovr"
            )
        except ValueError as e:
            # In case ROC can't be computed due to class issues
            metrics["roc_auc"] = None
            metrics["roc_curve"] = None
            warnings.warn(f"ROC computation failed for multiclass: {e}")

    return {"model": clf, "vectorizer": vectorizer, "metrics": metrics}
