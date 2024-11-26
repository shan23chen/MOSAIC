from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import (
    StandardScaler,
    label_binarize,
    LabelEncoder,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
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
from typing import Dict, List, Tuple, Any
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


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_iter: int = 5000
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 5000
    patience: int = 10
    lr_scheduler_step_size: int = 100  # Period of learning rate decay
    lr_scheduler_gamma: float = 0.1  # Multiplicative factor of learning rate decay

    # Common parameters for both binary and multiclass
    probe_params = {
        "C": [0.0001, 0.001, 0.01, 0.1],
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
        self, df, hidden=True, binarize_value=None
    ) -> Dict[str, Any]:
        """Train an optimized linear probe classifier with proper binary/multiclass handling."""
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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.astype("float32"))
        y_train_tensor = torch.tensor(y_train.astype("int64"))
        X_test_tensor = torch.tensor(X_test.astype("float32"))
        y_test_tensor = torch.tensor(y_test.astype("int64"))

        # Create datasets and data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the linear probe model
        input_dim = X_train.shape[1]
        model = nn.Linear(input_dim, n_classes).to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_scheduler_step_size,
            gamma=self.config.lr_scheduler_gamma,
        )

        # Training loop with early stopping
        num_epochs = self.config.num_epochs
        patience = self.config.patience
        best_loss = float("inf")
        epochs_no_improve = 0
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)

                # Forward pass
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Step the learning rate scheduler
            scheduler.step()

            epoch_loss /= len(train_loader)
            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info("Early stopping triggered.")
                    break

        # Load the best model
        model.load_state_dict(best_model_state)

        # Evaluation
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

        # Compute metrics
        print("Computing metrics...")
        metrics = self.compute_metrics(y_test, y_pred, y_prob, classes, class_labels)

        # logging best accuracy
        logging.info(f"Linear Probe Best Accuracy: {metrics['accuracy']:.4f}")

        # Cross-validation scores
        print("Computing cross-validation scores...")
        cv_scores = cross_val_score(
            LogisticRegression(
                random_state=self.config.random_state, max_iter=self.config.max_iter
            ),
            X_train,
            y_train,
            cv=self.config.cv_folds,
        )

        # Get feature importance (coefficients)
        if n_classes == 2:
            coefficients = [model.weight[0].cpu().detach().numpy().tolist()]
        else:
            coefficients = [
                model.weight[i].cpu().detach().numpy().tolist()
                for i in range(n_classes)
            ]

        # Return results
        return {
            "model": model,
            "accuracy": metrics["accuracy"],
            "classes": classes,
            "class_labels": class_labels,
            "n_classes": n_classes,
            "metrics": metrics,
            "cv_scores": {
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "scores": cv_scores.tolist(),
            },
            "feature_importance": {"coefficients": coefficients},
            "label_encoder": self.label_encoder,
            "hidden": hidden,
        }

    def train_decision_tree(
        self, df, hidden=True, binarize_value=None
    ) -> Dict[str, Any]:
        """Train an optimized decision tree classifier."""
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

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        clf = DecisionTreeClassifier(random_state=self.config.random_state)
        grid_search = GridSearchCV(
            clf,
            self.config.tree_params,
            cv=self.config.cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        metrics = self.compute_metrics(y_test, y_pred, y_prob, classes, class_labels)
        cv_scores = cross_val_score(
            best_model, X_train, y_train, cv=self.config.cv_folds
        )

        results = {
            "model": best_model,
            "best_params": grid_search.best_params_,
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

        logging.info(f"Decision Tree Best Accuracy: {metrics['accuracy']:.4f}")
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

        # Save metrics in JSON format
        metrics_path = output_dir / f"{model_type}_{hidden}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results_copy, f, indent=2)

        logging.info(f"Saved model and metrics to {output_dir}")


def run_training_pipeline(
    df: pd.DataFrame,
    config: TrainingConfig,
    output_dir: Path,
    model_name: str,
    layer: str,
):
    """Run the complete training pipeline."""
    trainer = ModelTrainer(config)

    linear_results = trainer.train_linear_probe(df)
    trainer.save_results(linear_results, output_dir, model_name, layer, "linear_probe")

    tree_results = trainer.train_decision_tree(df)
    trainer.save_results(tree_results, output_dir, model_name, layer, "decision_tree")

    return linear_results, tree_results
