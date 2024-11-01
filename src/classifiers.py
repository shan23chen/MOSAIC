from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
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
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
import joblib
from typing import Dict, List, Tuple, Any


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_iter: int = 5000

    # Common parameters for both binary and multiclass
    probe_params = {
        "C": [0.001, 0.01, 0.1, 1.0],
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

    # def prepare_hidden_states(
    #     self, hidden_states: List[np.ndarray], max_length: int = None
    # ) -> np.ndarray:
    #     """Prepare hidden states with padding and normalization."""
    #     if max_length is None:
    #         max_length = max(state.shape[0] for state in hidden_states)

    #     hidden_dim = hidden_states[0].shape[1]
    #     n_samples = len(hidden_states)
    #     padded_states = np.zeros((n_samples, max_length, hidden_dim))

    #     for i, state in enumerate(hidden_states):
    #         seq_len = min(state.shape[0], max_length)
    #         padded_states[i, :seq_len] = state[:seq_len]

    #     # Reshape and normalize
    #     reshaped = padded_states.reshape(n_samples, -1)
    #     normalized = self.scaler.fit_transform(reshaped)

    #     return normalized

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

    def train_linear_probe(self, df, hidden=True) -> Dict[str, Any]:
        """Train an optimized linear probe classifier with proper binary/multiclass handling."""
        logging.info("Training linear probe classifier...")

        if hidden:
            # Prepare data
            X = df["hidden_states"]
        else:
            X = df["features"]

        # Encode labels
        y = df["label"]
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

        # Create base classifier with common parameters
        base_classifier = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )

        # For multiclass, wrap with OneVsRestClassifier
        if n_classes > 2:
            classifier = OneVsRestClassifier(base_classifier)
        else:
            classifier = base_classifier

        # Grid search
        grid_search = GridSearchCV(
            classifier,
            self.config.probe_params,
            cv=self.config.cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            grid_search.fit(X_train, y_train)

        # Get predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob, classes, class_labels)
        cv_scores = cross_val_score(
            best_model, X_train, y_train, cv=self.config.cv_folds
        )

        # Get feature importance
        if n_classes == 2:
            coefficients = [best_model.coef_[0].tolist()]
        else:
            coefficients = [coef.tolist() for coef in best_model.estimators_[0].coef_]

        results = {
            "model": best_model,
            "best_params": grid_search.best_params_,
            "cv_scores": {
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "scores": cv_scores.tolist(),
            },
            "metrics": metrics,
            "feature_importance": {"coefficients": coefficients},
            "n_classes": n_classes,
            "classes": class_labels.tolist(),
            "label_encoder": self.label_encoder,
            "hidden": hidden,
        }

        logging.info(f"Linear Probe Best Accuracy: {metrics['accuracy']:.4f}")
        return results

    def train_decision_tree(self, df, hidden=True) -> Dict[str, Any]:
        """Train an optimized decision tree classifier."""
        logging.info("Training decision tree classifier...")

        if hidden:
            X = df["hidden_states"]
        else:
            X = df["features"]

        # Encode labels
        y = df["label"]
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
        output_dir = Path(output_dir)
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
