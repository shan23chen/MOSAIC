from sae_lens import SAE
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
from sklearn.preprocessing import StandardScaler, label_binarize
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 2
    max_iter: int = 20

    # Linear probe specific
    linear_probe_params = {
        "C": [0.1, 1.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }

    # Decision tree specific
    tree_params = {
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }


class ModelTrainer:
    """Class to handle model training and evaluation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = StandardScaler()

    def prepare_hidden_states(
        self, hidden_states: List[np.ndarray], max_length: int = None
    ) -> np.ndarray:
        """Prepare hidden states with padding and normalization."""
        if max_length is None:
            max_length = max(state.shape[0] for state in hidden_states)

        hidden_dim = hidden_states[0].shape[1]
        n_samples = len(hidden_states)
        padded_states = np.zeros((n_samples, max_length, hidden_dim))

        for i, state in enumerate(hidden_states):
            seq_len = min(state.shape[0], max_length)
            padded_states[i, :seq_len] = state[:seq_len]

        # Reshape and normalize
        n_samples, seq_len, hidden_dim = padded_states.shape
        reshaped = padded_states.reshape(n_samples, -1)
        normalized = self.scaler.fit_transform(reshaped)

        return normalized

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics with multiclass support."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
        }

        # Add ROC AUC if probabilities are available
        if y_prob is not None:
            # Get unique classes
            classes = np.unique(y_true)
            n_classes = len(classes)

            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                metrics["roc_auc"] = auc(fpr, tpr)
                metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            else:
                # Multiclass classification
                # One-vs-Rest ROC curves
                y_true_bin = label_binarize(y_true, classes=classes)
                metrics["roc_auc"] = {}
                metrics["roc_curve"] = {}

                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    metrics["roc_auc"][str(classes[i])] = auc(fpr, tpr)
                    metrics["roc_curve"][str(classes[i])] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                    }

                # Compute micro-average ROC curve
                fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
                metrics["roc_auc"]["micro"] = auc(fpr_micro, tpr_micro)
                metrics["roc_curve"]["micro"] = {
                    "fpr": fpr_micro.tolist(),
                    "tpr": tpr_micro.tolist(),
                }

                # Compute macro-average ROC AUC
                metrics["roc_auc"]["macro"] = roc_auc_score(
                    y_true_bin, y_prob, average="macro", multi_class="ovr"
                )

        return metrics

    def train_linear_probe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train an optimized linear probe classifier."""
        logging.info("Training linear probe classifier...")

        # Prepare data
        X = self.prepare_hidden_states(list(df["hidden_state"]))
        y = df["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Modify parameters for multiclass
        params = self.config.linear_probe_params.copy()
        if len(np.unique(y)) > 2:
            # For multiclass, we need to use different solvers
            params["solver"] = ["lbfgs", "newton-cg", "sag"]
            params["penalty"] = [
                "l2"
            ]  # Only l2 penalty is supported with these solvers

        # Grid search for best parameters
        clf = LogisticRegression(
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            multi_class="ovr",  # Use one-vs-rest for multiclass
        )

        grid_search = GridSearchCV(
            clf, params, cv=self.config.cv_folds, scoring="accuracy", n_jobs=-1
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob)
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
                "coefficients": [coef.tolist() for coef in best_model.coef_]
            },
        }

        logging.info(f"Linear Probe Best Accuracy: {metrics['accuracy']:.4f}")
        return results

    def train_decision_tree(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train an optimized decision tree classifier."""
        logging.info("Training decision tree classifier...")

        # Prepare data
        X = np.stack(df["features"].values)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Grid search for best parameters
        clf = DecisionTreeClassifier(random_state=self.config.random_state)
        grid_search = GridSearchCV(
            clf,
            self.config.tree_params,
            cv=self.config.cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob)
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
    ):
        """Save model results and artifacts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f"{model_name}_{layer}_{model_type}_model.joblib"
        joblib.dump(results["model"], model_path)

        # Save metrics and results
        results_copy = results.copy()
        results_copy.pop("model")  # Remove model object for JSON serialization

        metrics_path = output_dir / f"{model_name}_{layer}_{model_type}_metrics.json"
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

    # Train linear probe
    linear_results = trainer.train_linear_probe(df)
    trainer.save_results(linear_results, output_dir, model_name, layer, "linear_probe")

    # Train decision tree
    tree_results = trainer.train_decision_tree(df)
    trainer.save_results(tree_results, output_dir, model_name, layer, "decision_tree")

    return linear_results, tree_results
