import cupy as cp
import cudf
from cuml.preprocessing import StandardScaler, LabelEncoder
from cuml.model_selection import train_test_split
from cuml.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import joblib
from dataclasses import dataclass
from typing import Dict, Any
from itertools import product
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score
)
from sklearn.model_selection import KFold

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    test_size: float = 0.9
    random_state: int = 42
    cv_folds: int = 5
    max_iter: int = 5000
    # Common parameters for both binary and multiclass
    probe_params = {
        "C": [0.001, 0.01, 0.1, 1.0, 3.0],
        "penalty": ["l2"],
        "solver": ["qn"],  # 'qn' is equivalent to 'lbfgs' in cuML
    }

class ModelTrainer:
    """Class to handle model training and evaluation with cuML and sklearn."""

    def __init__(self, config: TrainingConfig, label_encoder: LabelEncoder = None):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = label_encoder or LabelEncoder()

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        classes: np.ndarray,
        class_labels: list,
    ) -> Dict[str, Any]:
        """Compute metrics using NumPy arrays and scikit-learn functions."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
        }

        if len(classes) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics["roc_auc"] = auc(fpr, tpr)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        else:
            # Multiclass classification
            # Ensure classes are NumPy arrays
            classes_np = np.array(classes)

            # Use sklearn's label_binarize function
            y_true_bin = label_binarize(y_true, classes=classes_np)

            metrics["roc_auc"] = {}
            metrics["roc_curve"] = {}

            for i, class_label in enumerate(class_labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                metrics["roc_auc"][class_label] = auc(fpr, tpr)
                metrics["roc_curve"][class_label] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }

            # Compute micro and macro ROC AUC scores
            fpr_micro, tpr_micro, _ = roc_curve(
                y_true_bin.ravel(), y_prob.ravel()
            )
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
        """Train an optimized linear probe classifier with cuML."""
        logging.info("Training linear probe classifier with cuML...")

        # Extract features based on 'hidden' flag
        if hidden:
            X = cp.vstack(df["hidden_states"].to_pandas().values)
        else:
            X = cp.vstack(df["features"].to_pandas().values)

        # Encode labels with cuML's LabelEncoder
        y = self.label_encoder.fit_transform(df["label"])
        classes = cp.unique(y)
        class_labels = self.label_encoder.classes_.to_pandas().tolist()
        n_classes = len(classes)

        logging.info(f"Number of classes: {n_classes}")
        logging.info(f"Classes: {class_labels}")

        # Split data with cuML's train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Grid search for best hyperparameters
        best_score, best_params, best_model = -1, None, None
        param_combinations = list(product(
            self.config.probe_params["C"],
            self.config.probe_params["penalty"],
            self.config.probe_params["solver"]
        ))

        for C, penalty, solver in param_combinations:
            clf = LogisticRegression(
                penalty=penalty, solver=solver, C=C, max_iter=self.config.max_iter
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test).get()
            # print(type(y_pred), type(y_test))
            score = accuracy_score(y_test.to_numpy(), y_pred)
            print(f"Accuracy: {score:.4f}, C: {C}, penalty: {penalty}, solver: {solver}")
            f1 = f1_score(y_test.to_numpy(), y_pred, average='weighted')
            print(f"F1 Score: {f1:.4f}")
            macro_f1 = f1_score(y_test.to_numpy(), y_pred, average='macro')
            print(f"Macro F1 Score: {macro_f1:.4f}")

            if score > best_score:
                best_score, best_params, best_model = score, {'C': C, 'penalty': penalty, 'solver': solver}, clf

        return macro_f1 
    
    def train_linear_probe_hidden(self, X_train, y_train, X_test, y_test) -> float:
        """Train an optimized linear probe classifier with cuML using provided datasets."""
        logging.info("Training linear probe classifier with cuML...")

        # Encode labels with cuML's LabelEncoder
        y_train = self.label_encoder.fit_transform(y_train)
        y_test = self.label_encoder.transform(y_test)

        classes = cp.unique(y_train)
        class_labels = self.label_encoder.classes_.to_pandas().tolist()
        n_classes = len(classes)

        logging.info(f"Number of classes: {n_classes}")
        logging.info(f"Classes: {class_labels}")

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Grid search for best hyperparameters
        best_score, best_params, best_model = -1, None, None
        param_combinations = list(product(
            self.config.probe_params["C"],
            self.config.probe_params["penalty"],
            self.config.probe_params["solver"]
        ))

        for C, penalty, solver in param_combinations:
            clf = LogisticRegression(
                penalty=penalty, solver=solver, C=C, max_iter=self.config.max_iter
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test).get()
            score = accuracy_score(y_test.to_numpy(), y_pred)
            print(f"Accuracy: {score:.4f}, C: {C}, penalty: {penalty}, solver: {solver}")
            f1 = f1_score(y_test.to_numpy(), y_pred, average='weighted')
            print(f"F1 Score: {f1:.4f}")
            macro_f1 = f1_score(y_test.to_numpy(), y_pred, average='macro')
            print(f"Macro F1 Score: {macro_f1:.4f}")

            if score > best_score:
                best_score, best_params, best_model = score, {'C': C, 'penalty': penalty, 'solver': solver}, clf

        return macro_f1
