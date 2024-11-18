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
        # y_pred = best_model.predict(X_test).get()
    #     y_prob = best_model.predict_proba(X_test).get()
    #     y_test_np = y_test.to_numpy()
    #     classes_np = classes.to_numpy()

    #     # Compute metrics using NumPy arrays
    #     metrics = self.compute_metrics(
    #         y_true=y_test_np,
    #         y_pred=y_pred,
    #         y_prob=y_prob,
    #         classes=classes_np,
    #         class_labels=class_labels
    #     )

    #     # Cross-validation
    #     cv_scores = self.cross_validate_model(best_model, X_train, y_train)

    #     # Save feature importance (coefficients) for linear model
    #     coefficients = best_model.coef_.get().tolist()
    #     results = {
    #         "model": best_model,
    #         "best_params": best_params,
    #         "cv_scores": {
    #             "mean": float(np.mean(cv_scores)),
    #             "std": float(np.std(cv_scores)),
    #             "scores": cv_scores,
    #         },
    #         "metrics": metrics,
    #         "feature_importance": {"coefficients": coefficients},
    #         "n_classes": n_classes,
    #         "classes": class_labels,
    #         "label_encoder": self.label_encoder,
    #         "hidden": hidden,
    #     }

    #     logging.info(f"Linear Probe Best Accuracy: {metrics['accuracy']:.4f}")
    #     return results

    # def cross_validate_model(self, model, X_train, y_train):
    #     """Perform cross-validation using sklearn's KFold."""
    #     kfold = KFold(
    #         n_splits=self.config.cv_folds,
    #         shuffle=True,
    #         random_state=self.config.random_state,
    #     )
    #     scores = []
    #     X_train_np = X_train.get()
    #     y_train_np = y_train.get()

    #     for train_indices, val_indices in kfold.split(X_train_np):
    #         X_fold_train, X_fold_val = X_train_np[train_indices], X_train_np[val_indices]
    #         y_fold_train, y_fold_val = y_train_np[train_indices], y_train_np[val_indices]

    #         model.fit(X_fold_train, y_fold_train)
    #         y_pred = model.predict(X_fold_val).get()
    #         score = accuracy_score(y_fold_val, y_pred)
    #         scores.append(score)

    #     return scores

    # def save_results(
    #     self,
    #     results: Dict[str, Any],
    #     output_dir: Path,
    #     model_name: str,
    #     layer: str,
    #     model_type: str,
    #     hidden: bool = True,
    # ):
    #     """Save model results and artifacts."""
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     # Save model using joblib
    #     model_path = output_dir / f"{model_type}_{hidden}_model.joblib"
    #     joblib.dump(results["model"], model_path)

    #     # Save label encoder
    #     encoder_path = output_dir / f"{model_type}_{hidden}_label_encoder.joblib"
    #     joblib.dump(results["label_encoder"], encoder_path)

    #     # Remove non-serializable objects for JSON
    #     results_copy = results.copy()
    #     results_copy.pop("model")
    #     results_copy.pop("label_encoder")

    #     metrics_path = output_dir / f"{model_type}_{hidden}_metrics.json"
    #     with open(metrics_path, "w") as f:
    #         json.dump(results_copy, f, indent=2)

    #     logging.info(f"Saved model and metrics to {output_dir}")

# # Example usage:
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     # Load your data from npz files or other sources
#     # For demonstration, create a dummy DataFrame
#     num_samples, num_features = 1000, 128
#     X_np = np.random.rand(num_samples, num_features)
#     y_np = np.random.randint(0, 2, num_samples)

#     # Convert to pandas DataFrame
#     df = pd.DataFrame({"hidden_states": list(X_np), "label": y_np})

#     # Define training configuration
#     config = TrainingConfig()

#     # Define output directory, model name, and layer
#     output_dir = Path("./model_output")
#     model_name, layer = "my_model", "layer_1"

#     # Run training pipeline
#     trainer = ModelTrainer(config)
#     df_cudf = cudf.from_pandas(df)
#     results = trainer.train_linear_probe(df_cudf, hidden=True)
#     trainer.save_results(results, output_dir, model_name, layer, "linear_probe")