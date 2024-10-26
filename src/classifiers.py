import numpy as np
import pandas as pd
from sae_lens import SAE
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def prepare_hidden_states_for_linear_probe(hidden_states, max_length=None):
    """Prepare hidden states for linear probe by handling padding."""
    if max_length is None:
        max_length = max(state.shape[0] for state in hidden_states)

    hidden_dim = hidden_states[0].shape[1]
    n_samples = len(hidden_states)
    padded_states = np.zeros((n_samples, max_length, hidden_dim))

    for i, state in enumerate(hidden_states):
        seq_len = min(state.shape[0], max_length)
        padded_states[i, :seq_len] = state[:seq_len]

    return padded_states


def train_linear_probe(df, test_size=0.2, random_state=42):
    """Train a linear probe classifier using hidden states."""
    logging.info("Training linear probe classifier...")

    # Prepare hidden states
    hidden_states = list(df["hidden_state"])
    padded_states = prepare_hidden_states_for_linear_probe(hidden_states)

    # Reshape to 2D
    n_samples, seq_len, hidden_dim = padded_states.shape
    X = padded_states.reshape(n_samples, -1)
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    results = {
        "model": clf,
        "predictions": y_pred,
        "y_test": y_test,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classes": clf.classes_,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    logging.info(f"\nLinear Probe Accuracy: {accuracy:.4f}")

    return results


def train_decision_tree(df, max_depth=5, test_size=0.2, random_state=42):
    """Train a decision tree classifier using feature vectors."""
    logging.info("Training decision tree classifier...")

    X = np.stack(df["features"].values)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    results = {
        "model": clf,
        "predictions": y_pred,
        "y_test": y_test,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classes": clf.classes_,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    logging.info(f"\nDecision Tree Accuracy: {accuracy:.4f}")

    return results


# In your main function, modify the dashboard_data creation:
def get_model_info(model, feature_names):
    """Extract relevant information from sklearn model"""
    info = {}

    # For linear model coefficients
    if hasattr(model, "coef_"):
        # For binary classification
        if len(model.coef_.shape) == 2:
            info["coefficients"] = model.coef_[0].tolist()
        # For single output
        else:
            info["coefficients"] = model.coef_.tolist()

    # For tree feature importances
    if hasattr(model, "feature_importances_"):
        info["feature_importances"] = model.feature_importances_.tolist()

    # For decision tree structure
    if isinstance(model, DecisionTreeClassifier):
        tree = model.tree_
        info["tree_structure"] = {
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "n_node_samples": tree.n_node_samples.tolist(),
            "impurity": tree.impurity.tolist(),
            "value": tree.value.tolist(),
        }

    return info


def save_classification_results(
    results, output_dir, model_name, layer, classifier_type
):
    """Save classification results and plots."""
    results_dir = Path(output_dir) / f"{model_name}_{layer}_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save normal results to NPY file
    normal_results = {
        key: value
        for key, value in results.items()
        if key
        not in ["model", "confusion_matrix", "accuracy", "precision", "recall", "f1"]
    }
    normal_results_path = results_dir / f"{classifier_type}_normal_results.npy"
    np.save(normal_results_path, normal_results)

    # Save metrics to their own NPY file
    metrics = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
    }
    metrics_path = results_dir / f"{classifier_type}_metrics.npy"
    np.save(metrics_path, metrics)

    # Save confusion matrix
    cm_path = results_dir / f"{classifier_type}_confusion_matrix.npy"
    np.save(cm_path, results["confusion_matrix"])

    # Save classes
    classes_path = results_dir / f"{classifier_type}_classes.npy"
    np.save(classes_path, results["classes"])


def plot_decision_tree(tree_results, feature_names, class_names, output_path):
    """Plot and save decision tree visualization."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_results["model"],
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
    )
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved decision tree plot to {output_path}")
