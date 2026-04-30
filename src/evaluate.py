"""
evaluate.py — Model Evaluation & Reporting Utilities.

Generates classification reports, confusion matrices, and ROC curves
for both CNN base learners and the ensemble meta-learner.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from tensorflow.keras.models import load_model

from src.config import CATEGORY_LABELS, CATEGORY_FULL_NAMES, NUM_CLASSES


def print_classification_report(y_true, y_pred, title="Classification Report"):
    """Print a formatted sklearn classification report."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(classification_report(
        y_true, y_pred,
        target_names=CATEGORY_LABELS,
        digits=4,
    ))


def compute_confusion_matrix(y_true, y_pred, normalize=True):
    """Compute confusion matrix, optionally row-normalized."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm


def compute_per_class_roc(y_true, y_score):
    """
    Compute per-class ROC curves and AUC values.

    Parameters
    ----------
    y_true  : np.ndarray — integer labels, shape (N,)
    y_score : np.ndarray — softmax probabilities, shape (N, 8)

    Returns
    -------
    fpr_dict, tpr_dict, auc_dict — keyed by class index
    """
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        binary_true = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_true, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a Keras model and print metrics."""
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n[evaluate] {model_name} — Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    print_classification_report(y_test, y_pred, title=f"{model_name} Report")
    return y_pred, acc
