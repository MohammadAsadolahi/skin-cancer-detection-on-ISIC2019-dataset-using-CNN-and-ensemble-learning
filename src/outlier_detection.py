"""
outlier_detection.py — Local Outlier Factor-based Data Cleaning.

Removes anomalous samples before training to improve model robustness.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.config import OutlierConfig


def remove_outliers(
    X: np.ndarray,
    y: np.ndarray,
    cfg: OutlierConfig = OutlierConfig(),
):
    """
    Detect and remove outliers using Local Outlier Factor.

    Parameters
    ----------
    X : np.ndarray — Image data, shape (N, H, W, C)
    y : np.ndarray — Labels, shape (N,)

    Returns
    -------
    X_clean, y_clean, n_removed
    """
    lof = LocalOutlierFactor(
        n_neighbors=cfg.n_neighbors,
        contamination=cfg.contamination,
    )
    flat = X.reshape(X.shape[0], -1)
    preds = lof.fit_predict(flat, y)
    outlier_idx = np.where(preds == -1)[0]

    X_clean = np.delete(X, outlier_idx, axis=0)
    y_clean = np.delete(y, outlier_idx, axis=0)

    print(f"[outlier] Removed {len(outlier_idx)} outliers "
          f"({len(outlier_idx)/len(y)*100:.1f}%). "
          f"Remaining: {len(y_clean)} samples.")
    return X_clean, y_clean, len(outlier_idx)
