"""
ensemble.py — XGBoost Stacking Meta-Learner over CNN Base Models.

Implements a two-level stacking ensemble:
  Level 0 : K independently trained CNN models (base learners)
  Level 1 : XGBClassifier trained on concatenated CNN softmax outputs
"""

import datetime
from typing import List

import numpy as np
from numpy import dstack
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from src.config import EnsembleConfig


def _stacked_dataset(members: list, X: np.ndarray) -> np.ndarray:
    """Produce the stacked feature matrix from base-model predictions."""
    stack = None
    for model in members:
        preds = model.predict(X, verbose=0)
        stack = preds if stack is None else dstack((stack, preds))
    return stack.reshape(stack.shape[0], -1)


def fit_stacked_model(
    members: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: EnsembleConfig = EnsembleConfig(),
):
    """Train the XGBoost meta-learner on stacked CNN predictions."""
    stacked_X = _stacked_dataset(members, X_train)
    meta_model = XGBClassifier(**cfg.xgb_params)
    t0 = datetime.datetime.now()
    meta_model.fit(stacked_X, y_train)
    training_time = datetime.datetime.now() - t0
    return meta_model, training_time


def stacked_prediction(
    members: list, meta_model, X: np.ndarray
) -> np.ndarray:
    """Generate predictions through the full stacking pipeline."""
    stacked_X = _stacked_dataset(members, X)
    return meta_model.predict(stacked_X)


def load_base_models(n_models: int = 2) -> list:
    """Load saved CNN fold models from disk."""
    models = []
    for i in range(1, n_models + 1):
        path = f"./CNN_fold{i}.h5"
        models.append(load_model(path))
        print(f"[ensemble] Loaded {path}")
    return models


def evaluate_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    members: list = None,
    n_models: int = 2,
):
    """End-to-end ensemble training + evaluation. Returns accuracy & predictions."""
    if members is None:
        members = load_base_models(n_models)

    meta_model, train_time = fit_stacked_model(members, X_train, y_train)
    y_pred = stacked_prediction(members, meta_model, X_test)
    acc = accuracy_score(y_test, y_pred)
    print(
        f"[ensemble] XGBoost meta-learner accuracy: {acc:.4f}  (trained in {train_time})")
    return acc, y_pred, meta_model
