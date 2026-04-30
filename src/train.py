"""
train.py — Nested K-Fold Cross-Validation Training Pipeline.

Orchestrates the full training workflow:
  1. Load & preprocess data
  2. Remove outliers via LOF
  3. Outer 10-fold CV for unbiased evaluation
  4. Inner 2-fold CV to train CNN base learners
  5. XGBoost meta-learner stacking over CNN outputs
"""

import numpy as np
from sklearn.model_selection import KFold

from src.config import TrainingConfig, CNNConfig
from src.model import build_cnn
from src.ensemble import evaluate_ensemble


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    train_cfg: TrainingConfig = TrainingConfig(),
    cnn_cfg: CNNConfig = CNNConfig(),
):
    """
    Execute the full nested cross-validation pipeline.

    Returns
    -------
    fold_results : list of dict with keys {fold, cnn_acc, ensemble_acc}
    """
    outer_kfold = KFold(
        n_splits=train_cfg.outer_folds,
        shuffle=True,
        random_state=train_cfg.random_state,
    )
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_kfold.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"  OUTER FOLD {fold_idx}/{train_cfg.outer_folds}")
        print(f"{'='*60}")

        X_outer_train, y_outer_train = X[train_idx], y[train_idx]
        X_outer_test, y_outer_test = X[test_idx], y[test_idx]

        inner_kfold = KFold(
            n_splits=train_cfg.inner_folds,
            shuffle=True,
            random_state=train_cfg.random_state,
        )

        members = []
        for inner_idx, (t, v) in enumerate(
            inner_kfold.split(X_outer_train, y_outer_train), 1
        ):
            print(f"\n  Inner fold {inner_idx}/{train_cfg.inner_folds}")
            model = build_cnn(cnn_cfg)
            model.fit(
                X_outer_train[t],
                y_outer_train[t],
                epochs=train_cfg.epochs,
                batch_size=train_cfg.batch_size,
                validation_data=(X_outer_train[v], y_outer_train[v]),
                verbose=0,
            )
            val_loss, val_acc = model.evaluate(
                X_outer_train[v], y_outer_train[v], verbose=0
            )
            print(f"    Val accuracy: {val_acc:.4f}")

            save_path = train_cfg.model_save_pattern.format(fold=inner_idx)
            model.save(save_path)
            members.append(model)

        # CNN standalone evaluation
        cnn_loss, cnn_acc = members[0].evaluate(
            X_outer_test, y_outer_test, verbose=0)
        print(f"\n  CNN (fold-1) test accuracy: {cnn_acc:.4f}")

        # Ensemble evaluation
        ens_acc, y_pred, _ = evaluate_ensemble(
            X_outer_train, y_outer_train,
            X_outer_test, y_outer_test,
            members=members,
        )

        results.append({
            "fold": fold_idx,
            "cnn_acc": cnn_acc,
            "ensemble_acc": ens_acc,
        })

    return results


if __name__ == "__main__":
    from src.data_loader import build_numpy_arrays
    from src.outlier_detection import remove_outliers

    X, y = build_numpy_arrays()
    X, y, _ = remove_outliers(X, y)
    results = run_nested_cv(X, y)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in results:
        print(
            f"  Fold {r['fold']:2d}  |  CNN: {r['cnn_acc']:.4f}  |  Ensemble: {r['ensemble_acc']:.4f}")
    accs = [r["ensemble_acc"] for r in results]
    print(
        f"\n  Mean Ensemble Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
