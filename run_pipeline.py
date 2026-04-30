"""
Skin Cancer Detection — ISIC 2019
CNN + XGBoost Ensemble with Nested Cross-Validation

Usage:
    python run_pipeline.py [--epochs 60] [--batch-size 256] [--outer-folds 10]

This script orchestrates the full pipeline:
    1. Load & preprocess ISIC 2019 images
    2. Apply data augmentation
    3. Remove outliers via Local Outlier Factor
    4. Train CNN base learners with nested K-fold CV
    5. Stack predictions with XGBoost meta-learner
    6. Report per-fold and aggregate metrics
"""

import argparse
import sys
import numpy as np

from src.config import TrainingConfig, CNNConfig, DataConfig, AugmentationConfig
from src.data_loader import (
    load_ground_truth,
    get_class_image_lists,
    organize_dataset,
    augment_dataset,
    build_numpy_arrays,
)
from src.outlier_detection import remove_outliers
from src.train import run_nested_cv
from src.visualization import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Skin Cancer Detection — CNN + XGBoost Ensemble Pipeline"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip dataset organization & augmentation")
    parser.add_argument("--generate-figures", action="store_true",
                        help="Generate publication figures and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate_figures:
        generate_all_figures()
        return

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
    )

    # ── Step 1: Data Preparation ──────────────────────────────────
    if not args.skip_preprocessing:
        print("\n[pipeline] Loading ground truth...")
        df = load_ground_truth()
        address_lists = get_class_image_lists(df)

        print("[pipeline] Organizing dataset...")
        organize_dataset(address_lists)

        print("[pipeline] Augmenting dataset...")
        augment_dataset(address_lists)

    # ── Step 2: Build Arrays ──────────────────────────────────────
    print("\n[pipeline] Building numpy arrays...")
    X, y = build_numpy_arrays()

    # ── Step 3: Outlier Removal ───────────────────────────────────
    print("\n[pipeline] Removing outliers...")
    X, y, n_removed = remove_outliers(X, y)

    # ── Step 4: Nested Cross-Validation ───────────────────────────
    print("\n[pipeline] Starting nested cross-validation...")
    results = run_nested_cv(X, y, train_cfg=train_cfg)

    # ── Step 5: Summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS — NESTED 10-FOLD CROSS-VALIDATION")
    print("=" * 70)
    for r in results:
        print(
            f"  Fold {r['fold']:2d}  │  CNN: {r['cnn_acc']:.4f}  │  Ensemble: {r['ensemble_acc']:.4f}")

    ens_accs = [r["ensemble_acc"] for r in results]
    cnn_accs = [r["cnn_acc"] for r in results]
    print(
        f"\n  CNN Mean Accuracy:      {np.mean(cnn_accs):.4f} ± {np.std(cnn_accs):.4f}")
    print(
        f"  Ensemble Mean Accuracy: {np.mean(ens_accs):.4f} ± {np.std(ens_accs):.4f}")
    print(
        f"  Ensemble Improvement:   +{(np.mean(ens_accs) - np.mean(cnn_accs))*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
