"""
config.py — Centralized Configuration for the Skin Cancer Detection Pipeline.

Defines all hyperparameters, dataset paths, class labels, and training
constants used across the pipeline. Modify this single file to change
any experimental parameter.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────
# Diagnostic Categories (ISIC 2019)
# ──────────────────────────────────────────────────────────────────────
CATEGORY_LABELS: List[str] = [
    "MEL",   # Melanoma
    "NV",    # Melanocytic Nevus
    "BCC",   # Basal Cell Carcinoma
    "AK",    # Actinic Keratosis
    "BKL",   # Benign Keratosis
    "DF",    # Dermatofibroma
    "VASC",  # Vascular Lesion
    "SCC",   # Squamous Cell Carcinoma
]

CATEGORY_FULL_NAMES: List[str] = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Squamous Cell Carcinoma",
]

NUM_CLASSES: int = 8

# ──────────────────────────────────────────────────────────────────────
# Image & Data Configuration
# ──────────────────────────────────────────────────────────────────────
INPUT_SIZE: int = 256
IMAGE_CHANNELS: int = 3
INPUT_SHAPE: Tuple[int, int, int] = (INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNELS)


@dataclass
class DataConfig:
    """Paths and parameters related to the ISIC 2019 dataset."""
    ground_truth_csv: str = "../input/isic-2019/ISIC_2019_Training_GroundTruth.csv"
    images_dir: str = "../input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"
    dataset_dir: str = "Dataset"
    augmented_dir: str = "DatasetAugmented"
    input_size: int = INPUT_SIZE


@dataclass
class AugmentationConfig:
    """Data augmentation hyperparameters."""
    brightness_range: Tuple[float, float] = (0.2, 0.4)
    rotation_range: float = 0.2
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    augmentation_steps: int = 10


# ──────────────────────────────────────────────────────────────────────
# CNN Architecture & Training
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CNNConfig:
    """Hyperparameters for the custom CNN architecture."""
    conv_filters: List[int] = field(
        default_factory=lambda: [256, 512, 256, 256])
    conv_kernel_size: int = 3
    conv_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 1])
    conv_dropout_rates: List[float] = field(
        default_factory=lambda: [0.3, 0.3, 0.3, 0.2])
    dense_units: List[int] = field(default_factory=lambda: [256, 128, 128])
    dense_dropout_rates: List[float] = field(
        default_factory=lambda: [0.2, 0.2, 0.3])
    num_classes: int = NUM_CLASSES
    activation_hidden: str = "relu"
    activation_output: str = "softmax"
    optimizer: str = "Adam"
    loss: str = "sparse_categorical_crossentropy"


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    epochs: int = 60
    batch_size: int = 256
    outer_folds: int = 10
    inner_folds: int = 2
    random_state: int = None  # None → fully stochastic
    model_save_pattern: str = "./CNN_fold{fold}.h5"


# ──────────────────────────────────────────────────────────────────────
# Ensemble (XGBoost Meta-Learner)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class EnsembleConfig:
    """Configuration for the XGBoost stacking meta-learner."""
    n_base_models: int = 2
    xgb_params: dict = field(default_factory=lambda: {
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    })


# ──────────────────────────────────────────────────────────────────────
# Outlier Detection
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OutlierConfig:
    """Parameters for Local Outlier Factor pre-processing."""
    n_neighbors: int = 20
    contamination: str = "auto"
