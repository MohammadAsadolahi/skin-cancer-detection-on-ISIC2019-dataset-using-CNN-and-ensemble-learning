"""
data_loader.py — ISIC 2019 Dataset Loading, Preprocessing & Augmentation.

Handles:
  1. Parsing the ISIC 2019 ground-truth CSV
  2. Organizing images into per-class directories
  3. Resizing all images to a uniform resolution
  4. Applying augmentation (brightness, rotation, shift)
  5. Building final numpy arrays for training
"""

import os
import glob
import shutil

import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image as keras_image

from src.config import (
    DataConfig,
    AugmentationConfig,
    CATEGORY_LABELS,
    INPUT_SIZE,
)


def load_ground_truth(cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Load and clean the ISIC 2019 ground truth CSV."""
    df = pd.read_csv(cfg.ground_truth_csv)
    # Drop the last column (UNK / unknown)
    df = df.drop(columns=[df.columns[-1]])
    return df


def get_class_image_lists(df: pd.DataFrame):
    """Return a list of 8 Series, each containing image IDs for one class."""
    address_lists = []
    for i in range(1, 9):
        col = df.columns[i]
        address_lists.append(df[df[col] == 1]["image"])
    return address_lists


def organize_dataset(address_lists, cfg: DataConfig = DataConfig()):
    """Resize raw ISIC images and save to per-class directories."""
    for i in range(8):
        os.makedirs(f"{cfg.dataset_dir}/{i}/{i}", exist_ok=True)

    for i in range(8):
        for j in range(len(address_lists[i])):
            img_id = address_lists[i].iloc[j]
            src = os.path.join(cfg.images_dir, img_id + ".jpg")
            dst = f"{cfg.dataset_dir}/{i}/{i}/{img_id}.png"
            img = cv2.resize(cv2.imread(src), (cfg.input_size, cfg.input_size))
            cv2.imwrite(dst, img)
            del img
        print(f"[data_loader] Category {CATEGORY_LABELS[i]} organized.")


def augment_dataset(
    address_lists,
    cfg: DataConfig = DataConfig(),
    aug_cfg: AugmentationConfig = AugmentationConfig(),
):
    """Apply augmentation transforms and merge augmented images back."""
    for i in range(8):
        os.makedirs(f"{cfg.augmented_dir}/{i}", exist_ok=True)

    transforms = [
        ("b", {"brightness_range": list(aug_cfg.brightness_range)}),
        ("r", {"rotation_range": aug_cfg.rotation_range}),
        ("w", {"width_shift_range": aug_cfg.width_shift_range}),
        ("h", {"height_shift_range": aug_cfg.height_shift_range}),
    ]

    for i in range(8):
        original_path = f"{cfg.dataset_dir}/{i}/"
        augmented_path = f"{cfg.augmented_dir}/{i}/"
        batch_size = max(
            1, len(address_lists[i]) // aug_cfg.augmentation_steps)

        for prefix, params in transforms:
            gen = keras_image.ImageDataGenerator(**params).flow_from_directory(
                original_path,
                batch_size=batch_size,
                save_to_dir=augmented_path,
                save_prefix=prefix,
                target_size=(cfg.input_size, cfg.input_size),
            )
            for _ in range(aug_cfg.augmentation_steps):
                gen.next()
            del gen

    # Move augmented images back into Dataset/
    for i in range(8):
        for img_path in glob.glob(f"{cfg.augmented_dir}/{i}/*.png"):
            shutil.move(img_path, f"{cfg.dataset_dir}/{i}/{i}/")
    shutil.rmtree(cfg.augmented_dir, ignore_errors=True)
    print("[data_loader] Augmentation complete.")


def build_numpy_arrays(cfg: DataConfig = DataConfig()):
    """
    Build normalized image arrays and integer label vectors from the
    organized dataset directory.

    Returns
    -------
    train_data : np.ndarray  — shape (N, H, W, 3), values in [0, 1]
    labels     : np.ndarray  — shape (N,), integer class labels
    """
    all_images, all_labels = [], []
    for i in range(8):
        paths = sorted(glob.glob(f"{cfg.dataset_dir}/{i}/{i}/*.png"))
        all_images.extend(paths)
        all_labels.extend([i] * len(paths))
        print(f"[data_loader] Class {CATEGORY_LABELS[i]}: {len(paths)} images")

    labels = np.array(all_labels)
    data = np.array(
        [cv2.imread(p).astype(np.float32) / 255.0 for p in all_images]
    )
    return data, labels
