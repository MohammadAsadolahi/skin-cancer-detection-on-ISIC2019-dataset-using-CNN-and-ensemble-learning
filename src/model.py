"""
model.py — CNN Architecture & Compilation.

Builds a custom 4-layer convolutional neural network with dropout
regularization, followed by a 3-layer dense head for 8-class skin
lesion classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from src.config import CNNConfig, INPUT_SHAPE


def build_cnn(cfg: CNNConfig = CNNConfig()) -> Sequential:
    """
    Construct and compile the CNN.

    Architecture
    ────────────
    Conv2D(256, 3×3, stride=2) → Dropout(0.3)
    Conv2D(512, 3×3, stride=2) → Dropout(0.3)
    Conv2D(256, 3×3, stride=2) → Dropout(0.3)
    Conv2D(256, 3×3, stride=1) → Dropout(0.2)
    Flatten
    Dense(256) → Dropout(0.2)
    Dense(128) → Dropout(0.2)
    Dense(128) → Dropout(0.3)
    Dense(8, softmax)
    """
    model = Sequential()

    for i, (filters, stride, drop) in enumerate(
        zip(cfg.conv_filters, cfg.conv_strides, cfg.conv_dropout_rates)
    ):
        kwargs = {"padding": "same", "activation": cfg.activation_hidden}
        if i == 0:
            kwargs["input_shape"] = INPUT_SHAPE
        model.add(Conv2D(filters, cfg.conv_kernel_size,
                  strides=stride, **kwargs))
        model.add(Dropout(drop))

    model.add(Flatten())

    for units, drop in zip(cfg.dense_units, cfg.dense_dropout_rates):
        model.add(Dense(units, activation=cfg.activation_hidden))
        model.add(Dropout(drop))

    model.add(Dense(cfg.num_classes, activation=cfg.activation_output))

    model.compile(
        loss=cfg.loss,
        optimizer=cfg.optimizer,
        metrics=["accuracy"],
    )
    return model
