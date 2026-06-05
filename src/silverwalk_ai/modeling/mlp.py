"""Keras MLP model utilities for point-level risk regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from silverwalk_ai.features.preprocessing import TARGET_COLUMN, TARGET_LOG_COLUMN


DEFAULT_METADATA_COLUMNS = ["POINT_ID", "위도", "경도", TARGET_COLUMN, TARGET_LOG_COLUMN]


@dataclass(frozen=True)
class MLPTrainingConfig:
    hidden_units: tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    huber_delta: float = 1.0
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 4
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_units": list(self.hidden_units),
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "huber_delta": self.huber_delta,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "random_state": self.random_state,
        }


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in DEFAULT_METADATA_COLUMNS]


def split_features_target(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_LOG_COLUMN,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    features = feature_columns or feature_columns_from_frame(frame)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns missing from frame: {missing}")

    x = frame[features].to_numpy(dtype=np.float32)
    y = frame[target_column].to_numpy(dtype=np.float32)
    return x, y, features


def build_mlp_model(
    input_dim: int,
    config: MLPTrainingConfig | None = None,
) -> keras.Model:
    config = config or MLPTrainingConfig()

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for layer_index, units in enumerate(config.hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{layer_index + 1}")(x)
        if layer_index < 2:
            x = layers.BatchNormalization(name=f"batch_norm_{layer_index + 1}")(x)
            if config.dropout_rate > 0:
                x = layers.Dropout(config.dropout_rate, name=f"dropout_{layer_index + 1}")(x)

    outputs = layers.Dense(1, activation="linear", name="risk_log1p")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="silverwalk_mlp_risk")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.Huber(delta=config.huber_delta),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def make_training_callbacks(
    model_path: Path | None = None,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 4,
) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
        ),
    ]

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )

    return callbacks


def train_mlp_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: MLPTrainingConfig | None = None,
    feature_columns: list[str] | None = None,
    model_path: Path | None = None,
) -> tuple[keras.Model, keras.callbacks.History, list[str]]:
    config = config or MLPTrainingConfig()
    set_random_seed(config.random_state)

    x_train, y_train, features = split_features_target(train_frame, feature_columns)
    x_val, y_val, _ = split_features_target(val_frame, features)

    model = build_mlp_model(input_dim=x_train.shape[1], config=config)
    callbacks = make_training_callbacks(
        model_path=model_path,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history, features


def predict_risk(
    model: keras.Model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    batch_size: int = 4096,
) -> pd.DataFrame:
    x, _, _ = split_features_target(frame, feature_columns)
    pred_log = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1)
    pred_risk = np.expm1(pred_log)
    pred_risk = np.clip(pred_risk, 0, None)

    result = frame[["POINT_ID", "위도", "경도", TARGET_COLUMN, TARGET_LOG_COLUMN]].copy()
    result["pred_위험도_log1p"] = pred_log
    result["pred_위험도"] = pred_risk
    return result


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))

    denominator = np.sum(np.square(actual - np.mean(actual)))
    r2 = 1.0 - (np.sum(np.square(error)) / denominator) if denominator > 0 else np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "log_scale": regression_metrics(
            predictions[TARGET_LOG_COLUMN].to_numpy(),
            predictions["pred_위험도_log1p"].to_numpy(),
        ),
        "risk_scale": regression_metrics(
            predictions[TARGET_COLUMN].to_numpy(),
            predictions["pred_위험도"].to_numpy(),
        ),
        "risk_threshold_counts": {
            "actual_gt_0": int((predictions[TARGET_COLUMN] > 0).sum()),
            "pred_gt_0": int((predictions["pred_위험도"] > 0).sum()),
            "actual_gt_50": int((predictions[TARGET_COLUMN] > 50).sum()),
            "pred_gt_50": int((predictions["pred_위험도"] > 50).sum()),
        },
    }


def history_to_frame(history: keras.callbacks.History) -> pd.DataFrame:
    return pd.DataFrame(history.history)

