"""포인트 단위 위험도 회귀를 위한 Keras MLP 유틸리티."""

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


ACCIDENT_TARGET_COLUMN = "사고발생"
DEFAULT_METADATA_COLUMNS = [
    "POINT_ID",
    "위도",
    "경도",
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
    ACCIDENT_TARGET_COLUMN,
]


@dataclass(frozen=True)
class MLPTrainingConfig:
    """MLP 모델 생성과 학습에 사용하는 하이퍼파라미터."""

    hidden_units: tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    huber_delta: float = 1.0
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 4
    random_state: int = 42
    verbose: int = 2

    def to_dict(self) -> dict[str, Any]:
        """리포트 저장을 위해 JSON으로 직렬화 가능한 dict를 반환한다."""
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
            "verbose": self.verbose,
        }


@dataclass(frozen=True)
class TensorFlowDeviceInfo:
    """학습 리포트에 저장할 TensorFlow 장치 설정 결과."""

    requested_device: str
    active_device: str
    physical_gpus: list[str]
    logical_gpus: list[str]
    memory_growth: bool
    mixed_precision: bool

    def to_dict(self) -> dict[str, Any]:
        """지표 파일 저장을 위해 JSON으로 직렬화 가능한 dict를 반환한다."""
        return {
            "requested_device": self.requested_device,
            "active_device": self.active_device,
            "physical_gpus": self.physical_gpus,
            "logical_gpus": self.logical_gpus,
            "memory_growth": self.memory_growth,
            "mixed_precision": self.mixed_precision,
        }


def configure_tensorflow_device(
    device: str = "auto",
    enable_memory_growth: bool = True,
    enable_mixed_precision: bool = False,
) -> TensorFlowDeviceInfo:
    """사용 가능한 경우 TensorFlow가 GPU를 쓰도록 설정한다.

    `device="auto"`는 TensorFlow가 GPU를 감지하면 GPU를 쓰고, 없으면 CPU를 쓴다.
    `device="gpu"`는 GPU가 없으면 즉시 실패한다.
    `device="cpu"`는 TensorFlow에서 GPU를 숨기고 CPU만 사용한다.
    """
    normalized_device = device.lower()
    if normalized_device not in {"auto", "gpu", "cpu"}:
        raise ValueError("device must be one of: auto, gpu, cpu")

    physical_gpus = tf.config.list_physical_devices("GPU")

    # CPU 모드는 재현성 확인이나 CUDA 설치가 불완전한 환경에서 유용하다.
    if normalized_device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        return TensorFlowDeviceInfo(
            requested_device=normalized_device,
            active_device="cpu",
            physical_gpus=[gpu.name for gpu in physical_gpus],
            logical_gpus=[gpu.name for gpu in logical_gpus],
            memory_growth=False,
            mixed_precision=False,
        )

    if normalized_device == "gpu" and not physical_gpus:
        raise RuntimeError(
            "GPU 사용을 요청했지만 TensorFlow가 물리 GPU를 감지하지 못했습니다. "
            "NVIDIA driver, CUDA/cuDNN, tensorflow 설치 상태를 확인하십시오."
        )

    # 메모리 증가 옵션은 TensorFlow가 시작 시 GPU 메모리를 전부 선점하지 않게 한다.
    memory_growth_enabled = False
    if physical_gpus and enable_memory_growth:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        memory_growth_enabled = True

    logical_gpus = tf.config.list_logical_devices("GPU")
    active_device = "gpu" if logical_gpus else "cpu"

    # 혼합 정밀도는 GPU가 활성화된 경우에만 켜고, CPU 학습은 float32로 유지한다.
    mixed_precision_enabled = bool(enable_mixed_precision and logical_gpus)
    if mixed_precision_enabled:
        keras.mixed_precision.set_global_policy("mixed_float16")

    return TensorFlowDeviceInfo(
        requested_device=normalized_device,
        active_device=active_device,
        physical_gpus=[gpu.name for gpu in physical_gpus],
        logical_gpus=[gpu.name for gpu in logical_gpus],
        memory_growth=memory_growth_enabled,
        mixed_precision=mixed_precision_enabled,
    )


def set_random_seed(seed: int) -> None:
    """모델 생성 전에 NumPy와 TensorFlow 난수 seed를 고정한다."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    """메타데이터와 타겟 컬럼을 제외해 모델 feature 컬럼을 추론한다."""
    return [column for column in frame.columns if column not in DEFAULT_METADATA_COLUMNS]


def split_features_target(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_LOG_COLUMN,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """전처리된 DataFrame을 float32 feature 행렬과 타겟 벡터로 분리한다."""
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
    """`위험도_log1p` 회귀를 위한 기본 표 형식 MLP를 생성한다."""
    config = config or MLPTrainingConfig()

    # 입력 차원은 전처리 config에 저장된 feature 순서와 일치해야 한다.
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    # 이 데이터는 이미지/텍스트가 아닌 표 형식 데이터이므로 Dense block을 작게 유지한다.
    for layer_index, units in enumerate(config.hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{layer_index + 1}")(x)
        if layer_index < 2:
            x = layers.BatchNormalization(name=f"batch_norm_{layer_index + 1}")(x)
            if config.dropout_rate > 0:
                x = layers.Dropout(config.dropout_rate, name=f"dropout_{layer_index + 1}")(x)

    # 혼합 정밀도를 쓰더라도 회귀 출력 정밀도가 낮아지지 않도록 출력은 float32로 고정한다.
    outputs = layers.Dense(1, activation="linear", dtype="float32", name="risk_log1p")(x)
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


def build_accident_classifier_model(
    input_dim: int,
    config: MLPTrainingConfig | None = None,
) -> keras.Model:
    """`위험도 > 0` 여부를 예측하는 사고 발생 분류 MLP를 생성한다."""
    config = config or MLPTrainingConfig()

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for layer_index, units in enumerate(config.hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{layer_index + 1}")(x)
        if layer_index < 2:
            x = layers.BatchNormalization(name=f"batch_norm_{layer_index + 1}")(x)
            if config.dropout_rate > 0:
                x = layers.Dropout(config.dropout_rate, name=f"dropout_{layer_index + 1}")(x)

    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="accident_probability")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="silverwalk_mlp_accident_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            keras.metrics.Precision(name="precision", thresholds=0.5),
            keras.metrics.Recall(name="recall", thresholds=0.5),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def make_training_callbacks(
    model_path: Path | None = None,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 4,
    monitor: str = "val_loss",
    mode: str = "min",
) -> list[keras.callbacks.Callback]:
    """조기 종료, 학습률 감소, 체크포인트 저장 콜백을 만든다."""
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
        ),
    ]

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # 검증 손실이 가장 좋은 epoch만 저장해 품질이 낮은 체크포인트를 남기지 않는다.
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor=monitor,
                mode=mode,
                save_best_only=True,
            )
        )

    return callbacks


def add_accident_target(frame: pd.DataFrame) -> pd.DataFrame:
    """원본 위험도에서 모델 1 분류 라벨인 `사고발생`을 파생한다."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    result = frame.copy()
    result[ACCIDENT_TARGET_COLUMN] = (result[TARGET_COLUMN] > 0).astype(np.float32)
    return result


def train_mlp_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: MLPTrainingConfig | None = None,
    feature_columns: list[str] | None = None,
    model_path: Path | None = None,
) -> tuple[keras.Model, keras.callbacks.History, list[str]]:
    """MLP를 학습하고 학습된 모델, Keras 학습 이력, feature 순서를 반환한다."""
    config = config or MLPTrainingConfig()
    set_random_seed(config.random_state)

    # 검증 데이터도 학습 데이터와 완전히 같은 feature 순서를 사용한다.
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
        verbose=config.verbose,
    )
    return model, history, features


def train_accident_classifier_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: MLPTrainingConfig | None = None,
    feature_columns: list[str] | None = None,
    model_path: Path | None = None,
) -> tuple[keras.Model, keras.callbacks.History, list[str]]:
    """모델 1: 사고 발생 여부 분류 MLP를 학습한다."""
    config = config or MLPTrainingConfig()
    set_random_seed(config.random_state)

    train_frame = add_accident_target(train_frame)
    val_frame = add_accident_target(val_frame)
    x_train, y_train, features = split_features_target(
        train_frame,
        feature_columns=feature_columns,
        target_column=ACCIDENT_TARGET_COLUMN,
    )
    x_val, y_val, _ = split_features_target(
        val_frame,
        feature_columns=features,
        target_column=ACCIDENT_TARGET_COLUMN,
    )

    model = build_accident_classifier_model(input_dim=x_train.shape[1], config=config)
    callbacks = make_training_callbacks(
        model_path=model_path,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        monitor="val_pr_auc",
        mode="max",
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    return model, history, features


def predict_risk(
    model: keras.Model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    batch_size: int = 4096,
) -> pd.DataFrame:
    """위험도를 예측하고 로그 스케일과 원래 스케일 예측값을 함께 반환한다."""
    x, _, _ = split_features_target(frame, feature_columns)
    pred_log = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1)

    # 모델은 log1p 위험도를 예측하므로 리포트와 지도 시각화를 위해 원래 scale로 복원한다.
    pred_risk = np.expm1(pred_log)
    pred_risk = np.clip(pred_risk, 0, None)

    result = frame[["POINT_ID", "위도", "경도", TARGET_COLUMN, TARGET_LOG_COLUMN]].copy()
    result["pred_위험도_log1p"] = pred_log
    result["pred_위험도"] = pred_risk
    return result


def predict_accident_probability(
    model: keras.Model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    batch_size: int = 4096,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """사고 발생 확률을 예측하고 실제 사고 발생 라벨과 함께 반환한다."""
    labeled_frame = add_accident_target(frame)
    x, y_true, _ = split_features_target(
        labeled_frame,
        feature_columns=feature_columns,
        target_column=ACCIDENT_TARGET_COLUMN,
    )
    probabilities = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1)

    result = labeled_frame[["POINT_ID", "위도", "경도", TARGET_COLUMN, ACCIDENT_TARGET_COLUMN]].copy()
    result["사고발생확률"] = probabilities
    result["사고발생예측"] = (probabilities >= threshold).astype(int)
    result[ACCIDENT_TARGET_COLUMN] = y_true.astype(int)
    return result


def predict_unlabeled_risk(
    model: keras.Model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    batch_size: int = 4096,
) -> pd.DataFrame:
    """라벨 없는 전체 포인트 데이터에 대해 위험도를 예측해 `위험도_pred`를 반환한다."""
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    pred_log = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1)

    # 모델 출력은 log1p 위험도이므로 사람이 해석하는 원래 위험도 scale로 되돌린다.
    pred_risk = np.expm1(pred_log)
    pred_risk = np.clip(pred_risk, 0, None)

    result = frame[["POINT_ID", "위도", "경도"]].copy()
    result["위험도_pred_log1p"] = pred_log
    result["위험도_pred"] = pred_risk
    return result


def binary_classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """고정 threshold에서 기본 이진 분류 지표를 계산한다."""
    actual = np.asarray(actual, dtype=np.int32)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = (probability >= threshold).astype(np.int32)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(actual) if len(actual) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def top_k_classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    k_values: tuple[int, ...] = (100, 300, 500, 700, 1000),
) -> dict[str, dict[str, float]]:
    """사고 발생 확률 상위 K개 기준 Precision@K, Recall@K, F1@K를 계산한다."""
    actual = np.asarray(actual, dtype=np.int32)
    probability = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-probability)
    positive_count = int((actual == 1).sum())

    metrics: dict[str, dict[str, float]] = {}
    for requested_k in k_values:
        k = min(int(requested_k), len(actual))
        if k <= 0:
            continue
        top_indices = order[:k]
        hit_count = int((actual[top_indices] == 1).sum())
        precision = hit_count / k
        recall = hit_count / positive_count if positive_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics[str(requested_k)] = {
            "k": k,
            "hit_count": hit_count,
            "precision_at_k": float(precision),
            "recall_at_k": float(recall),
            "f1_at_k": float(f1),
        }
    return metrics


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """하나의 타겟 scale에 대해 기본 회귀 지표를 계산한다."""
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


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """로그 스케일과 복원된 원래 위험도 스케일에서 예측 성능을 평가한다."""
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


def evaluate_accident_predictions(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    k_values: tuple[int, ...] = (100, 300, 500, 700, 1000),
) -> dict[str, Any]:
    """모델 1 사고 발생 분류 결과를 threshold와 Top-K 관점에서 평가한다."""
    actual = predictions[ACCIDENT_TARGET_COLUMN].to_numpy()
    probability = predictions["사고발생확률"].to_numpy()
    return {
        "target_counts": {
            "rows": int(len(predictions)),
            "actual_positive": int((predictions[ACCIDENT_TARGET_COLUMN] == 1).sum()),
            "actual_negative": int((predictions[ACCIDENT_TARGET_COLUMN] == 0).sum()),
        },
        "threshold_metrics": binary_classification_metrics(
            actual,
            probability,
            threshold=threshold,
        ),
        "top_k_metrics": top_k_classification_metrics(
            actual,
            probability,
            k_values=k_values,
        ),
    }


def history_to_frame(history: keras.callbacks.History) -> pd.DataFrame:
    """Keras History 객체를 CSV 저장용 DataFrame으로 변환한다."""
    return pd.DataFrame(history.history)
