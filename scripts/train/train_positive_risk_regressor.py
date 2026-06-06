#!/usr/bin/env python3
"""모델 2: `위험도 > 0` 포인트만 사용해 사고 발생 시 위험도 크기를 예측한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from silverwalk_ai.data.paths import PATHS, ensure_project_dirs
from silverwalk_ai.features.preprocessing import TARGET_COLUMN
from silverwalk_ai.modeling.mlp import (
    MLPTrainingConfig,
    configure_tensorflow_device,
    evaluate_predictions,
    history_to_frame,
    predict_risk,
    train_mlp_model,
)


def parse_args() -> argparse.Namespace:
    """데이터 경로, 학습 옵션, 저장 경로를 CLI 인자로 받는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=PATHS.processed / "original_train_train_preprocessed.csv",
        help="전처리된 train CSV 경로.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=PATHS.processed / "original_train_val_preprocessed.csv",
        help="전처리된 validation CSV 경로.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PATHS.processed / "original_train_test_preprocessed.csv",
        help="전처리된 test CSV 경로.",
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=PATHS.preprocessors / "original_train_preprocess_config.json",
        help="전처리 config JSON 경로.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PATHS.models / "mlp_positive_risk_regressor.keras",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=PATHS.reports / "mlp_positive_risk_regressor_metrics.json",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=PATHS.reports / "mlp_positive_risk_regressor_history.csv",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=PATHS.predictions / "mlp_positive_risk_regressor_test_predictions.csv",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-patience", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="TensorFlow 장치 모드. 'gpu'는 GPU가 없으면 실패한다.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="GPU가 활성화된 경우 혼합 정밀도 mixed_float16을 사용한다.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="빠른 동작 확인을 위한 split별 선택적 행 수 제한.",
    )
    return parser.parse_args()


def read_split(path: Path, limit_rows: int | None = None) -> pd.DataFrame:
    """전처리된 split 하나를 읽고, 필요한 경우 행 수를 제한한다."""
    return pd.read_csv(path, nrows=limit_rows)


def filter_positive_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """모델 2 학습 대상인 `위험도 > 0` 행만 남긴다."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    return frame[frame[TARGET_COLUMN] > 0].reset_index(drop=True)


def project_display_path(path: Path) -> str:
    """가능하면 프로젝트 루트 기준 상대 경로로 표시한다."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PATHS.root))
    except ValueError:
        return str(resolved)


def main() -> None:
    """양수 위험도 포인트만으로 MLP 회귀 모델을 학습하고 평가한다."""
    args = parse_args()
    ensure_project_dirs()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)

    device_info = configure_tensorflow_device(
        device=args.device,
        enable_memory_growth=True,
        enable_mixed_precision=args.mixed_precision,
    )

    with args.preprocess_config.open("r", encoding="utf-8") as file:
        preprocess_config = json.load(file)
    feature_columns = preprocess_config["feature_columns"]

    train_frame_raw = read_split(args.train, args.limit_rows)
    val_frame_raw = read_split(args.val, args.limit_rows)
    test_frame_raw = read_split(args.test, args.limit_rows)

    train_frame = filter_positive_risk(train_frame_raw)
    val_frame = filter_positive_risk(val_frame_raw)
    test_frame = filter_positive_risk(test_frame_raw)
    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("모델 2 학습에는 train/val/test split마다 `위험도 > 0` 행이 필요합니다.")

    config = MLPTrainingConfig(
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    model, history, used_features = train_mlp_model(
        train_frame=train_frame,
        val_frame=val_frame,
        config=config,
        feature_columns=feature_columns,
        model_path=args.model_path,
    )

    if not args.model_path.exists():
        model.save(args.model_path)

    test_predictions = predict_risk(
        model,
        test_frame,
        feature_columns=used_features,
        batch_size=max(args.batch_size, 1024),
    )
    metrics = evaluate_predictions(test_predictions)
    metrics.update(
        {
            "model_role": "positive_risk_regressor",
            "target_definition": "E(위험도 | 위험도 > 0, x)",
            "training_config": config.to_dict(),
            "device": device_info.to_dict(),
            "tensorflow_version": tf.__version__,
            "feature_count": len(used_features),
            "raw_split_rows": {
                "train": len(train_frame_raw),
                "val": len(val_frame_raw),
                "test": len(test_frame_raw),
            },
            "positive_split_rows": {
                "train": len(train_frame),
                "val": len(val_frame),
                "test": len(test_frame),
            },
            "model_path": project_display_path(args.model_path),
            "history_path": project_display_path(args.history_path),
            "predictions_path": project_display_path(args.predictions_path),
        }
    )

    args.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_to_frame(history).to_csv(args.history_path, index=False)
    test_predictions.to_csv(args.predictions_path, index=False)

    print("Positive risk regressor training complete.")
    print(f"Features: {len(used_features):,}")
    print(f"Raw train rows: {len(train_frame_raw):,} -> positive: {len(train_frame):,}")
    print(f"Raw validation rows: {len(val_frame_raw):,} -> positive: {len(val_frame):,}")
    print(f"Raw test rows: {len(test_frame_raw):,} -> positive: {len(test_frame):,}")
    print(f"Model: {project_display_path(args.model_path)}")
    print(f"Metrics: {project_display_path(args.metrics_path)}")
    print(f"History: {project_display_path(args.history_path)}")
    print(f"Predictions: {project_display_path(args.predictions_path)}")
    print("Test risk-scale metrics:")
    print(json.dumps(metrics["risk_scale"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
