#!/usr/bin/env python3
"""전처리된 split CSV 파일로 SilverWalk MLP 위험도 모델을 학습한다."""

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
from silverwalk_ai.modeling.mlp import (
    MLPTrainingConfig,
    configure_tensorflow_device,
    evaluate_predictions,
    history_to_frame,
    predict_risk,
    train_mlp_model,
)


def parse_args() -> argparse.Namespace:
    """데이터 경로, 학습 옵션, TensorFlow 장치 옵션을 CLI 인자로 받는다."""
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
    parser.add_argument("--model-path", type=Path, default=PATHS.models / "mlp_risk.keras")
    parser.add_argument("--metrics-path", type=Path, default=PATHS.reports / "mlp_risk_metrics.json")
    parser.add_argument("--history-path", type=Path, default=PATHS.reports / "mlp_risk_history.csv")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=PATHS.predictions / "mlp_risk_test_predictions.csv",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-patience", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
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


def project_display_path(path: Path) -> str:
    """가능하면 프로젝트 루트 기준 상대 경로로 표시한다."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PATHS.root))
    except ValueError:
        return str(resolved)


def main() -> None:
    """MLP 학습, 평가, 산출물 저장을 처음부터 끝까지 실행한다."""
    args = parse_args()
    ensure_project_dirs()
    PATHS.predictions.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    # TensorFlow 연산이 올바른 장치에 배치되도록 모델 생성 전에 GPU/CPU를 설정한다.
    device_info = configure_tensorflow_device(
        device=args.device,
        enable_memory_growth=True,
        enable_mixed_precision=args.mixed_precision,
    )

    # 전처리 config를 feature 순서의 단일 기준으로 사용한다.
    with args.preprocess_config.open("r", encoding="utf-8") as file:
        preprocess_config = json.load(file)
    feature_columns = preprocess_config["feature_columns"]

    # 이 CSV들은 이미 변환된 feature와 로그 스케일 타겟을 포함한다.
    train_frame = read_split(args.train, args.limit_rows)
    val_frame = read_split(args.val, args.limit_rows)
    test_frame = read_split(args.test, args.limit_rows)

    # 지표 리포트로 재현할 수 있도록 하이퍼파라미터를 하나의 dataclass에 모은다.
    config = MLPTrainingConfig(
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        random_state=args.random_state,
    )

    # train split으로 학습하고 val split은 콜백과 모델 선택에만 사용한다.
    model, history, used_features = train_mlp_model(
        train_frame=train_frame,
        val_frame=val_frame,
        config=config,
        feature_columns=feature_columns,
        model_path=args.model_path,
    )

    # 보통 ModelCheckpoint가 모델을 저장하지만, 예외 상황을 대비해 대체 저장을 둔다.
    if not args.model_path.exists():
        model.save(args.model_path)

    # test split은 학습 완료 후 최종 일반화 성능 추정에 한 번만 사용한다.
    test_predictions = predict_risk(
        model,
        test_frame,
        feature_columns=used_features,
        batch_size=max(args.batch_size, 1024),
    )
    metrics = evaluate_predictions(test_predictions)
    metrics.update(
        {
            "training_config": config.to_dict(),
            "device": device_info.to_dict(),
            "tensorflow_version": tf.__version__,
            "feature_count": len(used_features),
            "train_rows": len(train_frame),
            "val_rows": len(val_frame),
            "test_rows": len(test_frame),
            "model_path": project_display_path(args.model_path),
            "history_path": project_display_path(args.history_path),
            "predictions_path": project_display_path(args.predictions_path),
        }
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)

    # 노트북과 Streamlit에서 재사용할 수 있도록 리포트와 예측 행을 저장한다.
    args.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_to_frame(history).to_csv(args.history_path, index=False)
    test_predictions.to_csv(args.predictions_path, index=False)

    print("Training complete.")
    print(f"Features: {len(used_features):,}")
    print(f"Train rows: {len(train_frame):,}")
    print(f"Validation rows: {len(val_frame):,}")
    print(f"Test rows: {len(test_frame):,}")
    print(f"Model: {project_display_path(args.model_path)}")
    print(f"Metrics: {project_display_path(args.metrics_path)}")
    print(f"History: {project_display_path(args.history_path)}")
    print(f"Predictions: {project_display_path(args.predictions_path)}")
    print("Device:")
    print(json.dumps(device_info.to_dict(), ensure_ascii=False, indent=2))
    print("Test metrics:")
    print(json.dumps(metrics["risk_scale"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
