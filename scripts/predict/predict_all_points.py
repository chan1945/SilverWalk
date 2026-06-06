#!/usr/bin/env python3
"""학습된 MLP로 전체 서울 포인트의 위험도를 예측한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from silverwalk_ai.data.paths import PATHS, ensure_project_dirs
from silverwalk_ai.features.preprocessing import OriginalTrainPreprocessor, TARGET_COLUMN
from silverwalk_ai.modeling.mlp import configure_tensorflow_device, predict_unlabeled_risk


def parse_args() -> argparse.Namespace:
    """전체 추론에 필요한 입력, 모델, 전처리 객체, 출력 경로를 CLI 인자로 받는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PATHS.data / "original_train_data" / "seoul_road_points.csv",
        help="예측할 전체 포인트 CSV 경로.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PATHS.models / "mlp_risk.keras",
        help="학습된 Keras MLP 모델 경로.",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=PATHS.preprocessors / "original_train_preprocessor.joblib",
        help="학습 데이터로 fit한 전처리 객체 경로.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PATHS.predictions / "mlp_risk_all_points.csv",
        help="전체 포인트 예측 CSV 저장 경로.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="TensorFlow 장치 모드. 'gpu'는 GPU가 없으면 실패한다.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    """필수 입력 파일이 없을 때 어떤 파일이 필요한지 명확히 실패시킨다."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def project_display_path(path: Path) -> str:
    """가능하면 프로젝트 루트 기준 상대 경로로 출력한다."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PATHS.root))
    except ValueError:
        return str(resolved)


def add_prediction_percent(predictions: pd.DataFrame) -> pd.DataFrame:
    """전체 예측값의 최대치를 100%로 두고 상대 위험도 백분율을 계산한다."""
    result = predictions.copy()
    max_risk = result["위험도_pred"].max()
    if pd.isna(max_risk) or max_risk <= 0:
        result["위험도_pred_percent"] = 0.0
    else:
        result["위험도_pred_percent"] = (result["위험도_pred"] / max_risk * 100).clip(0, 100)
    return result


def main() -> None:
    """전체 포인트를 전처리하고 MLP 예측 결과를 CSV로 저장한다."""
    args = parse_args()
    ensure_project_dirs()
    require_file(args.input, "Input CSV")
    require_file(args.model_path, "MLP model")
    require_file(args.preprocessor_path, "Preprocessor")

    # 모델 생성/로드 전에 TensorFlow 장치 정책을 확정한다.
    device_info = configure_tensorflow_device(device=args.device)

    # 원본 위험도는 모델 입력이 아니라, 사고 이력 0 포인트만 고르는 필터로만 사용한다.
    raw_frame = pd.read_csv(args.input)
    if TARGET_COLUMN not in raw_frame.columns:
        raise ValueError(f"Input CSV must contain `{TARGET_COLUMN}` for zero-risk filtering.")

    prediction_source_frame = raw_frame[raw_frame[TARGET_COLUMN] == 0].copy()
    preprocessor = OriginalTrainPreprocessor.load(args.preprocessor_path)
    feature_frame = preprocessor.transform_features(prediction_source_frame)

    model = tf.keras.models.load_model(args.model_path)
    predictions = predict_unlabeled_risk(
        model=model,
        frame=feature_frame,
        feature_columns=preprocessor.feature_columns,
        batch_size=args.batch_size,
    )
    predictions.insert(3, "위험도_actual", prediction_source_frame[TARGET_COLUMN].to_numpy())
    predictions = add_prediction_percent(predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    high_risk_count = int((predictions["위험도_pred_percent"] > 50).sum())
    print("Prediction complete.")
    print(f"Input rows: {len(raw_frame):,}")
    print(f"Prediction target rows where 위험도 = 0: {len(prediction_source_frame):,}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Predicted 위험도 percent > 50%: {high_risk_count:,}")
    print(f"Max predicted 위험도: {predictions['위험도_pred'].max():.6f}")
    print(f"Output: {project_display_path(args.output)}")
    print(f"Device: {device_info.active_device}")


if __name__ == "__main__":
    main()
