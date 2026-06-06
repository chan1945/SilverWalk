#!/usr/bin/env python3
"""모델 1과 모델 2를 결합해 사고 이력 0 포인트의 최종 위험도 점수를 계산한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

################################
#   1. 원본 데이터 로드
#   2. 위험도_actual = 0 인 행만 필터링
#   3. 기존 전처리 객체로 feature 변환
#   4. 모델 1에서 p = 사고발생확률 예측
#   5. 모델 2에서 r = 조건부위험도 예측
#   6. 최종위험도점수 = p * r 계산
#   7. 최종위험도점수_percent 계산
#   8. CSV 저장
#################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from silverwalk_ai.data.paths import PATHS, ensure_project_dirs
from silverwalk_ai.features.preprocessing import OriginalTrainPreprocessor, TARGET_COLUMN
from silverwalk_ai.modeling.mlp import configure_tensorflow_device


def parse_args() -> argparse.Namespace:
    """두 모델 결합 추론에 필요한 입력, 모델, 전처리 객체, 출력 경로를 받는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PATHS.data / "original_train_data" / "seoul_road_points.csv",
        help="예측할 원본 포인트 CSV 경로.",
    )
    parser.add_argument(
        "--classifier-model-path",
        type=Path,
        default=PATHS.models / "mlp_accident_classifier.keras",
        help="모델 1 사고 발생 분류 Keras 모델 경로.",
    )
    parser.add_argument(
        "--regressor-model-path",
        type=Path,
        default=PATHS.models / "mlp_positive_risk_regressor.keras",
        help="모델 2 양수 위험도 회귀 Keras 모델 경로.",
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
        default=PATHS.predictions / "two_stage_zero_risk_predictions.csv",
        help="사고 이력 0 포인트의 최종 위험도 점수 CSV 저장 경로.",
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


def add_score_percent(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """최종 점수 최대치를 100%로 두고 상대 위험도 백분율을 계산한다."""
    result = frame.copy()
    max_score = result[score_column].max()
    percent_column = f"{score_column}_percent"
    if pd.isna(max_score) or max_score <= 0:
        result[percent_column] = 0.0
    else:
        result[percent_column] = (result[score_column] / max_score * 100).clip(0, 100)
    return result


def main() -> None:
    """위험도 0 포인트에 대해 p, r, p*r 최종 점수를 계산한다."""
    args = parse_args()
    ensure_project_dirs()
    require_file(args.input, "Input CSV")
    require_file(args.classifier_model_path, "Accident classifier model")
    require_file(args.regressor_model_path, "Positive risk regressor model")
    require_file(args.preprocessor_path, "Preprocessor")

    device_info = configure_tensorflow_device(device=args.device)

    raw_frame = pd.read_csv(args.input)
    if TARGET_COLUMN not in raw_frame.columns:
        raise ValueError(f"Input CSV must contain `{TARGET_COLUMN}` for zero-risk filtering.")

    zero_risk_frame = raw_frame[raw_frame[TARGET_COLUMN] == 0].copy()
    preprocessor = OriginalTrainPreprocessor.load(args.preprocessor_path)
    feature_frame = preprocessor.transform_features(zero_risk_frame)
    features = feature_frame[preprocessor.feature_columns].to_numpy(dtype=np.float32)

    classifier_model = tf.keras.models.load_model(args.classifier_model_path)
    regressor_model = tf.keras.models.load_model(args.regressor_model_path)

    accident_probability = classifier_model.predict(
        features,
        batch_size=args.batch_size,
        verbose=0,
    ).reshape(-1)

    conditional_risk_log1p = regressor_model.predict(
        features,
        batch_size=args.batch_size,
        verbose=0,
    ).reshape(-1)
    conditional_risk = np.expm1(conditional_risk_log1p)
    conditional_risk = np.clip(conditional_risk, 0, None)

    final_score = accident_probability * conditional_risk

    result = zero_risk_frame[["POINT_ID", "위도", "경도", TARGET_COLUMN]].copy()
    result = result.rename(columns={TARGET_COLUMN: "위험도_actual"})
    result["사고발생확률_p"] = accident_probability
    result["조건부위험도_log1p_r"] = conditional_risk_log1p
    result["조건부위험도_r"] = conditional_risk
    result["최종위험도점수"] = final_score
    result = add_score_percent(result, "최종위험도점수")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    high_risk_count = int((result["최종위험도점수_percent"] > 50).sum())
    print("Two-stage zero-risk prediction complete.")
    print(f"Input rows: {len(raw_frame):,}")
    print(f"Zero-risk target rows: {len(zero_risk_frame):,}")
    print(f"Output rows: {len(result):,}")
    print(f"Final score percent > 50%: {high_risk_count:,}")
    print(f"Max accident probability p: {result['사고발생확률_p'].max():.6f}")
    print(f"Max conditional risk r: {result['조건부위험도_r'].max():.6f}")
    print(f"Max final score p*r: {result['최종위험도점수'].max():.6f}")
    print(f"Output: {project_display_path(args.output)}")
    print(f"Device: {device_info.active_device}")


if __name__ == "__main__":
    main()
