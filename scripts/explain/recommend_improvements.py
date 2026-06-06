#!/usr/bin/env python3
"""SHAP 값으로 고위험 포인트별 개선우선순위를 추천한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from silverwalk_ai.data.paths import PATHS, ensure_project_dirs
from silverwalk_ai.features.preprocessing import OriginalTrainPreprocessor, TARGET_COLUMN
from silverwalk_ai.modeling.mlp import configure_tensorflow_device


RECOMMENDATION_MAPPING = {
    "과속방지턱개수": "과속방지턱 추가",
    "횡단보도개수": "횡단보도 추가",
    "신호등설치여부": "신호등 설치",
    "노인장애인보호구역여부": "노인/장애인 보호구역 지정",
    "가로등개수": "가로등 추가",
    "보행자우선도로여부": "보행자우선도로 지정",
    "CCTV개수": "CCTV 추가",
    "제한속도": "제한속도 하향",
    "횡단보도예고표시여부": "횡단보도 예고표시 설치",
}
DEFAULT_CLASSIFIER_MODEL_PATH = PATHS.models / "mlp_accident_classifier.keras"
DEFAULT_REGRESSOR_MODEL_PATH = PATHS.models / "mlp_positive_risk_regressor.keras"
NOTEBOOK_CLASSIFIER_MODEL_PATH = PATHS.models / "mlp_accident_classifier_notebook.keras"
NOTEBOOK_REGRESSOR_MODEL_PATH = PATHS.models / "mlp_positive_risk_regressor_notebook.keras"


def parse_args() -> argparse.Namespace:
    """추천 생성에 필요한 원본 데이터, 예측 결과, 모델, 출력 경로를 받는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PATHS.data / "original_train_data" / "seoul_road_points.csv",
        help="원본 포인트 CSV 경로.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=PATHS.predictions / "two_stage_zero_risk_predictions.csv",
        help="두 모델 결합 최종 예측 CSV 경로.",
    )
    parser.add_argument(
        "--classifier-model-path",
        type=Path,
        default=DEFAULT_CLASSIFIER_MODEL_PATH,
        help="모델 1 사고 발생 분류 Keras 모델 경로.",
    )
    parser.add_argument(
        "--regressor-model-path",
        type=Path,
        default=DEFAULT_REGRESSOR_MODEL_PATH,
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
        default=PATHS.recommendations / "point_improvement_recommendations.csv",
        help="포인트별 개선우선순위 추천 CSV 저장 경로.",
    )
    parser.add_argument(
        "--percent-threshold",
        type=float,
        default=10.0,
        help="추천을 생성할 최종위험도점수_percent 초과 기준.",
    )
    parser.add_argument("--top-n", type=int, default=3, help="포인트별 추천 개수.")
    parser.add_argument("--background-size", type=int, default=256, help="SHAP 기준 샘플 수.")
    parser.add_argument("--nsamples", type=int, default=200, help="SHAP GradientExplainer 샘플 수.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="디버그용 최대 추천 포인트 수. 기본값은 전체 대상.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="TensorFlow 장치 모드. 'gpu'는 GPU가 없으면 실패한다.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    """필수 파일이 없을 때 명확한 오류를 낸다."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_default_model_path(path: Path, default_path: Path, notebook_path: Path, description: str) -> Path:
    """기본 모델 파일이 없고 notebook 산출물이 있으면 notebook 모델을 사용한다."""
    if path.exists():
        return path
    if path == default_path and notebook_path.exists():
        print(f"{description} not found at {project_display_path(path)}.")
        print(f"Using notebook model instead: {project_display_path(notebook_path)}")
        return notebook_path
    return path


def project_display_path(path: Path) -> str:
    """가능하면 프로젝트 루트 기준 상대 경로로 출력한다."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PATHS.root))
    except ValueError:
        return str(resolved)


def load_prediction_targets(predictions_path: Path, threshold: float, max_points: int | None) -> pd.DataFrame:
    """지도 표시 대상과 같은 조건의 포인트만 추천 대상으로 고른다."""
    predictions = pd.read_csv(predictions_path)
    required_columns = [
        "POINT_ID",
        "위도",
        "경도",
        "위험도_actual",
        "사고발생확률_p",
        "조건부위험도_r",
        "최종위험도점수",
        "최종위험도점수_percent",
    ]
    missing = [column for column in required_columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    targets = predictions[
        (predictions["위험도_actual"] == 0)
        & (predictions["최종위험도점수_percent"] > threshold)
    ].copy()
    targets = targets.sort_values("최종위험도점수_percent", ascending=False)
    if max_points is not None:
        targets = targets.head(max_points).copy()
    return targets


def shap_values_for_model(model: tf.keras.Model, background: np.ndarray, target: np.ndarray, nsamples: int) -> np.ndarray:
    """Keras MLP의 단일 출력에 대한 SHAP 값을 2차원 배열로 반환한다."""
    import shap

    explainer = shap.GradientExplainer(model, background)
    values = explainer.shap_values(target, nsamples=nsamples)
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def normalize_positive(values: np.ndarray) -> np.ndarray:
    """포인트별 양수 SHAP 값을 합이 1이 되도록 정규화한다."""
    positive = np.clip(values, 0, None)
    sums = positive.sum(axis=1, keepdims=True)
    return np.divide(positive, sums, out=np.zeros_like(positive), where=sums > 0)


def build_recommendations(
    targets: pd.DataFrame,
    feature_columns: list[str],
    classifier_shap: np.ndarray,
    regressor_shap: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    """두 모델에서 공통으로 위험 기여가 양수인 개선 feature를 우선 추천한다."""
    recommendation_features = [
        feature for feature in RECOMMENDATION_MAPPING if feature in feature_columns
    ]
    if not recommendation_features:
        raise ValueError("No recommendation features are present in model feature columns.")

    feature_indices = np.array([feature_columns.index(feature) for feature in recommendation_features])
    classifier_reco = classifier_shap[:, feature_indices]
    regressor_reco = regressor_shap[:, feature_indices]
    classifier_norm = normalize_positive(classifier_reco)
    regressor_norm = normalize_positive(regressor_reco)
    common_positive = (classifier_reco > 0) & (regressor_reco > 0)
    combined_scores = (classifier_norm + regressor_norm) * common_positive

    output = targets[
        [
            "POINT_ID",
            "위도",
            "경도",
            "위험도_actual",
            "사고발생확률_p",
            "조건부위험도_r",
            "최종위험도점수",
            "최종위험도점수_percent",
        ]
    ].copy()

    for rank in range(1, top_n + 1):
        output[f"개선우선순위{rank}"] = ""
        output[f"추천근거_feature{rank}"] = ""
        output[f"추천점수{rank}"] = 0.0
        output[f"모델1_shap{rank}"] = 0.0
        output[f"모델2_shap{rank}"] = 0.0

    for row_index in range(len(output)):
        row_scores = combined_scores[row_index]
        ordered_indices = np.argsort(-row_scores)
        selected = [index for index in ordered_indices if row_scores[index] > 0][:top_n]

        for rank, feature_index in enumerate(selected, start=1):
            feature = recommendation_features[feature_index]
            output.iat[row_index, output.columns.get_loc(f"개선우선순위{rank}")] = RECOMMENDATION_MAPPING[feature]
            output.iat[row_index, output.columns.get_loc(f"추천근거_feature{rank}")] = feature
            output.iat[row_index, output.columns.get_loc(f"추천점수{rank}")] = float(row_scores[feature_index])
            output.iat[row_index, output.columns.get_loc(f"모델1_shap{rank}")] = float(classifier_reco[row_index, feature_index])
            output.iat[row_index, output.columns.get_loc(f"모델2_shap{rank}")] = float(regressor_reco[row_index, feature_index])

    output["추천개수"] = (output[[f"개선우선순위{rank}" for rank in range(1, top_n + 1)]] != "").sum(axis=1)
    return output


def main() -> None:
    """고위험 포인트별 개선우선순위 추천 CSV를 생성한다."""
    args = parse_args()
    ensure_project_dirs()
    require_file(args.input, "Input CSV")
    require_file(args.predictions_path, "Two-stage prediction CSV")
    require_file(args.preprocessor_path, "Preprocessor")

    classifier_model_path = resolve_default_model_path(
        args.classifier_model_path,
        DEFAULT_CLASSIFIER_MODEL_PATH,
        NOTEBOOK_CLASSIFIER_MODEL_PATH,
        "Accident classifier model",
    )
    regressor_model_path = resolve_default_model_path(
        args.regressor_model_path,
        DEFAULT_REGRESSOR_MODEL_PATH,
        NOTEBOOK_REGRESSOR_MODEL_PATH,
        "Positive risk regressor model",
    )
    require_file(classifier_model_path, "Accident classifier model")
    require_file(regressor_model_path, "Positive risk regressor model")

    device_info = configure_tensorflow_device(device=args.device)

    raw_frame = pd.read_csv(args.input)
    if TARGET_COLUMN not in raw_frame.columns:
        raise ValueError(f"Input CSV must contain `{TARGET_COLUMN}`.")

    targets = load_prediction_targets(args.predictions_path, args.percent_threshold, args.max_points)
    if targets.empty:
        raise ValueError("No points match the recommendation target condition.")

    target_raw = targets[["POINT_ID"]].merge(raw_frame, on="POINT_ID", how="left", validate="one_to_one")
    if target_raw[TARGET_COLUMN].isna().any():
        raise ValueError("Some target POINT_ID values were not found in the input CSV.")

    zero_risk_frame = raw_frame[raw_frame[TARGET_COLUMN] == 0]
    background_size = min(args.background_size, len(zero_risk_frame))
    background_raw = zero_risk_frame.sample(n=background_size, random_state=args.random_state)

    preprocessor = OriginalTrainPreprocessor.load(args.preprocessor_path)
    target_features = preprocessor.transform_features(target_raw)[preprocessor.feature_columns].to_numpy(dtype=np.float32)
    background_features = preprocessor.transform_features(background_raw)[preprocessor.feature_columns].to_numpy(dtype=np.float32)

    classifier_model = tf.keras.models.load_model(classifier_model_path)
    regressor_model = tf.keras.models.load_model(regressor_model_path)

    classifier_shap = shap_values_for_model(
        classifier_model,
        background_features,
        target_features,
        nsamples=args.nsamples,
    )
    regressor_shap = shap_values_for_model(
        regressor_model,
        background_features,
        target_features,
        nsamples=args.nsamples,
    )

    recommendations = build_recommendations(
        targets=targets,
        feature_columns=preprocessor.feature_columns,
        classifier_shap=classifier_shap,
        regressor_shap=regressor_shap,
        top_n=args.top_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(args.output, index=False)

    no_recommendation_count = int((recommendations["추천개수"] == 0).sum())
    print("Point improvement recommendation complete.")
    print(f"Target points: {len(recommendations):,}")
    print(f"Recommendation features: {len(RECOMMENDATION_MAPPING):,}")
    print(f"Background samples: {len(background_features):,}")
    print(f"Points without common positive recommendation: {no_recommendation_count:,}")
    print(f"Output: {project_display_path(args.output)}")
    print(f"Device: {device_info.active_device}")


if __name__ == "__main__":
    main()
