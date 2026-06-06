"""Common metrics for SilverWalk experiment comparisons.

이 모듈은 기법별 성능 비교 notebook에서 같은 기준으로 지표를 계산하기 위한
DataFrame 기반 유틸리티를 제공한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error


def _validate_top_percent(top_percent: float) -> None:
    """top_percent 값이 0과 1 사이인지 확인한다."""
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be greater than 0 and less than or equal to 1.")


def _top_count(row_count: int, top_percent: float) -> int:
    """전체 행 개수와 비율을 기준으로 Top-N 개수를 계산한다."""
    _validate_top_percent(top_percent)
    if row_count <= 0:
        return 0
    return max(1, int(np.ceil(row_count * top_percent)))


def _drop_metric_na(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """지표 계산에 필요한 컬럼의 결측값을 제거한다."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    return df.dropna(subset=required_columns).copy()


def precision_at_top_percent(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    top_percent: float = 0.10,
    positive_label: int = 1,
) -> float:
    """예측 점수 상위 top_percent 안에서 실제 양성 비율을 계산한다."""
    metric_df = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
    top_n = _top_count(len(metric_df), top_percent)
    if top_n == 0:
        return float("nan")

    top_rows = metric_df.sort_values("y_score", ascending=False).head(top_n)
    return float((top_rows["y_true"] == positive_label).mean())


def recall_at_top_percent(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    top_percent: float = 0.10,
    positive_label: int = 1,
) -> float:
    """전체 실제 양성 중 예측 점수 상위 top_percent에 포함된 비율을 계산한다."""
    metric_df = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
    positive_count = int((metric_df["y_true"] == positive_label).sum())
    if positive_count == 0:
        return float("nan")

    top_n = _top_count(len(metric_df), top_percent)
    top_rows = metric_df.sort_values("y_score", ascending=False).head(top_n)
    top_positive_count = int((top_rows["y_true"] == positive_label).sum())
    return float(top_positive_count / positive_count)


def classification_metrics_from_dataframe(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    y_score_col: str,
    y_pred_col: str | None = None,
    threshold: float = 0.5,
    top_percent: float = 0.10,
    positive_label: int = 1,
    prefix: str = "",
) -> dict[str, float]:
    """분류 모델 평가 지표를 DataFrame에서 계산한다.

    y_pred_col을 지정하지 않으면 y_score_col >= threshold 기준으로 F1-score를 계산한다.
    """
    required_columns = [y_true_col, y_score_col]
    if y_pred_col is not None:
        required_columns.append(y_pred_col)
    metric_df = _drop_metric_na(df, required_columns)

    if metric_df.empty:
        return {
            f"{prefix}pr_auc": float("nan"),
            f"{prefix}f1": float("nan"),
            f"{prefix}precision@top{int(top_percent * 100)}pct": float("nan"),
            f"{prefix}recall@top{int(top_percent * 100)}pct": float("nan"),
        }

    y_true_binary = (metric_df[y_true_col] == positive_label).astype(int)
    y_score = metric_df[y_score_col].astype(float)

    if y_pred_col is None:
        y_pred_binary = (y_score >= threshold).astype(int)
    else:
        y_pred_binary = (metric_df[y_pred_col] == positive_label).astype(int)

    if y_true_binary.nunique() < 2:
        pr_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true_binary, y_score))

    top_suffix = f"top{int(top_percent * 100)}pct"
    return {
        f"{prefix}pr_auc": pr_auc,
        f"{prefix}f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        f"{prefix}precision@{top_suffix}": precision_at_top_percent(
            y_true_binary,
            y_score,
            top_percent=top_percent,
            positive_label=1,
        ),
        f"{prefix}recall@{top_suffix}": recall_at_top_percent(
            y_true_binary,
            y_score,
            top_percent=top_percent,
            positive_label=1,
        ),
    }


def high_precision_at_top_percent(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    top_percent: float = 0.10,
    high_threshold: float = 50.0,
) -> float:
    """예측 위험도 상위 top_percent 안에서 실제 고위험 포인트 비율을 계산한다."""
    metric_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    top_n = _top_count(len(metric_df), top_percent)
    if top_n == 0:
        return float("nan")

    top_rows = metric_df.sort_values("y_pred", ascending=False).head(top_n)
    return float((top_rows["y_true"] > high_threshold).mean())


def high_recall_at_top_percent(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    top_percent: float = 0.10,
    high_threshold: float = 50.0,
) -> float:
    """전체 실제 고위험 포인트 중 예측 위험도 상위 top_percent에 포함된 비율을 계산한다."""
    metric_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    high_count = int((metric_df["y_true"] > high_threshold).sum())
    if high_count == 0:
        return float("nan")

    top_n = _top_count(len(metric_df), top_percent)
    top_rows = metric_df.sort_values("y_pred", ascending=False).head(top_n)
    top_high_count = int((top_rows["y_true"] > high_threshold).sum())
    return float(top_high_count / high_count)


def regression_metrics_from_dataframe(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str,
    top_percent: float = 0.10,
    high_threshold: float = 50.0,
    prefix: str = "",
) -> dict[str, float]:
    """회귀 모델 평가 지표를 DataFrame에서 계산한다."""
    metric_df = _drop_metric_na(df, [y_true_col, y_pred_col])
    if metric_df.empty:
        return {
            f"{prefix}mae": float("nan"),
            f"{prefix}rmse": float("nan"),
            f"{prefix}high_precision@top{int(top_percent * 100)}pct": float("nan"),
            f"{prefix}high_recall@top{int(top_percent * 100)}pct": float("nan"),
        }

    y_true = metric_df[y_true_col].astype(float)
    y_pred = metric_df[y_pred_col].astype(float)
    top_suffix = f"top{int(top_percent * 100)}pct"

    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}high_precision@{top_suffix}": high_precision_at_top_percent(
            y_true,
            y_pred,
            top_percent=top_percent,
            high_threshold=high_threshold,
        ),
        f"{prefix}high_recall@{top_suffix}": high_recall_at_top_percent(
            y_true,
            y_pred,
            top_percent=top_percent,
            high_threshold=high_threshold,
        ),
    }


def experiment_row(
    experiment_name: str,
    metrics: dict[str, Any],
    *,
    model_name: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """실험명, 모델명, 지표를 하나의 결과 행으로 합친다."""
    row: dict[str, Any] = {"experiment": experiment_name}
    if model_name is not None:
        row["model"] = model_name
    row.update(metrics)
    if notes is not None:
        row["notes"] = notes
    return row
