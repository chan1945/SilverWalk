"""Preprocessing utilities for the original Seoul point training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = ["POINT_ID"]
COORDINATE_COLUMNS = ["위도", "경도"]
TARGET_COLUMN = "위험도"
TARGET_LOG_COLUMN = "위험도_log1p"
EXCLUDED_MODEL_COLUMNS = [*ID_COLUMNS, *COORDINATE_COLUMNS, TARGET_COLUMN]
BINARY_COLUMNS = [
    "전통시장여부",
    "보행자우선도로여부",
    "노인장애인보호구역여부",
    "횡단보도예고표시여부",
    "신호등설치여부",
]
RAW_SCALE_COLUMNS = ["제한속도"]


@dataclass
class OriginalTrainPreprocessor:
    """Fit and apply the MLP preprocessing rules for original_train_data."""

    upper_clip_quantile: float = 0.999
    feature_columns: list[str] = field(default_factory=list)
    binary_columns: list[str] = field(default_factory=list)
    log_scale_columns: list[str] = field(default_factory=list)
    raw_scale_columns: list[str] = field(default_factory=list)
    scale_columns: list[str] = field(default_factory=list)
    clip_upper_bounds: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    binary_modes: dict[str, int] = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    def fit(self, frame: pd.DataFrame) -> "OriginalTrainPreprocessor":
        validate_required_columns(frame)

        self.feature_columns = [
            column for column in frame.columns if column not in EXCLUDED_MODEL_COLUMNS
        ]
        self.binary_columns = [column for column in BINARY_COLUMNS if column in self.feature_columns]
        self.raw_scale_columns = [
            column for column in RAW_SCALE_COLUMNS if column in self.feature_columns
        ]
        self.log_scale_columns = [
            column
            for column in self.feature_columns
            if column not in self.binary_columns and column not in self.raw_scale_columns
        ]
        self.scale_columns = [*self.raw_scale_columns, *self.log_scale_columns]

        self._validate_numeric_nonnegative(frame)
        self._fit_imputation_values(frame)
        self._fit_clip_bounds(frame)

        scale_frame = self._make_scale_frame(frame)
        self.scaler.fit(scale_frame[self.scale_columns])
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        validate_required_columns(frame)

        metadata_frame = frame[[*ID_COLUMNS, *COORDINATE_COLUMNS, TARGET_COLUMN]].copy()
        metadata_frame[TARGET_LOG_COLUMN] = np.log1p(frame[TARGET_COLUMN].clip(lower=0))

        scale_frame = self._make_scale_frame(frame)
        scaled_values = self.scaler.transform(scale_frame[self.scale_columns])
        scaled_frame = pd.DataFrame(
            scaled_values,
            columns=self.scale_columns,
            index=frame.index,
        )

        binary_frame = pd.DataFrame(index=frame.index)
        for column in self.binary_columns:
            fill_value = self.binary_modes[column]
            binary_frame[column] = frame[column].fillna(fill_value).astype(int).clip(0, 1)

        feature_parts = {}
        for column in self.feature_columns:
            if column in binary_frame:
                feature_parts[column] = binary_frame[column]
            else:
                feature_parts[column] = scaled_frame[column]

        feature_frame = pd.DataFrame(feature_parts, index=frame.index)
        return pd.concat([metadata_frame, feature_frame], axis=1)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "OriginalTrainPreprocessor":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Unexpected preprocessor type: {type(loaded)!r}")
        return loaded

    def to_config(self) -> dict[str, Any]:
        return {
            "id_columns": ID_COLUMNS,
            "coordinate_columns": COORDINATE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "target_log_column": TARGET_LOG_COLUMN,
            "excluded_model_columns": EXCLUDED_MODEL_COLUMNS,
            "feature_columns": self.feature_columns,
            "binary_columns": self.binary_columns,
            "raw_scale_columns": self.raw_scale_columns,
            "log_scale_columns": self.log_scale_columns,
            "scale_columns": self.scale_columns,
            "upper_clip_quantile": self.upper_clip_quantile,
            "clip_upper_bounds": self.clip_upper_bounds,
            "medians": self.medians,
            "binary_modes": self.binary_modes,
        }

    def _fit_imputation_values(self, frame: pd.DataFrame) -> None:
        self.medians = {
            column: float(frame[column].median())
            for column in self.scale_columns
        }
        self.binary_modes = {}
        for column in self.binary_columns:
            modes = frame[column].dropna().mode()
            self.binary_modes[column] = int(modes.iloc[0]) if not modes.empty else 0

    def _fit_clip_bounds(self, frame: pd.DataFrame) -> None:
        self.clip_upper_bounds = {}
        for column in self.scale_columns:
            value = frame[column].quantile(self.upper_clip_quantile)
            self.clip_upper_bounds[column] = float(value)

    def _make_scale_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        scale_frame = pd.DataFrame(index=frame.index)
        for column in self.raw_scale_columns:
            values = frame[column].fillna(self.medians[column])
            values = values.clip(upper=self.clip_upper_bounds[column])
            scale_frame[column] = values

        for column in self.log_scale_columns:
            values = frame[column].fillna(self.medians[column])
            values = values.clip(lower=0, upper=self.clip_upper_bounds[column])
            scale_frame[column] = np.log1p(values)

        return scale_frame

    def _validate_numeric_nonnegative(self, frame: pd.DataFrame) -> None:
        non_numeric = [
            column for column in self.feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

        negative_columns = [
            column
            for column in self.log_scale_columns
            if (frame[column].dropna() < 0).any()
        ]
        if negative_columns:
            raise ValueError(f"log1p columns contain negative values: {negative_columns}")


def validate_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in EXCLUDED_MODEL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_risk_stratification_bins(risk: pd.Series) -> pd.Series:
    """Create stable bins so zero-heavy risk values stay balanced across splits."""
    return pd.cut(
        risk,
        bins=[-np.inf, 0, 1, 5, 20, 50, np.inf],
        labels=["zero", "0-1", "1-5", "5-20", "20-50", "50+"],
        include_lowest=True,
    ).astype(str)


def split_original_train_data(
    frame: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    stratify_bins = make_risk_stratification_bins(frame[TARGET_COLUMN])
    temp_size = val_size + test_size
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=temp_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_bins,
    )

    relative_test_size = test_size / temp_size
    temp_bins = make_risk_stratification_bins(temp_frame[TARGET_COLUMN])
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
        stratify=temp_bins,
    )

    return (
        train_frame.sort_index().reset_index(drop=True),
        val_frame.sort_index().reset_index(drop=True),
        test_frame.sort_index().reset_index(drop=True),
    )


def build_preprocessed_splits(
    frame: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[OriginalTrainPreprocessor, dict[str, pd.DataFrame]]:
    train_frame, val_frame, test_frame = split_original_train_data(
        frame,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = OriginalTrainPreprocessor().fit(train_frame)
    splits = {
        "train": preprocessor.transform(train_frame),
        "val": preprocessor.transform(val_frame),
        "test": preprocessor.transform(test_frame),
    }
    return preprocessor, splits
