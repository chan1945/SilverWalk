#!/usr/bin/env python3
"""Train the SilverWalk MLP risk model from preprocessed split CSV files."""

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
    evaluate_predictions,
    history_to_frame,
    predict_risk,
    train_mlp_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=PATHS.processed / "original_train_train_preprocessed.csv",
        help="Preprocessed train CSV path.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=PATHS.processed / "original_train_val_preprocessed.csv",
        help="Preprocessed validation CSV path.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PATHS.processed / "original_train_test_preprocessed.csv",
        help="Preprocessed test CSV path.",
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=PATHS.preprocessors / "original_train_preprocess_config.json",
        help="Preprocessing config JSON path.",
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
        "--limit-rows",
        type=int,
        default=None,
        help="Optional row limit per split for quick smoke tests.",
    )
    return parser.parse_args()


def read_split(path: Path, limit_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=limit_rows)


def project_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PATHS.root))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    PATHS.predictions.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    with args.preprocess_config.open("r", encoding="utf-8") as file:
        preprocess_config = json.load(file)
    feature_columns = preprocess_config["feature_columns"]

    train_frame = read_split(args.train, args.limit_rows)
    val_frame = read_split(args.val, args.limit_rows)
    test_frame = read_split(args.test, args.limit_rows)

    config = MLPTrainingConfig(
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        random_state=args.random_state,
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
            "training_config": config.to_dict(),
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
    print("Test metrics:")
    print(json.dumps(metrics["risk_scale"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
