#!/usr/bin/env python3
"""Preprocess data/original_train_data for MLP training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from silverwalk_ai.data.paths import PATHS, ensure_project_dirs
from silverwalk_ai.features.preprocessing import build_preprocessed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PATHS.data / "original_train_data" / "seoul_road_points.csv",
        help="Source CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.processed,
        help="Directory for preprocessed split CSV files.",
    )
    parser.add_argument(
        "--preprocessor-dir",
        type=Path,
        default=PATHS.preprocessors,
        help="Directory for fitted preprocessor artifacts.",
    )
    parser.add_argument("--prefix", default="original_train", help="Output filename prefix.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preprocessor_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    preprocessor, splits = build_preprocessed_splits(
        frame,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    output_paths = {}
    for split_name, split_frame in splits.items():
        output_path = args.output_dir / f"{args.prefix}_{split_name}_preprocessed.csv"
        split_frame.to_csv(output_path, index=False)
        output_paths[split_name] = str(output_path.relative_to(PATHS.root))

    preprocessor_path = args.preprocessor_dir / f"{args.prefix}_preprocessor.joblib"
    config_path = args.preprocessor_dir / f"{args.prefix}_preprocess_config.json"
    preprocessor.save(preprocessor_path)

    config = preprocessor.to_config()
    config.update(
        {
            "source_path": str(args.input.relative_to(PATHS.root)),
            "output_paths": output_paths,
            "preprocessor_path": str(preprocessor_path.relative_to(PATHS.root)),
            "split_sizes": {name: len(split_frame) for name, split_frame in splits.items()},
            "val_size": args.val_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Preprocessing complete.")
    print(f"Source rows: {len(frame):,}")
    print(f"Model feature columns: {len(preprocessor.feature_columns):,}")
    print(f"Binary columns: {len(preprocessor.binary_columns):,}")
    print(f"Scaled columns: {len(preprocessor.scale_columns):,}")
    for split_name, output_path in output_paths.items():
        print(f"{split_name}: {config['split_sizes'][split_name]:,} rows -> {output_path}")
    print(f"Preprocessor: {preprocessor_path.relative_to(PATHS.root)}")
    print(f"Config: {config_path.relative_to(PATHS.root)}")


if __name__ == "__main__":
    main()
