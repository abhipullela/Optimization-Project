"""Historical walk-forward stock-return prediction pipeline.

Input:  data/processed/ml_features_primary.csv
Output: data/processed/predictions.csv

Assumes decisions are made after the current session's market close.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ml_features_primary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "predictions.csv"
TARGET_COLUMN = "target_next_day_return"
MODEL_VERSION = "ridge_walk_forward_v1"

FEATURE_COLUMNS = [
    "daily_return", "return_lag_1", "return_lag_5",
    "rolling_volatility_5", "rolling_volatility_20",
    "rolling_mean_return_5", "moving_average_ratio_5_20",
    "volume_change_1", "return_lag_2", "return_lag_10",
    "rolling_mean_return_20", "rolling_volatility_60",
    "momentum_20", "momentum_60", "price_to_ma20", "price_to_ma50",
    "intraday_range", "opening_gap", "relative_volume_20",
]

OUTPUT_COLUMNS = [
    "date", "ticker", "predicted_return", "predicted_horizon",
    "return_type", "model_version",
]


def load_features(path: Path) -> pd.DataFrame:
    """Load and validate the engineered feature dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. Run src/features/build_features.py first."
        )

    df = pd.read_csv(path)
    required = ["date", "Symbol", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["Symbol"] = df["Symbol"].astype("string").str.strip()
    df = df.dropna(subset=["date", "Symbol"])
    df = df[df["Symbol"] != ""].copy()
    if df.duplicated(["date", "Symbol"]).any():
        raise ValueError("Duplicate date-Symbol rows found in feature dataset.")

    for col in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["Symbol", "date"]).reset_index(drop=True)

    # A label is only available once the next observation for that stock occurs.
    df["target_end_date"] = df.groupby("Symbol", sort=False)["date"].shift(-1)
    return df


def make_model() -> Pipeline:
    """Scaling is fitted on training data only, preventing preprocessing leakage."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=3000.0)),
    ])


def walk_forward_predict(
    df: pd.DataFrame,
    min_train_rows: int = 200,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate expanding-window out-of-sample predictions by decision date."""
    X_all = df[FEATURE_COLUMNS]
    features_valid = X_all.notna().all(axis=1)
    labels_valid = df[TARGET_COLUMN].notna() & df["target_end_date"].notna()
    results: list[pd.DataFrame] = []

    for raw_date in sorted(df.loc[features_valid, "date"].unique()):
        decision_date = pd.Timestamp(raw_date).normalize()
        if start_date is not None and decision_date < start_date:
            continue
        if end_date is not None and decision_date > end_date:
            continue

        # The row's feature date must precede the decision date, and its target
        # must already have been observed by the decision date.
        train_mask = (
            features_valid & labels_valid
            & (df["date"] < decision_date)
            & (df["target_end_date"] <= decision_date)
        )
        prediction_mask = features_valid & (df["date"] == decision_date)
        train = df.loc[train_mask]
        predict = df.loc[prediction_mask]
        if len(train) < min_train_rows or predict.empty:
            continue

        model = make_model()
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        values = model.predict(predict[FEATURE_COLUMNS])
        results.append(pd.DataFrame({
            "date": decision_date.strftime("%Y-%m-%d"),
            "ticker": predict["Symbol"].astype(str).to_numpy(),
            "predicted_return": values,
            "predicted_horizon": "1_trading_day",
            "return_type": "simple_return",
            "model_version": MODEL_VERSION,
        }))

    if not results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return (
        pd.concat(results, ignore_index=True)[OUTPUT_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate historical walk-forward stock-return predictions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-train-rows", type=int, default=200,
        help="Minimum eligible stock-date training rows before predicting.",
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="Optional first decision date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Optional last decision date (YYYY-MM-DD).")
    args = parser.parse_args()
    if args.min_train_rows < 1:
        parser.error("--min-train-rows must be at least 1")

    try:
        start = pd.Timestamp(args.start_date).normalize() if args.start_date else None
        end = pd.Timestamp(args.end_date).normalize() if args.end_date else None
    except ValueError as exc:
        parser.error(f"Invalid date: {exc}")
    if start is not None and end is not None and start > end:
        parser.error("--start-date must be on or before --end-date")

    df = load_features(args.input)
    predictions = walk_forward_predict(df, args.min_train_rows, start, end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(f"Loaded feature rows: {len(df):,}")
    print(f"Stocks: {df['Symbol'].nunique():,}")
    print(f"Prediction rows: {len(predictions):,}")
    if not predictions.empty:
        print(f"Decision dates: {predictions['date'].min()} to {predictions['date'].max()}")
    else:
        print("No predictions generated. Check feature completeness and --min-train-rows.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
