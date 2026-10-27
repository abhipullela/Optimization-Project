"""
Feature engineering pipeline for stock return prediction.

Input:
    data/processed/cleaned_primary_master.csv

Output:
    data/processed/ml_features_primary.csv

Responsibilities:
    - Validate and load cleaned stock data.
    - Generate return, momentum, volatility, price and volume features.
    - Generate the next-observation return target.
    - Save the complete feature dataset.
    - Provide select_model_matrix() for preparing X and y for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_primary_master.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "ml_features_primary.csv"
)

TARGET_COLUMN = "target_next_day_return"

FEATURE_COLUMNS = [
    # Existing features
    "daily_return",
    "return_lag_1",
    "return_lag_5",
    "rolling_volatility_5",
    "rolling_volatility_20",
    "rolling_mean_return_5",
    "moving_average_ratio_5_20",
    "volume_change_1",

    # Additional return features
    "return_lag_2",
    "return_lag_10",

    # Additional statistical features
    "rolling_mean_return_20",
    "rolling_volatility_60",

    # Momentum features
    "momentum_20",
    "momentum_60",

    # Price features
    "price_to_ma20",
    "price_to_ma50",
    "intraday_range",
    "opening_gap",

    # Volume features
    "relative_volume_20",
]


COLUMN_ALIASES = {
    "symbol": "Symbol",
    "date": "date",
    "open": "open",
    "open price": "open",
    "high": "high",
    "high price": "high",
    "low": "low",
    "low price": "low",
    "close": "close",
    "close price": "close",
    "volume": "volume",
    "total traded quantity": "volume",
}


# ============================================================
# 1. Load and validate cleaned data
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize known column names."""

    rename_map = {}

    for column in df.columns:
        normalized = " ".join(
            str(column).strip().lower().replace("_", " ").split()
        )

        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]

    df = df.rename(columns=rename_map)

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate columns after normalization: {duplicates}"
        )

    return df


def load_cleaned_data(input_path: Path) -> pd.DataFrame:
    """Load the cleaned master CSV and validate required columns."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run preprocess.py first."
        )

    df = pd.read_csv(input_path)
    df = normalize_columns(df)

    required_columns = [
        "Symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["Symbol"] = df["Symbol"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Symbol", "date", "close"])

    df = df[
        (df["Symbol"] != "")
        & (df["close"] > 0)
    ].copy()

    duplicate_mask = df.duplicated(
        subset=["Symbol", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = df.loc[
            duplicate_mask, ["Symbol", "date"]
        ].head(5)

        raise ValueError(
            "Duplicate Symbol-Date records found. "
            "Resolve them in preprocess.py first.\n"
            f"{examples.to_string(index=False)}"
        )

    df = df.sort_values(
        ["Symbol", "date"]
    ).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows.")
    print(f"Unique stocks: {df['Symbol'].nunique():,}")

    return df


# ============================================================
# 2. Return features
# ============================================================

def build_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns and lagged daily returns."""

    grouped = df.groupby("Symbol", sort=False)

    df["daily_return"] = grouped["close"].pct_change(
        fill_method=None
    )

    df["timestamp_ist"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "T15:30:00+05:30"
    )

    for lag in [1, 2, 5, 10]:
        df[f"return_lag_{lag}"] = (
            df.groupby("Symbol", sort=False)["daily_return"]
            .shift(lag)
        )

    return df


# ============================================================
# 3. Momentum features
# ============================================================

def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 20-observation and 60-observation price momentum."""

    grouped_close = df.groupby(
        "Symbol", sort=False
    )["close"]

    for window in [20, 60]:
        previous_close = grouped_close.shift(window)

        df[f"momentum_{window}"] = (
            df["close"] / previous_close
        ) - 1

    return df


# ============================================================
# 4. Volatility and rolling mean features
# ============================================================

def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling volatility and mean daily returns."""

    grouped_returns = df.groupby(
        "Symbol", sort=False
    )["daily_return"]

    for window in [5, 20, 60]:
        df[f"rolling_volatility_{window}"] = (
            grouped_returns
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )

    for window in [5, 20]:
        df[f"rolling_mean_return_{window}"] = (
            grouped_returns
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


# ============================================================
# 5. Price features
# ============================================================

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving-average ratios, intraday range and opening gap."""

    grouped = df.groupby("Symbol", sort=False)

    moving_averages = {}

    for window in [5, 20, 50]:
        moving_averages[window] = (
            grouped["close"]
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    ma5 = moving_averages[5]
    ma20 = moving_averages[20]
    ma50 = moving_averages[50]

    df["moving_average_ratio_5_20"] = (ma5 / ma20) - 1
    df["price_to_ma20"] = (df["close"] / ma20) - 1
    df["price_to_ma50"] = (df["close"] / ma50) - 1

    previous_close = grouped["close"].shift(1)

    df["intraday_range"] = (
        (df["high"] - df["low"]) / previous_close
    )

    df["opening_gap"] = (
        (df["open"] - previous_close) / previous_close
    )

    return df


# ============================================================
# 6. Volume features
# ============================================================

def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume changes and relative volume."""

    grouped_volume = df.groupby(
        "Symbol", sort=False
    )["volume"]

    df["volume_change_1"] = grouped_volume.pct_change(
        fill_method=None
    )

    rolling_volume_mean = (
        grouped_volume
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["relative_volume_20"] = (
        df["volume"] / rolling_volume_mean
    )

    return df


# ============================================================
# 7. Prediction target
# ============================================================

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the next-observation daily-return target per stock.

    The target for a row is the next available daily return for
    the same stock, not necessarily the next calendar day.
    """

    df[TARGET_COLUMN] = (
        df.groupby("Symbol", sort=False)["daily_return"]
        .shift(-1)
    )

    return df


# ============================================================
# 8. Select output columns
# ============================================================

def select_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select identifiers, model features and target."""

    identifier_columns = ["Symbol", "date"]

    if "timestamp_ist" in df.columns:
        identifier_columns.append("timestamp_ist")

    required = identifier_columns + FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Expected output columns were not generated: {missing}"
        )

    output = df[required].copy()

    output = output.replace([np.inf, -np.inf], np.nan)

    return output


# ============================================================
# 9. Prepare model matrix
# ============================================================

def select_model_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X and y for model training.

    Rows with missing feature values or targets are excluded from
    the returned training matrix, but not from the saved feature CSV.
    """

    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required feature/target columns: {sorted(missing)}"
        )

    X = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()

    return X.loc[valid].copy(), y.loc[valid].copy()


# ============================================================
# 10. Save feature dataset
# ============================================================

def save_features(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the generated feature dataset."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_path, index=False)

    print(f"\nFeature dataset saved: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        "Rows with valid target: "
        f"{df[TARGET_COLUMN].notna().sum():,}"
    )


# ============================================================
# 11. Pipeline entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stock-return prediction features."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input cleaned master CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ML features CSV.",
    )

    args = parser.parse_args()

    df = load_cleaned_data(args.input)

    df = build_return_features(df)
    df = build_momentum_features(df)
    df = build_volatility_features(df)
    df = build_price_features(df)
    df = build_volume_features(df)
    df = build_target(df)

    ml_dataset = select_ml_columns(df)

    save_features(ml_dataset, args.output)


if __name__ == "__main__":
    main()
