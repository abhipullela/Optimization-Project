"""Historical walk-forward cross-sectional rank prediction pipeline.

Input:  data/processed/ml_features_primary.csv
Output: data/processed/predictions_ranker.csv

Uses XGBRegressor targeting cross-sectionally neutralized returns (Z-scores)
to capture relative outperformance stably on small asset universes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ml_features_primary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "predictions_ranker.csv"
TARGET_COLUMN = "target_next_day_return"
TARGET_Z = "target_z_score"
MODEL_VERSION = "xgbregressor_zscore_v4"

FEATURE_COLUMNS = [
    "daily_return", "return_lag_1", "return_lag_5", 
    "rolling_volatility_5", "rolling_volatility_20",
    "rolling_mean_return_5", "moving_average_ratio_5_20", 
    "volume_change_1", "return_lag_2", "return_lag_10", 
    "rolling_mean_return_20", "rolling_volatility_60", 
    "momentum_20", "momentum_60", "price_to_ma20",
    "price_to_ma50", "intraday_range", "opening_gap", 
    "relative_volume_20"
]

OUTPUT_COLUMNS = [
    "date", "ticker", "predicted_return", "predicted_horizon",
    "return_type", "model_version"
]


def load_and_neutralize(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    
    df = df.sort_values(["Symbol", "date"])
    df["target_end_date"] = df.groupby("Symbol", sort=False)["date"].shift(-1)
    
    print("Applying cross-sectional neutralization to features AND target...")
    def daily_z_score(x):
        std = x.std()
        if pd.isna(std) or std == 0:
            return x - x.mean()
        return (x - x.mean()) / std

    # Neutralize features
    df[FEATURE_COLUMNS] = df.groupby("date")[FEATURE_COLUMNS].transform(daily_z_score)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    
    # CRITICAL: Neutralize the target to force the regressor to learn relative outperformance
    df[TARGET_Z] = df.groupby("date")[TARGET_COLUMN].transform(daily_z_score)
    df[TARGET_Z] = df[TARGET_Z].fillna(0.0)
    
    return df


def walk_forward_predict(
    df: pd.DataFrame,
    min_train_rows: int = 500,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    retrain_freq: int = 21 
) -> pd.DataFrame:
    X_all = df[FEATURE_COLUMNS]
    features_valid = X_all.notna().all(axis=1)
    labels_valid = df[TARGET_Z].notna() & df["target_end_date"].notna()
    results: list[pd.DataFrame] = []

    valid_dates = sorted(df.loc[features_valid, "date"].unique())
    
    active_model = None
    days_since_retrain = retrain_freq

    for raw_date in valid_dates:
        decision_date = pd.Timestamp(raw_date).normalize()
        if start_date is not None and decision_date < start_date:
            continue
        if end_date is not None and decision_date > end_date:
            continue

        prediction_mask = features_valid & (df["date"] == decision_date)
        predict = df.loc[prediction_mask]
        
        if predict.empty:
            continue

        if days_since_retrain >= retrain_freq or active_model is None:
            train_mask = (
                features_valid & labels_valid
                & (df["date"] < decision_date)
                & (df["target_end_date"] <= decision_date)
            )
            train = df.loc[train_mask].copy()
            
            if len(train) >= min_train_rows:
                
                # Switch to XGBRegressor targeting the Z-score with balanced parameters
                active_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,           # Slightly deeper than before, but still shallow
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1
                )
                
                active_model.fit(
                    train[FEATURE_COLUMNS], 
                    train[TARGET_Z], 
                    verbose=False
                )
                days_since_retrain = 0

        if active_model is not None:
            # Predict the Z-score directly
            z_score_preds = active_model.predict(predict[FEATURE_COLUMNS])
            
            # Scale the Z-score prediction directly into a pseudo-return (1.5% vol)
            pseudo_returns = z_score_preds * 0.015
            
            results.append(pd.DataFrame({
                "date": decision_date.strftime("%Y-%m-%d"),
                "ticker": predict["Symbol"].astype(str).to_numpy(),
                "predicted_return": pseudo_returns,
                "predicted_horizon": "1_trading_day",
                "return_type": "simple_return",
                "model_version": MODEL_VERSION,
            }))
        
        days_since_retrain += 1

    if not results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
        
    return (
        pd.concat(results, ignore_index=True)[OUTPUT_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Z-Score ML")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--retrain-freq", type=int, default=21)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)

    args = parser.parse_args()

    try:
        start = pd.Timestamp(args.start_date).normalize() if args.start_date else None
        end = pd.Timestamp(args.end_date).normalize() if args.end_date else None
    except ValueError as exc:
        parser.error(f"Invalid date: {exc}")

    print("Loading features...")
    df = load_and_neutralize(args.input)
    
    print("Training XGBRegressor on cross-sectional target...")
    predictions = walk_forward_predict(df, args.min_train_rows, start, end, args.retrain_freq)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(f"\nLoaded feature rows: {len(df):,}")
    print(f"Stocks: {df['Symbol'].nunique():,}")
    print(f"Prediction rows: {len(predictions):,}")
    if not predictions.empty:
        print(f"Decision dates: {predictions['date'].min()} to {predictions['date'].max()}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()