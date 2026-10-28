"""Chronologically tune Ridge alpha using walk-forward validation.

Run from the repository root:
python src/models/tune_ridge.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

This tunes only on the selected validation period. Keep a later, untouched test
period for final evaluation.
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
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ridge_alpha_tuning.csv"
TARGET_COLUMN = "target_next_day_return"

FEATURE_COLUMNS = [
    "daily_return", "return_lag_1", "return_lag_5",
    "rolling_volatility_5", "rolling_volatility_20",
    "rolling_mean_return_5", "moving_average_ratio_5_20",
    "volume_change_1", "return_lag_2", "return_lag_10",
    "rolling_mean_return_20", "rolling_volatility_60",
    "momentum_20", "momentum_60", "price_to_ma20", "price_to_ma50",
    "intraday_range", "opening_gap", "relative_volume_20",
]


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)
    required = ["date", "Symbol", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Feature file missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["Symbol"] = df["Symbol"].astype("string").str.strip()
    df = df.dropna(subset=["date", "Symbol"])
    df = df[df["Symbol"] != ""].copy()
    if df.duplicated(["date", "Symbol"]).any():
        raise ValueError("Duplicate (date, Symbol) rows in feature dataset.")

    for col in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["Symbol", "date"]).reset_index(drop=True)
    # The label for a row is known only when the next observation for that
    # stock has occurred. This matches return_prediction.py's cutoff logic.
    df["target_end_date"] = df.groupby("Symbol", sort=False)["date"].shift(-1)
    return df


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = actual - predicted
    nonzero = actual != 0
    direction = (
        float((np.sign(actual[nonzero]) == np.sign(predicted[nonzero])).mean())
        if nonzero.any() else np.nan
    )
    baseline_errors = actual  # baseline prediction is zero
    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {
        "n_predictions": int(len(actual)),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": direction,
        "mean_actual_return": float(np.mean(actual)),
        "mean_predicted_return": float(np.mean(predicted)),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "mae_improvement_pct": (
            (baseline_mae - mae) / baseline_mae * 100
            if baseline_mae else np.nan
        ),
        "rmse_improvement_pct": (
            (baseline_rmse - rmse) / baseline_rmse * 100
            if baseline_rmse else np.nan
        ),
    }


def tune_alpha(
    df: pd.DataFrame,
    alphas: list[float],
    min_train_rows: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS]
    features_valid = X.notna().all(axis=1)
    labels_valid = df[TARGET_COLUMN].notna() & df["target_end_date"].notna()
    date_values = sorted(
        d for d in df.loc[features_valid, "date"].unique()
        if start_date <= pd.Timestamp(d).normalize() <= end_date
    )

    rows = []
    for alpha in alphas:
        actuals: list[float] = []
        predictions: list[float] = []
        evaluated_dates: list[pd.Timestamp] = []
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ])

        for raw_date in date_values:
            decision_date = pd.Timestamp(raw_date).normalize()
            train_mask = (
                features_valid & labels_valid
                & (df["date"] < decision_date)
                & (df["target_end_date"] <= decision_date)
            )
            predict_mask = features_valid & (df["date"] == decision_date)
            train = df.loc[train_mask]
            predict = df.loc[predict_mask]
            if len(train) < min_train_rows or predict.empty:
                continue

            model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
            predicted = model.predict(predict[FEATURE_COLUMNS])
            valid_target = predict[TARGET_COLUMN].notna().to_numpy()
            # Score only predictions with a known target. The target itself
            # is never passed to model.predict().
            if valid_target.any():
                actuals.extend(
                    predict.loc[valid_target, TARGET_COLUMN].astype(float).tolist()
                )
                predictions.extend(predicted[valid_target].astype(float).tolist())
                evaluated_dates.extend([decision_date] * int(valid_target.sum()))

        if not actuals:
            rows.append({"alpha": alpha, "n_predictions": 0, "error": "No eligible validation predictions"})
            continue

        metrics = metric_row(np.asarray(actuals), np.asarray(predictions))
        rows.append({
            "alpha": alpha,
            **metrics,
            "n_dates": len(set(evaluated_dates)),
            "validation_start": min(evaluated_dates).date().isoformat(),
            "validation_end": max(evaluated_dates).date().isoformat(),
        })

    results = pd.DataFrame(rows)
    if "mae" in results.columns:
        results = results.sort_values(
            ["mae", "rmse"], ascending=[True, True], na_position="last"
        ).reset_index(drop=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", required=True, help="Validation start date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Validation end date, YYYY-MM-DD")
    parser.add_argument(
        "--alphas", default="0.001,0.01,0.1,1,10,100",
        help="Comma-separated positive alpha values (default: 0.001,0.01,0.1,1,10,100)",
    )
    parser.add_argument("--min-train-rows", type=int, default=200)
    args = parser.parse_args()

    try:
        start = pd.Timestamp(args.start_date).normalize()
        end = pd.Timestamp(args.end_date).normalize()
        alphas = [float(value.strip()) for value in args.alphas.split(",")]
    except (ValueError, TypeError) as exc:
        parser.error(f"Invalid date or alpha value: {exc}")

    if start > end:
        parser.error("--start-date must be on or before --end-date")
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        parser.error("All alpha values must be finite and greater than zero")
    if args.min_train_rows < 1:
        parser.error("--min-train-rows must be at least 1")

    df = load_features(args.input)
    results = tune_alpha(df, alphas, args.min_train_rows, start, end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"Validation period requested: {start.date()} to {end.date()}")
    print(f"Minimum training rows: {args.min_train_rows}")
    print("\nResults sorted by MAE (lower is better):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to: {args.output}")
    if not results.empty and "mae" in results.columns and results["mae"].notna().any():
        best = results.dropna(subset=["mae"]).iloc[0]
        print(
            f"\nBest validation alpha by MAE: {best['alpha']} "
            f"(MAE={best['mae']:.6f}, RMSE={best['rmse']:.6f}, "
            f"MAE improvement vs zero baseline={best['mae_improvement_pct']:.3f}%)"
        )
        print("Choose alpha using validation results only; evaluate it once on a later untouched test period.")


if __name__ == "__main__":
    main()
