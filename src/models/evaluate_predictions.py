"""Evaluate out-of-sample walk-forward return predictions.

Reads:
  data/processed/predictions.csv
  data/processed/ml_features_primary.csv

Writes:
  data/processed/model_metrics_summary.csv
  data/processed/model_metrics_by_ticker.csv
  data/processed/model_metrics_by_date.csv

Metrics are computed only where an out-of-sample prediction can be matched
to its realized next-observed-session return.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "predictions.csv"
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "ml_features_primary.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "model_metrics_summary.csv"
DEFAULT_BY_TICKER = PROJECT_ROOT / "data" / "processed" / "model_metrics_by_ticker.csv"
DEFAULT_BY_DATE = PROJECT_ROOT / "data" / "processed" / "model_metrics_by_date.csv"

KEY_COLUMNS = ["date", "ticker"]


def metric_row(group: pd.DataFrame, label: str) -> dict:
    """Calculate regression and directional metrics for a matched sample."""
    actual = group["actual_return"].to_numpy(dtype=float)
    predicted = group["predicted_return"].to_numpy(dtype=float)

    if len(actual) == 0:
        return {
            "model": label,
            "n_predictions": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "directional_accuracy": np.nan,
            "mean_actual_return": np.nan,
            "mean_predicted_return": np.nan,
            "prediction_bias": np.nan,
            "median_absolute_error": np.nan,
            "correlation": np.nan,
        }

    errors = predicted - actual
    nonzero_actual = actual != 0
    direction_accuracy = (
        float(
            (
                np.sign(actual[nonzero_actual])
                == np.sign(predicted[nonzero_actual])
            ).mean()
        )
        if nonzero_actual.any() and label != "zero_return_baseline"
        else np.nan
    )

    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if len(actual) >= 2
        and np.std(actual) > 0
        and np.std(predicted) > 0
        else np.nan
    )

    return {
        "model": label,
        "n_predictions": int(len(actual)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "directional_accuracy": direction_accuracy,
        "mean_actual_return": float(np.mean(actual)),
        "mean_predicted_return": float(np.mean(predicted)),
        # Positive bias means predictions are higher than actual returns.
        "prediction_bias": float(np.mean(errors)),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "correlation": correlation,
    }


def normalize_key_columns(df: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
    """Normalize date and ticker keys consistently."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df[symbol_column].astype("string").str.strip().str.upper()
    return df


def validate_unique_keys(df: pd.DataFrame, label: str) -> None:
    """Raise rather than silently discard duplicate date/ticker records."""
    duplicate_mask = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, KEY_COLUMNS].head(10)
        raise ValueError(
            f"{label} contains duplicate (date, ticker) keys. "
            "Resolve these records at the source instead of dropping them.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ticker-output", type=Path, default=DEFAULT_BY_TICKER)
    parser.add_argument("--date-output", type=Path, default=DEFAULT_BY_DATE)
    args = parser.parse_args()

    predictions_raw = pd.read_csv(args.predictions)
    features_raw = pd.read_csv(args.features)

    required_predictions = {"date", "ticker", "predicted_return"}
    required_features = {"date", "Symbol", "target_next_day_return"}

    missing_predictions = required_predictions - set(predictions_raw.columns)
    missing_features = required_features - set(features_raw.columns)
    if missing_predictions:
        raise ValueError(
            f"Predictions file missing columns: {sorted(missing_predictions)}"
        )
    if missing_features:
        raise ValueError(
            f"Features file missing columns: {sorted(missing_features)}"
        )

    predictions = normalize_key_columns(predictions_raw, "ticker")
    features = normalize_key_columns(features_raw, "Symbol")

    predictions["predicted_return"] = pd.to_numeric(
        predictions["predicted_return"], errors="coerce"
    )
    features["target_next_day_return"] = pd.to_numeric(
        features["target_next_day_return"], errors="coerce"
    )

    # Invalid keys cannot be matched and should be reported, not silently lost.
    invalid_prediction_keys = predictions["date"].isna() | predictions["ticker"].isna() | (predictions["ticker"] == "")
    invalid_feature_keys = features["date"].isna() | features["ticker"].isna() | (features["ticker"] == "")

    print("Input diagnostics")
    print(f"  Prediction rows read: {len(predictions):,}")
    print(f"  Feature rows read: {len(features):,}")
    print(f"  Predictions with invalid date/ticker keys: {int(invalid_prediction_keys.sum()):,}")
    print(f"  Feature rows with invalid date/ticker keys: {int(invalid_feature_keys.sum()):,}")

    predictions = predictions.loc[~invalid_prediction_keys].copy()
    features = features.loc[~invalid_feature_keys].copy()

    validate_unique_keys(predictions, "Predictions")
    validate_unique_keys(features, "Features")

    actuals = features[["date", "ticker", "target_next_day_return"]].rename(
        columns={"target_next_day_return": "actual_return"}
    )

    # Outer join lets us count unmatched predictions and feature targets.
    joined = predictions.merge(
        actuals,
        on=KEY_COLUMNS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    prediction_only = joined["_merge"].eq("left_only")
    actual_only = joined["_merge"].eq("right_only")
    matched = joined["_merge"].eq("both")

    matched_rows = joined.loc[matched].copy()
    invalid_prediction_values = (
        matched_rows["predicted_return"].isna()
        | ~np.isfinite(matched_rows["predicted_return"])
    )
    invalid_actual_values = (
        matched_rows["actual_return"].isna()
        | ~np.isfinite(matched_rows["actual_return"])
    )

    evaluable = matched_rows.loc[
        ~invalid_prediction_values & ~invalid_actual_values
    ].copy()

    # Dates with feature rows but no predictions are counted as actual-only,
    # even if the feature target is missing; report valid target coverage too.
    valid_actual_targets = actuals[
        actuals["actual_return"].notna()
        & np.isfinite(actuals["actual_return"].fillna(np.nan))
    ]

    diagnostics = {
        "predictions_without_matching_feature_key": int(prediction_only.sum()),
        "feature_keys_without_prediction": int(actual_only.sum()),
        "matched_keys": int(matched.sum()),
        "matched_rows_with_invalid_prediction": int(invalid_prediction_values.sum()),
        "matched_rows_with_invalid_actual": int(invalid_actual_values.sum()),
        "evaluable_matched_predictions": int(len(evaluable)),
        "feature_rows_with_valid_target": int(len(valid_actual_targets)),
    }

    summary_rows = [
        metric_row(evaluable, "ridge_walk_forward"),
        metric_row(
            evaluable.assign(predicted_return=0.0),
            "zero_return_baseline",
        ),
    ]
    summary = pd.DataFrame(summary_rows)

    ticker_rows = []
    for ticker, group in evaluable.groupby("ticker", sort=True):
        ridge_row = metric_row(group, "ridge_walk_forward")
        ridge_row["ticker"] = ticker
        ticker_rows.append(ridge_row)

        baseline_row = metric_row(
            group.assign(predicted_return=0.0),
            "zero_return_baseline",
        )
        baseline_row["ticker"] = ticker
        ticker_rows.append(baseline_row)

    by_ticker = pd.DataFrame(ticker_rows)

    # Per-date Ridge metrics. Daily cross-sectional correlation is omitted
    # when there are fewer than two valid observations or no variation.
    date_rows = []
    for date, group in evaluable.groupby("date", sort=True):
        row = metric_row(group, "ridge_walk_forward")
        row["date"] = date.date().isoformat()
        date_rows.append(row)
    by_date = pd.DataFrame(date_rows)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.ticker_output.parent.mkdir(parents=True, exist_ok=True)
    args.date_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    by_ticker.to_csv(args.ticker_output, index=False)
    by_date.to_csv(args.date_output, index=False)

    print("\nMatching and evaluation diagnostics")
    print(f"  Predictions without matching feature key: {diagnostics['predictions_without_matching_feature_key']:,}")
    print(f"  Feature keys without prediction: {diagnostics['feature_keys_without_prediction']:,}")
    print(f"  Matched prediction/feature keys: {diagnostics['matched_keys']:,}")
    print(f"  Matched rows with invalid prediction: {diagnostics['matched_rows_with_invalid_prediction']:,}")
    print(f"  Matched rows with invalid actual target: {diagnostics['matched_rows_with_invalid_actual']:,}")
    print(f"  Evaluable matched predictions: {diagnostics['evaluable_matched_predictions']:,}")
    print(f"  Feature rows with valid targets: {diagnostics['feature_rows_with_valid_target']:,}")

    print("\nOverall metrics (returns are decimals, not percentages):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved summary: {args.summary_output}")
    print(f"Saved per-ticker metrics: {args.ticker_output}")
    print(f"Saved per-date metrics: {args.date_output}")
    print("\nLower MAE/RMSE is better. Positive baseline-relative improvement means Ridge is better.")
    print("Zero-return baseline directional accuracy is marked NaN because it makes no direction forecast.")


if __name__ == "__main__":
    main()
