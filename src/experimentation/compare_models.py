"""Compare return-prediction models using chronological walk-forward validation.

Run from the repository root:
python src/models/compare_models.py --start-date 2025-01-01 --end-date 2025-06-30 --model ridge
Use --model all to run every candidate (the default).

The script uses the same date/ticker feature dataset and label-availability cutoff
as the Ridge tuning workflow. All preprocessing is fitted inside each training
fold. The requested validation period is for model selection, not final testing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ml_features_primary.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "model_comparison_predictions.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "model_comparison_summary.csv"
DEFAULT_BY_TICKER = PROJECT_ROOT / "data" / "processed" / "model_comparison_by_ticker.csv"
DEFAULT_BY_DATE = PROJECT_ROOT / "data" / "processed" / "model_comparison_by_date.csv"
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
        raise FileNotFoundError(
            f"Feature file not found: {path}\nRun build_features.py first."
        )
    df = pd.read_csv(path)
    required = ["date", "Symbol", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Feature file missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["Symbol"] = df["Symbol"].astype("string").str.strip().str.upper()
    df = df.dropna(subset=["date", "Symbol"])
    df = df[df["Symbol"] != ""].copy()

    if df.duplicated(["date", "Symbol"]).any():
        examples = df.loc[
            df.duplicated(["date", "Symbol"], keep=False), ["date", "Symbol"]
        ].head(10)
        raise ValueError(
            "Duplicate (date, Symbol) rows found in feature dataset:\n"
            f"{examples.to_string(index=False)}"
        )

    for col in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["Symbol", "date"]).reset_index(drop=True)

    # The target is the next observed return for the same ticker, so it is
    # only available once that ticker's next observation date has occurred.
    df["target_end_date"] = df.groupby("Symbol", sort=False)["date"].shift(-1)
    return df


def build_models(random_state: int) -> dict:
    """Return fixed, documented candidate configurations."""
    return {
        "zero_return_baseline": None,
        "ridge_alpha_3000": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=3000.0)),
        ]),
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "polynomial_ridge_degree2_alpha1000": Pipeline([
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1000.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=2,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def calculate_metrics(group: pd.DataFrame) -> dict:
    actual = group["actual_return"].to_numpy(dtype=float)
    predicted = group["predicted_return"].to_numpy(dtype=float)
    errors = predicted - actual
    nonzero = actual != 0
    directional_accuracy = (
        float((np.sign(actual[nonzero]) == np.sign(predicted[nonzero])).mean())
        if nonzero.any() else np.nan
    )
    baseline_mae = float(np.mean(np.abs(actual)))
    baseline_rmse = float(np.sqrt(np.mean(actual ** 2)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {
        "n_predictions": int(len(actual)),
        "n_dates": int(group["date"].nunique()),
        "n_tickers": int(group["ticker"].nunique()),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "mean_actual_return": float(np.mean(actual)),
        "mean_predicted_return": float(np.mean(predicted)),
        "prediction_bias": float(np.mean(errors)),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "mae_improvement_pct": (
            (baseline_mae - mae) / baseline_mae * 100
            if baseline_mae > 0 else np.nan
        ),
        "rmse_improvement_pct": (
            (baseline_rmse - rmse) / baseline_rmse * 100
            if baseline_rmse > 0 else np.nan
        ),
    }


def compare_models(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_train_rows: int,
    random_state: int,
    selected_model: str = "all",
) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS]
    features_valid = X.notna().all(axis=1)
    labels_valid = (
        df[TARGET_COLUMN].notna()
        & df["target_end_date"].notna()
        & np.isfinite(df[TARGET_COLUMN])
    )
    validation_dates = sorted(
        d for d in df.loc[features_valid, "date"].unique()
        if start_date <= pd.Timestamp(d).normalize() <= end_date
    )
    if not validation_dates:
        raise ValueError(
            "No feature-valid dates found in the requested validation period."
        )

    all_models = build_models(random_state)
    model_aliases = {
        "baseline": "zero_return_baseline",
        "ridge": "ridge_alpha_3000",
        "linear": "linear_regression",
        "polynomial": "polynomial_ridge_degree2_alpha1000",
        "random_forest": "random_forest",
        "hist_gradient_boosting": "hist_gradient_boosting",
    }
    if selected_model == "all":
        models_to_run = all_models
    else:
        canonical_name = model_aliases.get(selected_model, selected_model)
        if canonical_name not in all_models:
            raise ValueError(f"Unknown model selection: {selected_model}")
        models_to_run = {canonical_name: all_models[canonical_name]}
    model_names = list(models_to_run.keys())
    prediction_records = []
    skipped_dates = 0
    fitted_dates = 0

    print(f"Validation dates in requested range: {len(validation_dates):,}")
    print(f"Candidate models: {len(model_names)}")
    print(f"Minimum training rows: {min_train_rows:,}")
    print("Running chronological walk-forward evaluation...")

    for date_index, raw_date in enumerate(validation_dates, start=1):
        decision_date = pd.Timestamp(raw_date).normalize()
        # Do not train on a label until the next observation that defines it
        # has occurred on or before this decision date.
        train_mask = (
            features_valid
            & labels_valid
            & (df["date"] < decision_date)
            & (df["target_end_date"] <= decision_date)
        )
        predict_mask = features_valid & (df["date"] == decision_date)
        train = df.loc[train_mask]
        predict = df.loc[predict_mask]

        if len(train) < min_train_rows or predict.empty:
            skipped_dates += 1
            continue

        fitted_dates += 1
        # Only score rows with a realized target. Rows without a next observed
        # return can still be predicted, but cannot be used for metrics.
        score_mask = predict[TARGET_COLUMN].notna() & np.isfinite(predict[TARGET_COLUMN])
        score_rows = predict.loc[score_mask]
        if score_rows.empty:
            continue

        # Re-create each estimator per date so fitted state never carries over.
        # Fit only the selected model when --model is specified.
        for model_name, model in models_to_run.items():
            try:
                if model is None:
                    pred_values = np.zeros(len(score_rows), dtype=float)
                else:
                    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
                    pred_values = model.predict(score_rows[FEATURE_COLUMNS]).astype(float)
            except Exception as exc:
                raise RuntimeError(
                    f"Model {model_name!r} failed on decision date "
                    f"{decision_date.date()}: {exc}"
                ) from exc

            for row, pred_value in zip(score_rows.itertuples(index=False), pred_values):
                prediction_records.append({
                    "date": decision_date.date().isoformat(),
                    "ticker": str(row.Symbol),
                    "model": model_name,
                    "predicted_return": float(pred_value),
                    "actual_return": float(getattr(row, TARGET_COLUMN)),
                })

        if date_index == 1 or date_index % 20 == 0 or date_index == len(validation_dates):
            print(
                f"  Processed {date_index:,}/{len(validation_dates):,} dates "
                f"({decision_date.date()})"
            )

    if not prediction_records:
        raise ValueError(
            "No evaluable predictions were produced. Check the date range, "
            "feature completeness, and --min-train-rows."
        )

    predictions = pd.DataFrame(prediction_records)
    predictions.attrs["skipped_dates"] = skipped_dates
    predictions.attrs["fitted_dates"] = fitted_dates
    return predictions


def make_reports(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    ticker_rows = []
    date_rows = []

    for model_name, group in predictions.groupby("model", sort=False):
        summary_rows.append({"model": model_name, **calculate_metrics(group)})
        for ticker, ticker_group in group.groupby("ticker", sort=True):
            ticker_rows.append({
                "model": model_name,
                "ticker": ticker,
                **calculate_metrics(ticker_group),
            })
        for date, date_group in group.groupby("date", sort=True):
            date_rows.append({
                "model": model_name,
                "date": date,
                **calculate_metrics(date_group),
            })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["mae", "rmse"], ascending=[True, True]
    ).reset_index(drop=True)
    by_ticker = pd.DataFrame(ticker_rows).sort_values(
        ["model", "ticker"]
    ).reset_index(drop=True)
    by_date = pd.DataFrame(date_rows).sort_values(
        ["date", "model"]
    ).reset_index(drop=True)
    return summary, by_ticker, by_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", default="all", choices=[
        "all", "baseline", "ridge", "linear", "polynomial",
        "random_forest", "hist_gradient_boosting",
    ], help="Choose one model to run, or 'all' to run every candidate.")
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--ticker-output", type=Path, default=None)
    parser.add_argument("--date-output", type=Path, default=None)
    parser.add_argument("--start-date", required=True, help="Validation start date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Validation end date, YYYY-MM-DD")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        start_date = pd.Timestamp(args.start_date).normalize()
        end_date = pd.Timestamp(args.end_date).normalize()
    except (TypeError, ValueError) as exc:
        parser.error(f"Invalid date: {exc}")
    if start_date > end_date:
        parser.error("--start-date must be on or before --end-date")
    if args.min_train_rows < 1:
        parser.error("--min-train-rows must be at least 1")

    # Give single-model runs their own filenames by default so running a
    # second model does not overwrite the first model's results.
    suffix = "" if args.model == "all" else f"_{args.model}"
    predictions_output = args.predictions_output or DEFAULT_PREDICTIONS.with_name(
        f"model_comparison_predictions{suffix}.csv"
    )
    summary_output = args.summary_output or DEFAULT_SUMMARY.with_name(
        f"model_comparison_summary{suffix}.csv"
    )
    ticker_output = args.ticker_output or DEFAULT_BY_TICKER.with_name(
        f"model_comparison_by_ticker{suffix}.csv"
    )
    date_output = args.date_output or DEFAULT_BY_DATE.with_name(
        f"model_comparison_by_date{suffix}.csv"
    )

    df = load_features(args.input)
    print(f"Loaded {len(df):,} feature rows across {df['Symbol'].nunique():,} tickers.")
    print(f"Selected model: {args.model}")
    predictions = compare_models(
        df, start_date, end_date, args.min_train_rows, args.random_state,
        selected_model=args.model,
    )
    summary, by_ticker, by_date = make_reports(predictions)

    for output_path in [
        predictions_output, summary_output, ticker_output, date_output,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_output, index=False)
    summary.to_csv(summary_output, index=False)
    by_ticker.to_csv(ticker_output, index=False)
    by_date.to_csv(date_output, index=False)

    print("\nEvaluation diagnostics")
    print(f"  Dates with at least one fitted model: {predictions['date'].nunique():,}")
    print(f"  Prediction rows (all models): {len(predictions):,}")
    print(f"  Skipped dates (insufficient training data or no prediction rows): {predictions.attrs.get('skipped_dates', 0):,}")
    print("\nOverall comparison (returns are decimals, not percentages):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved files:")
    print(f"  Predictions: {predictions_output}")
    print(f"  Summary:     {summary_output}")
    print(f"  By ticker:   {ticker_output}")
    print(f"  By date:     {date_output}")
    print("\nLower MAE/RMSE is better. Positive baseline-relative improvement means")
    print("a model beats the zero-return baseline on MAE/RMSE.")
    print("Use this validation comparison for selection; reserve a later period for final testing.")


if __name__ == "__main__":
    main()
