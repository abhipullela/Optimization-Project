"""Validated interface between ML predictions and portfolio optimization.

This module does not optimize weights. It converts the existing prediction
CSV into a date-specific, ticker-aligned expected-return vector and fails
explicitly when the requested inputs are incomplete or inconsistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED_PREDICTION_COLUMNS = {
    "date",
    "ticker",
    "predicted_return",
    "predicted_horizon",
    "return_type",
    "model_version",
}
SUPPORTED_HORIZON = "1_trading_day"
SUPPORTED_RETURN_TYPE = "simple_return"


@dataclass(frozen=True)
class PredictionSlice:
    """Predictions for one decision date in a stable ticker order."""

    decision_date: pd.Timestamp
    tickers: tuple[str, ...]
    expected_returns: np.ndarray
    model_versions: tuple[str, ...]

    def as_series(self) -> pd.Series:
        """Return expected returns indexed by ticker for readable diagnostics."""
        return pd.Series(
            self.expected_returns.copy(),
            index=pd.Index(self.tickers, name="ticker"),
            name="predicted_return",
        )


def _normalize_tickers(values: Sequence[object]) -> list[str]:
    tickers = [str(value).strip().upper() for value in values]
    if any(not ticker or ticker == "NAN" for ticker in tickers):
        raise ValueError("Ticker identifiers must be present and non-empty.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Ticker list contains duplicates after normalization.")
    return tickers


def prepare_prediction_slice(
    predictions: pd.DataFrame,
    decision_date: str | pd.Timestamp,
    eligible_tickers: Sequence[str] | None = None,
) -> PredictionSlice:
    """Validate predictions and return a ticker-aligned vector for one date.

    Parameters
    ----------
    predictions:
        Existing ML prediction table with the six documented columns.
    decision_date:
        Date on which the portfolio decision is made.
    eligible_tickers:
        Optional ordered candidate universe. If supplied, every requested
        ticker must have exactly one prediction on the decision date. The
        output follows this exact order, not the CSV's row order.

    Raises
    ------
    ValueError
        For missing columns, duplicate date/ticker rows, invalid values,
        unsupported return conventions, or missing predictions.

    Notes
    -----
    This function intentionally does not fill missing predictions with zero.
    It also does not select stocks based on their predicted return values.
    """
    missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Prediction table is missing columns: {missing_columns}")

    frame = predictions[list(REQUIRED_PREDICTION_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Prediction table contains invalid or missing dates.")

    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    if frame["ticker"].isna().any() or frame["ticker"].eq("").any():
        raise ValueError("Prediction table contains missing or empty tickers.")

    frame["predicted_return"] = pd.to_numeric(frame["predicted_return"], errors="coerce")
    values = frame["predicted_return"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predicted returns must all be finite numeric values.")

    if frame["predicted_horizon"].isna().any() or not frame["predicted_horizon"].astype(str).eq(SUPPORTED_HORIZON).all():
        raise ValueError(f"Only prediction horizon '{SUPPORTED_HORIZON}' is supported.")
    if frame["return_type"].isna().any() or not frame["return_type"].astype(str).eq(SUPPORTED_RETURN_TYPE).all():
        raise ValueError(f"Only return type '{SUPPORTED_RETURN_TYPE}' is supported.")
    if frame["model_version"].isna().any() or frame["model_version"].astype(str).str.strip().eq("").any():
        raise ValueError("Model version must be present for every prediction.")

    duplicate_mask = frame.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, ["date", "ticker"]].head(5)
        raise ValueError(
            "Duplicate date/ticker predictions found; refusing ambiguous inputs.\n"
            f"{examples.to_string(index=False)}"
        )

    try:
        target_date = pd.Timestamp(decision_date).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid decision date: {decision_date!r}") from exc
    if pd.isna(target_date):
        raise ValueError(f"Invalid decision date: {decision_date!r}")

    dated = frame.loc[frame["date"].eq(target_date)].copy()
    if dated.empty:
        raise ValueError(f"No ML predictions are available for {target_date.date()}.")

    if eligible_tickers is None:
        ordered_tickers = sorted(dated["ticker"].astype(str).tolist())
    else:
        ordered_tickers = _normalize_tickers(eligible_tickers)
        available = set(dated["ticker"].astype(str))
        missing_tickers = [ticker for ticker in ordered_tickers if ticker not in available]
        if missing_tickers:
            raise ValueError(
                f"Missing predictions for eligible tickers on {target_date.date()}: "
                f"{missing_tickers}"
            )
        # Extra predictions outside the requested universe are deliberately ignored.
        dated = dated.loc[dated["ticker"].isin(ordered_tickers)]

    indexed = dated.set_index("ticker")

    if not indexed.index.is_unique:
        raise ValueError("Duplicate ticker records found for the selected date.")
    indexed = indexed.reindex(ordered_tickers)
    if indexed["predicted_return"].isna().any():
        # Defensive guard: missing values must never silently reach the optimizer.
        raise ValueError("Prediction alignment produced missing expected returns.")

    return PredictionSlice(
        decision_date=target_date,
        tickers=tuple(ordered_tickers),
        expected_returns=indexed["predicted_return"].to_numpy(dtype=float),
        model_versions=tuple(indexed["model_version"].astype(str).tolist()),
    )
