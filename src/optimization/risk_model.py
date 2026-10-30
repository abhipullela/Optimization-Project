"""
Historical covariance estimation for portfolio optimization.

This module consumes the existing cleaned historical dataset. It does not
retrain the ML model or generate new predictions.

Expected input columns:
    Symbol
    Date
    daily_return

The covariance matrix preserves the ticker order supplied by the caller.

Important:
    The current implementation uses the existing daily_return column.
    Validate its calculation and investigate corporate-action adjustments
    before relying on the covariance estimates for final backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovarianceEstimate:
    # Container for a historical covariance estimate.

    decision_date: pd.Timestamp
    tickers: tuple[str, ...]
    covariance: np.ndarray
    observations: int
    window: int

    def as_frame(self) -> pd.DataFrame:
        # Return the covariance matrix as a labeled DataFrame.
        return pd.DataFrame(
            self.covariance.copy(),
            index=list(self.tickers),
            columns=list(self.tickers),
        )


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    # Normalize ticker identifiers while preserving their order.
    if isinstance(tickers, (str, bytes)):
        raise TypeError("tickers must be a sequence of ticker identifiers.")

    normalized = []

    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Every ticker must be a non-empty string.")

        normalized.append(ticker.strip().upper())

    if not normalized:
        raise ValueError("At least one ticker must be supplied.")

    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate tickers are not allowed.")

    return tuple(normalized)


def _prepare_historical_returns(
    historical_data: pd.DataFrame,
    decision_date: str | pd.Timestamp,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """
    Validate and align historical returns.

    Returns a date-indexed wide DataFrame with one column per ticker.
    Dates after decision_date are excluded.
    """
    required_columns = {"Symbol", "Date", "daily_return"}
    missing_columns = required_columns.difference(historical_data.columns)

    if missing_columns:
        raise ValueError(
            "Historical data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if historical_data.empty:
        raise ValueError("Historical data is empty.")

    data = historical_data.loc[
        :, ["Symbol", "Date", "daily_return"]
    ].copy()

    # Normalize ticker identifiers.
    if data["Symbol"].isna().any():
        raise ValueError("Historical data contains missing ticker identifiers.")

    data["Symbol"] = data["Symbol"].astype(str).str.strip().str.upper()

    if data["Symbol"].eq("").any():
        raise ValueError("Historical data contains empty ticker identifiers.")

    # Parse dates and numeric returns.
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    if data["Date"].isna().any():
        raise ValueError("Historical data contains invalid dates.")

    data["daily_return"] = pd.to_numeric(
        data["daily_return"], errors="coerce"
    )

    # Normalize timestamps to calendar dates. The source data contains
    # daily observations, so intraday timestamps are not required here.
    data["Date"] = data["Date"].dt.normalize()

    requested_date = pd.Timestamp(decision_date)

    if pd.isna(requested_date):
        raise ValueError("decision_date must be a valid date.")

    if requested_date.tzinfo is not None:
        requested_date = requested_date.tz_localize(None)

    requested_date = requested_date.normalize()

    # Strict point-in-time filter: never use observations after the
    # portfolio decision date.
    data = data.loc[data["Date"] <= requested_date]

    # Restrict the dataset to the requested portfolio universe.
    data = data.loc[data["Symbol"].isin(tickers)]

    if data.empty:
        raise ValueError(
            "No historical observations are available for the requested "
            "tickers on or before the decision date."
        )

    # Multiple rows for a ticker and date make the daily return ambiguous.
    duplicate_mask = data.duplicated(
        subset=["Date", "Symbol"], keep=False
    )

    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, ["Date", "Symbol"]].head(5)

        raise ValueError(
            "Duplicate ticker-date observations found. Resolve these in the source data before estimating covariance. Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    # A missing return is not a zero return. Keep it as missing so that
    # the caller can apply an explicit alignment policy.
    finite_mask = np.isfinite(data["daily_return"].to_numpy(dtype=float))
    data.loc[~finite_mask, "daily_return"] = np.nan

    wide = data.pivot(
        index="Date",
        columns="Symbol",
        values="daily_return",
    ).sort_index()

    # Ensure that every requested ticker is represented.
    missing_tickers = [ticker for ticker in tickers if ticker not in wide.columns]

    if missing_tickers:
        raise ValueError(
            "No historical return records were found for these tickers: "
            f"{missing_tickers}"
        )

    # Explicitly preserve caller-specified order.
    wide = wide.reindex(columns=list(tickers))

    return wide


def estimate_covariance(
    historical_data: pd.DataFrame,
    decision_date: str | pd.Timestamp,
    tickers: Sequence[str],
    window: int = 252,
    min_observations: int | None = None,
) -> CovarianceEstimate:
    """
    Estimate a sample covariance matrix from historical daily returns.

    Parameters
    ----------
    historical_data:
        Cleaned historical data containing Symbol, Date, and daily_return.
    decision_date:
        Date at which the portfolio is being formed. No later observations
        are permitted.
    tickers:
        Tickers in the exact order required for the output matrix.
    window:
        Maximum number of aligned historical return observations to use.
        Defaults to 252.
    min_observations:
        Minimum number of complete observations required. Defaults to
        min(window, max(30, number_of_tickers + 2)).

    Returns
    -------
    CovarianceEstimate
        Contains the decision date, ordered tickers, covariance matrix,
        number of observations used, and configured window.

    Missing-data policy
    -------------------
    Only dates on which every requested ticker has a finite return are
    retained. The latest `window` complete observations are used.

    This avoids silently treating missing returns as zero. If there are
    insufficient complete observations, the function raises ValueError.
    """
    if not isinstance(historical_data, pd.DataFrame):
        raise TypeError("historical_data must be a pandas DataFrame.")

    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError("window must be an integer greater than or equal to 2.")

    ordered_tickers = _normalize_tickers(tickers)

    if min_observations is None:
        min_observations = min(
            window,
            max(30, len(ordered_tickers) + 2),
        )

    if (
        not isinstance(min_observations, int)
        or isinstance(min_observations, bool)
        or min_observations < 2
    ):
        raise ValueError(
            "min_observations must be an integer greater than or equal to 2."
        )

    if min_observations > window:
        raise ValueError("min_observations cannot exceed window.")

    wide = _prepare_historical_returns(
        historical_data=historical_data,
        decision_date=decision_date,
        tickers=ordered_tickers,
    )

    # Use only complete cross-sectional observations. This is a deliberate policy: no zero filling or pairwise covariance estimation.
    complete_returns = wide.dropna(axis=0, how="any")

    # Keep the latest observations available at the decision date.
    selected_returns = complete_returns.tail(window)

    if len(selected_returns) < min_observations:
        raise ValueError(
            "Insufficient complete historical observations for covariance "
            f"estimation: found {len(selected_returns)}, require at least "
            f"{min_observations}. Requested tickers: "
            f"{list(ordered_tickers)}."
        )

    values = selected_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite returns remain after historical-data preparation."
        )

    # Sample covariance uses ddof=1. NumPy returns a scalar for one ticker, so force the result to remain a two-dimensional matrix.
    covariance = np.atleast_2d(
        np.cov(values, rowvar=False, ddof=1)
    ).astype(float)

    expected_shape = (len(ordered_tickers), len(ordered_tickers))

    if covariance.shape != expected_shape:
        raise RuntimeError(
            "Unexpected covariance matrix shape: "
            f"{covariance.shape}; expected {expected_shape}."
        )

    if not np.isfinite(covariance).all():
        raise ValueError("Estimated covariance matrix contains non-finite values.")

    # Remove tiny floating-point asymmetries.
    covariance = (covariance + covariance.T) / 2.0

    # Reject materially negative eigenvalues. Tiny negative values can arise from floating-point rounding and are handled by the tolerance below.
    eigenvalues = np.linalg.eigvalsh(covariance)

    scale = max(1.0, float(np.max(np.abs(covariance))))
    tolerance = 1e-10 * scale

    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(
            "Estimated covariance matrix is not positive semidefinite within numerical tolerance."
        )

    # Record tiny negative eigenvalues as numerical diagnostics only. 
    # Instead of silently applying arbitrary shrinkage or altering the risk estimates.
    decision_timestamp = pd.Timestamp(decision_date)

    if decision_timestamp.tzinfo is not None:
        decision_timestamp = decision_timestamp.tz_localize(None)

    decision_timestamp = decision_timestamp.normalize()

    return CovarianceEstimate(
        decision_date=decision_timestamp,
        tickers=ordered_tickers,
        covariance=covariance,
        observations=len(selected_returns),
        window=window,
    )