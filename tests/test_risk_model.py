import numpy as np
import pandas as pd
import pytest

from src.optimization.risk_model import estimate_covariance


def make_history(
    tickers=("AAA", "BBB", "CCC"),
    observations=300,
):
    """Generate deterministic synthetic daily returns."""
    dates = pd.bdate_range("2024-01-01", periods=observations)

    rows = []

    for ticker_index, ticker in enumerate(tickers):
        for day_index, date in enumerate(dates):
            # Deterministic, varying returns; no random seed required.
            ret = (
                0.001 * (ticker_index + 1)
                + 0.002 * np.sin((day_index + 1) * (ticker_index + 1) / 7)
                + 0.001 * np.cos((day_index + 2) / 5)
            )

            rows.append(
                {
                    "Symbol": ticker,
                    "Date": date.strftime("%Y-%m-%d"),
                    "daily_return": ret,
                }
            )

    return pd.DataFrame(rows)


def test_covariance_shape_and_ticker_order():
    history = make_history()

    result = estimate_covariance(
        history,
        decision_date="2025-02-28",
        tickers=["CCC", "AAA", "BBB"],
        window=100,
    )

    assert result.covariance.shape == (3, 3)
    assert result.tickers == ("CCC", "AAA", "BBB")
    assert list(result.as_frame().index) == ["CCC", "AAA", "BBB"]
    assert list(result.as_frame().columns) == ["CCC", "AAA", "BBB"]


def test_covariance_is_finite_and_symmetric():
    history = make_history()

    result = estimate_covariance(
        history,
        decision_date="2025-02-28",
        tickers=["AAA", "BBB", "CCC"],
        window=100,
    )

    assert np.isfinite(result.covariance).all()
    assert np.allclose(result.covariance, result.covariance.T)


def test_covariance_is_positive_semidefinite_within_tolerance():
    history = make_history()

    result = estimate_covariance(
        history,
        decision_date="2025-02-28",
        tickers=["AAA", "BBB", "CCC"],
        window=100,
    )

    eigenvalues = np.linalg.eigvalsh(result.covariance)

    assert eigenvalues.min() >= -1e-10


def test_only_uses_observations_before_decision_date():
    history = make_history(observations=300)

    decision_date = "2024-06-01"

    result = estimate_covariance(
        history,
        decision_date=decision_date,
        tickers=["AAA", "BBB", "CCC"],
        window=50,
    )

    assert result.decision_date == pd.Timestamp(decision_date)
    assert result.observations == 50


def test_missing_returns_are_not_zero_filled():
    history = make_history(observations=80)

    # Make one date incomplete across the requested universe.
    missing_date = history.loc[
        history["Symbol"].eq("AAA"), "Date"
    ].iloc[-10]

    mask = (
        history["Symbol"].eq("AAA")
        & history["Date"].eq(missing_date)
    )

    history.loc[mask, "daily_return"] = np.nan

    result = estimate_covariance(
        history,
        decision_date="2025-01-01",
        tickers=["AAA", "BBB", "CCC"],
        window=60,
        min_observations=30,
    )

    # The missing cross-sectional observation must be excluded.
    assert result.observations == 60


def test_raises_for_insufficient_complete_observations():
    history = make_history(observations=20)

    with pytest.raises(ValueError, match="Insufficient complete"):
        estimate_covariance(
            history,
            decision_date="2025-01-01",
            tickers=["AAA", "BBB", "CCC"],
            window=50,
            min_observations=30,
        )


def test_raises_for_duplicate_ticker_date():
    history = make_history(observations=40)

    history = pd.concat(
        [history, history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate ticker-date"):
        estimate_covariance(
            history,
            decision_date="2025-01-01",
            tickers=["AAA", "BBB", "CCC"],
            window=30,
        )


def test_raises_when_ticker_has_no_history():
    history = make_history(tickers=("AAA", "BBB"), observations=60)

    with pytest.raises(ValueError, match="No historical return records"):
        estimate_covariance(
            history,
            decision_date="2025-01-01",
            tickers=["AAA", "MISSING"],
            window=30,
        )


def test_rejects_duplicate_requested_tickers():
    history = make_history()

    with pytest.raises(ValueError, match="Duplicate tickers"):
        estimate_covariance(
            history,
            decision_date="2025-01-01",
            tickers=["AAA", "AAA"],
            window=30,
        )

# Run the test :
# python -m pytest -q tests/test_data_interface.py tests/test_risk_model.py