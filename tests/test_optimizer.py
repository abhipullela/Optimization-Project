import numpy as np
import pandas as pd
import pytest

from src.optimization.optimizer import optimize_portfolio


@pytest.fixture
def sample_inputs():
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.bdate_range("2025-01-01", periods=80)

    rows = []
    for i, ticker in enumerate(tickers):
        for j, date in enumerate(dates):
            daily_return = (
                0.001 * (i + 1)
                + 0.004 * np.sin((j + 1) * (i + 1) / 5)
                + 0.002 * np.cos((j + 1) / 7)
            )
            rows.append({
                "Symbol": ticker,
                "Date": date,
                "daily_return": daily_return,
            })

    historical = pd.DataFrame(rows)
    decision_date = dates[-1]

    # Deliberately shuffled to test ticker alignment.
    predictions = pd.DataFrame({
        "date": [decision_date] * 4,
        "ticker": ["CCC", "AAA", "DDD", "BBB"],
        "predicted_return": [0.012, 0.020, 0.004, 0.015],
        "predicted_horizon": ["1_trading_day"] * 4,
        "return_type": ["simple_return"] * 4,
        "model_version": ["test_model"] * 4,
    })

    return predictions, historical, decision_date, tickers


def test_easy_optimizer_returns_valid_portfolio(sample_inputs):
    predictions, historical, decision_date, tickers = sample_inputs

    result = optimize_portfolio(
        predictions=predictions,
        historical_data=historical,
        decision_date=decision_date,
        mode="easy",
        eligible_tickers=tickers,
        covariance_window=60,
        min_covariance_observations=30,
        max_weight=0.50,
        solver="CLARABEL",
    )

    assert result.mode == "easy"
    assert result.tickers == tuple(tickers)
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)
    assert np.all(result.weights >= -1e-8)
    assert np.all(result.weights <= 0.50 + 1e-6)
    assert result.covariance.shape == (4, 4)
    assert result.covariance_observations == 60
    assert np.isfinite(result.expected_portfolio_return)
    assert np.isfinite(result.portfolio_volatility)
    assert result.turnover is not None
    assert result.transaction_cost is None


def test_hard_optimizer_returns_cardinality_constrained_portfolio(sample_inputs):
    predictions, historical, decision_date, tickers = sample_inputs

    result = optimize_portfolio(
        predictions=predictions,
        historical_data=historical,
        decision_date=decision_date,
        mode="hard",
        eligible_tickers=tickers,
        covariance_window=60,
        min_covariance_observations=30,
        max_weight=0.50,
        max_holdings=2,
        max_turnover=0.50,
        transaction_costs=0.001,
        solver="SCIP",
    )

    assert result.mode == "hard"
    assert result.tickers == tuple(tickers)
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)
    assert np.all(result.weights >= -1e-8)
    assert np.all(result.weights <= 0.50 + 1e-6)
    assert np.count_nonzero(result.weights > 1e-5) <= 2
    assert result.turnover is not None
    assert result.turnover <= 0.50 + 1e-6
    assert result.transaction_cost is not None
    assert result.transaction_cost >= 0.0


def test_optimizer_rejects_unknown_mode(sample_inputs):
    predictions, historical, decision_date, _ = sample_inputs

    with pytest.raises(ValueError, match="mode"):
        optimize_portfolio(
            predictions=predictions,
            historical_data=historical,
            decision_date=decision_date,
            mode="unknown",
        )

# Run the test :
# python -m pytest -q tests/test_optimizer.py