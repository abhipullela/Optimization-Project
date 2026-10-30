import numpy as np
import pytest

from src.optimization.constraints import PortfolioConstraints
from src.optimization.mean_variance import optimize_mean_variance


def example_inputs():
    tickers = ["AAA", "BBB", "CCC"]

    expected_returns = np.array([0.02, 0.015, 0.005])

    covariance = np.array(
        [
            [0.0004, 0.00005, 0.00002],
            [0.00005, 0.0003, 0.00001],
            [0.00002, 0.00001, 0.0002],
        ]
    )

    return expected_returns, covariance, tickers


def test_returns_valid_fully_invested_portfolio():
    mu, sigma, tickers = example_inputs()

    result = optimize_mean_variance(
        expected_returns=mu,
        covariance=sigma,
        tickers=tickers,
        risk_aversion=5.0,
    )

    assert result.weights.shape == (3,)
    assert np.isfinite(result.weights).all()
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-5)
    assert np.all(result.weights >= -1e-5)
    assert np.isfinite(result.expected_return)
    assert result.portfolio_variance >= 0
    assert result.portfolio_volatility >= 0
    assert result.status in {"optimal", "optimal_inaccurate"}


def test_respects_maximum_position_weight():
    mu, sigma, tickers = example_inputs()

    constraints = PortfolioConstraints(
        fully_invested=True,
        long_only=True,
        max_weight=0.50,
    )

    result = optimize_mean_variance(
        mu,
        sigma,
        tickers,
        risk_aversion=5.0,
        constraints=constraints,
    )

    assert np.isclose(result.weights.sum(), 1.0, atol=1e-5)
    assert np.all(result.weights <= 0.50 + 1e-5)


def test_higher_risk_aversion_does_not_increase_objective_value():
    mu, sigma, tickers = example_inputs()

    low_risk_aversion = optimize_mean_variance(
        mu, sigma, tickers, risk_aversion=1.0
    )

    high_risk_aversion = optimize_mean_variance(
        mu, sigma, tickers, risk_aversion=10.0
    )

    # Compare each solution under the same objective function.
    low_solution_under_high_penalty = (
        mu @ low_risk_aversion.weights
        - 0.5
        * 10.0
        * (low_risk_aversion.weights @ sigma @ low_risk_aversion.weights)
    )

    high_solution_objective = (
        mu @ high_risk_aversion.weights
        - 0.5
        * 10.0
        * (
            high_risk_aversion.weights
            @ sigma
            @ high_risk_aversion.weights
        )
    )

    assert high_solution_objective >= low_solution_under_high_penalty - 1e-6


@pytest.mark.parametrize(
    "mu, sigma, tickers, risk_aversion",
    [
        ([0.01, 0.02], np.eye(3), ["AAA", "BBB"], 5.0),
        ([0.01, np.nan], np.eye(2), ["AAA", "BBB"], 5.0),
        ([0.01, 0.02], np.array([[1.0, 2.0], [2.0, 1.0]]), ["AAA", "BBB"], 5.0),
        ([0.01, 0.02], np.eye(2), ["AAA", "AAA"], 5.0),
        ([0.01, 0.02], np.eye(2), ["AAA", "BBB"], -1.0),
    ],
)
def test_rejects_invalid_inputs(mu, sigma, tickers, risk_aversion):
    with pytest.raises((ValueError, TypeError)):
        optimize_mean_variance(
            expected_returns=mu,
            covariance=sigma,
            tickers=tickers,
            risk_aversion=risk_aversion,
        )


def test_infeasible_position_cap_is_reported():
    mu, sigma, tickers = example_inputs()

    constraints = PortfolioConstraints(
        fully_invested=True,
        long_only=True,
        max_weight=0.20,
    )

    with pytest.raises(RuntimeError):
        optimize_mean_variance(
            mu,
            sigma,
            tickers,
            constraints=constraints,
        )