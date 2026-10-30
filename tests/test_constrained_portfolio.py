import cvxpy as cp
import numpy as np
import pytest

from src.optimization.constrained_portfolio import (
    _validate_costs,
    _validate_previous_weights,
    optimize_constrained_portfolio,
)


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

    previous_weights = np.array([1 / 3, 1 / 3, 1 / 3])

    return expected_returns, covariance, tickers, previous_weights


def test_costs_accept_scalar_or_vector():
    assert np.allclose(_validate_costs(0.001, 3), [0.001, 0.001, 0.001])
    assert np.allclose(_validate_costs([0.001, 0.002, 0.003], 3), [0.001, 0.002, 0.003])


def test_costs_reject_negative_or_nonfinite_values():
    with pytest.raises(ValueError):
        _validate_costs(-0.001, 3)

    with pytest.raises(ValueError):
        _validate_costs([0.001, np.nan, 0.003], 3)


def test_previous_weights_must_be_fully_invested():
    with pytest.raises(ValueError):
        _validate_previous_weights([0.2, 0.2, 0.2], 3)


def test_previous_weights_must_have_correct_shape():
    with pytest.raises(ValueError):
        _validate_previous_weights([0.5, 0.5], 3)


def test_infeasible_cardinality_and_position_cap_rejected():
    mu, sigma, tickers, previous = example_inputs()

    with pytest.raises(ValueError):
        optimize_constrained_portfolio(
            mu,
            sigma,
            tickers,
            previous,
            max_weight=0.20,
            max_holdings=3,
        )


def test_invalid_max_turnover_rejected():
    mu, sigma, tickers, previous = example_inputs()

    with pytest.raises(ValueError):
        optimize_constrained_portfolio(
            mu,
            sigma,
            tickers,
            previous,
            max_turnover=-0.1,
        )


def test_installed_solvers_reported():
    # Diagnostic test: documents solver availability without assuming
    # that any installed solver supports mixed-integer quadratic problems.
    installed = cp.installed_solvers()
    assert isinstance(installed, list)
    assert len(installed) > 0

def test_scip_solves_cardinality_constrained_portfolio():
    mu = np.array([0.020, 0.015, 0.010, 0.005])

    sigma = np.array(
        [
            [0.00040, 0.00005, 0.00002, 0.00001],
            [0.00005, 0.00030, 0.00003, 0.00001],
            [0.00002, 0.00003, 0.00025, 0.00002],
            [0.00001, 0.00001, 0.00002, 0.00020],
        ]
    )

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    previous_weights = np.array([0.25, 0.25, 0.25, 0.25])

    result = optimize_constrained_portfolio(
        expected_returns=mu,
        covariance=sigma,
        tickers=tickers,
        previous_weights=previous_weights,
        risk_aversion=5.0,
        max_weight=0.50,
        max_holdings=2,
        max_turnover=0.50,
        transaction_costs=0.001,
        solver="SCIP",
    )

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.solver.upper() == "SCIP"

    assert np.isclose(result.weights.sum(), 1.0, atol=1e-5)
    assert np.all(result.weights >= -1e-5)
    assert np.all(result.weights <= 0.50 + 1e-5)

    assert np.count_nonzero(result.weights > 1e-5) <= 2
    assert result.selected.sum() <= 2

    assert result.turnover <= 0.50 + 1e-5
    assert result.transaction_cost >= 0.0
    assert np.isfinite(result.objective_value)


def test_selected_indicators_match_positive_weights():
    mu = np.array([0.020, 0.015, 0.010, 0.005])

    sigma = np.diag([0.00040, 0.00030, 0.00025, 0.00020])

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    previous_weights = np.array([0.25, 0.25, 0.25, 0.25])

    result = optimize_constrained_portfolio(
        expected_returns=mu,
        covariance=sigma,
        tickers=tickers,
        previous_weights=previous_weights,
        risk_aversion=5.0,
        max_weight=0.50,
        max_holdings=2,
        max_turnover=0.50,
        transaction_costs=0.001,
        solver="SCIP",
    )

    # Every material holding must be selected.
    material_holdings = result.weights > 1e-5
    assert np.all(result.selected[material_holdings])

    # The reported selection must respect the cardinality limit.
    assert result.selected.sum() <= 2


def test_transaction_cost_matches_weight_changes():
    mu = np.array([0.020, 0.015, 0.010, 0.005])
    sigma = np.diag([0.00040, 0.00030, 0.00025, 0.00020])

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    previous_weights = np.array([0.25, 0.25, 0.25, 0.25])
    costs = np.array([0.001, 0.002, 0.003, 0.004])

    result = optimize_constrained_portfolio(
        expected_returns=mu,
        covariance=sigma,
        tickers=tickers,
        previous_weights=previous_weights,
        risk_aversion=5.0,
        max_weight=0.50,
        max_holdings=2,
        max_turnover=0.50,
        transaction_costs=costs,
        solver="SCIP",
    )

    expected_cost = float(
        costs @ np.abs(result.weights - previous_weights)
    )

    assert np.isclose(
        result.transaction_cost,
        expected_cost,
        atol=1e-8,
    )


def test_reported_objective_matches_mathematical_formula():
    mu = np.array([0.020, 0.015, 0.010, 0.005])
    sigma = np.diag([0.00040, 0.00030, 0.00025, 0.00020])

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    previous_weights = np.array([0.25, 0.25, 0.25, 0.25])
    costs = np.array([0.001, 0.002, 0.003, 0.004])
    risk_aversion = 5.0

    result = optimize_constrained_portfolio(
        expected_returns=mu,
        covariance=sigma,
        tickers=tickers,
        previous_weights=previous_weights,
        risk_aversion=risk_aversion,
        max_weight=0.50,
        max_holdings=2,
        max_turnover=0.50,
        transaction_costs=costs,
        solver="SCIP",
    )

    w = result.weights

    independently_calculated_objective = (
        mu @ w
        - 0.5 * risk_aversion * (w @ sigma @ w)
        - costs @ np.abs(w - previous_weights)
    )

    assert np.isclose(
        result.objective_value,
        independently_calculated_objective,
        atol=1e-7,
    )


def test_turnover_matches_one_way_turnover_definition():
    mu = np.array([0.020, 0.015, 0.010, 0.005])
    sigma = np.diag([0.00040, 0.00030, 0.00025, 0.00020])

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    previous_weights = np.array([0.25, 0.25, 0.25, 0.25])

    result = optimize_constrained_portfolio(
        expected_returns=mu,
        covariance=sigma,
        tickers=tickers,
        previous_weights=previous_weights,
        risk_aversion=5.0,
        max_weight=0.50,
        max_holdings=2,
        max_turnover=0.50,
        transaction_costs=0.001,
        solver="SCIP",
    )

    independently_calculated_turnover = (
        0.5 * np.abs(result.weights - previous_weights).sum()
    )

    assert np.isclose(
        result.turnover,
        independently_calculated_turnover,
        atol=1e-8,
    )

# Run the test:
# python -m pytest -q tests/test_data_interface.py tests/test_risk_model.py tests/test_constraints.py tests/test_mean_variance.py tests/test_constrained_portfolio.py
# python -m pytest -q tests/test_constrained_portfolio.py