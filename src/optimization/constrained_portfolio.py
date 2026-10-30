"""
Hard portfolio optimization with cardinality, turnover, and transaction costs.

The model is a mixed-integer convex quadratic optimization problem:

    maximize:
        mu @ w
        - 0.5 * risk_aversion * (w.T @ Sigma @ w)
        - transaction_costs @ abs(w - previous_weights)

    subject to:
        sum(w) == 1
        0 <= w[i] <= max_weight * z[i]
        sum(z) <= max_holdings
        0.5 * sum(abs(w - previous_weights)) <= max_turnover

where:
    w[i] = portfolio weight of asset i
    z[i] = binary variable indicating whether asset i is selected
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np

from src.optimization.constraints import (
    PortfolioConstraints,
    calculate_turnover,
    validate_portfolio_weights,
)
from src.optimization.mean_variance import _validate_inputs


@dataclass(frozen=True)
class ConstrainedPortfolioResult:
    """Results from the hard portfolio optimization problem."""

    tickers: tuple[str, ...]
    weights: np.ndarray
    selected: np.ndarray
    expected_return: float
    portfolio_variance: float
    portfolio_volatility: float
    turnover: float
    transaction_cost: float
    objective_value: float
    risk_aversion: float
    solver: str
    status: str

    def weights_frame(self):
        """Return weights and selection flags as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "ticker": list(self.tickers),
                "weight": self.weights.copy(),
                "selected": self.selected.copy(),
            }
        ).set_index("ticker")


def _validate_previous_weights(
    previous_weights: Sequence[float] | np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Validate the previous portfolio weights."""

    previous = np.asarray(previous_weights, dtype=float)

    if previous.ndim != 1 or previous.size != n_assets:
        raise ValueError(
            f"previous_weights must be a vector of length {n_assets}."
        )

    if not np.isfinite(previous).all():
        raise ValueError("previous_weights contains non-finite values.")

    if np.any(previous < -1e-8):
        raise ValueError("previous_weights must be non-negative.")

    if not np.isclose(previous.sum(), 1.0, atol=1e-6):
        raise ValueError(
            "previous_weights must sum to 1.0 for a fully invested portfolio."
        )

    # Remove tiny numerical negatives.
    previous = np.maximum(previous, 0.0)

    # Preserve a fully invested starting portfolio.
    previous /= previous.sum()

    return previous


def _validate_costs(
    transaction_costs: Sequence[float] | np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Validate per-asset transaction-cost rates."""

    costs = np.asarray(transaction_costs, dtype=float)

    if costs.ndim == 0:
        costs = np.full(n_assets, float(costs))

    if costs.ndim != 1 or costs.size != n_assets:
        raise ValueError(
            "transaction_costs must be a scalar or a vector "
            f"of length {n_assets}."
        )

    if not np.isfinite(costs).all():
        raise ValueError("transaction_costs contains non-finite values.")

    if np.any(costs < 0):
        raise ValueError("transaction_costs must be non-negative.")

    return costs


def optimize_constrained_portfolio(
    expected_returns: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    tickers: Sequence[str],
    previous_weights: Sequence[float] | np.ndarray,
    risk_aversion: float = 5.0,
    max_weight: float = 0.20,
    max_holdings: int = 10,
    max_turnover: float = 0.50,
    transaction_costs: float | Sequence[float] | np.ndarray = 0.001,
    solver: str | None = None,
    selection_tolerance: float = 1e-6,
) -> ConstrainedPortfolioResult:
    """
    Optimize a long-only portfolio with cardinality and trading constraints.

    Parameters
    ----------
    expected_returns:
        Predicted returns for the current decision date.
    covariance:
        Historical covariance matrix in the same ticker order.
    tickers:
        Ticker labels matching expected_returns and covariance.
    previous_weights:
        Previous portfolio weights in exactly the same ticker order.
        Must be fully invested.
    risk_aversion:
        Non-negative penalty on portfolio variance.
    max_weight:
        Maximum weight permitted for any one stock.
    max_holdings:
        Maximum number of selected stocks.
    max_turnover:
        Maximum one-way turnover, measured as
        0.5 * sum(abs(new_weights - previous_weights)).
    transaction_costs:
        Non-negative cost rates per unit of traded portfolio weight.
        A scalar applies the same cost rate to every asset.
    solver:
        Optional CVXPY solver name.
    selection_tolerance:
        Numerical tolerance for interpreting optimized weights as selected.

    Returns
    -------
    ConstrainedPortfolioResult
        Optimized portfolio and its trading statistics.

    Notes
    -----
    This is a mixed-integer quadratic optimization problem. An installed
    solver must support the required problem class. The function will raise
    a clear error if the chosen solver cannot solve it.
    """

    mu, sigma, ordered_tickers = _validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        tickers=tickers,
        risk_aversion=risk_aversion,
    )

    n_assets = mu.size

    if isinstance(max_weight, (bool, np.bool_)):
        raise ValueError("max_weight must be a number in (0, 1].")

    try:
        max_weight = float(max_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_weight must be a number in (0, 1].") from exc

    if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be a number in (0, 1].")

    if isinstance(max_holdings, (bool, np.bool_)):
        raise ValueError("max_holdings must be a positive integer.")

    if not isinstance(max_holdings, (int, np.integer)):
        raise ValueError("max_holdings must be a positive integer.")

    if not 1 <= max_holdings <= n_assets:
        raise ValueError(
            f"max_holdings must be between 1 and {n_assets}."
        )

    if max_holdings * max_weight < 1.0 - 1e-10:
        raise ValueError(
            "The requested cardinality and position cap are infeasible: "
            "max_holdings * max_weight must be at least 1.0."
        )

    if isinstance(max_turnover, (bool, np.bool_)):
        raise ValueError("max_turnover must be finite and non-negative.")

    try:
        max_turnover = float(max_turnover)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_turnover must be finite and non-negative."
        ) from exc

    if not np.isfinite(max_turnover) or max_turnover < 0.0:
        raise ValueError("max_turnover must be finite and non-negative.")

    if isinstance(selection_tolerance, (bool, np.bool_)):
        raise ValueError("selection_tolerance must be finite and positive.")

    try:
        selection_tolerance = float(selection_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selection_tolerance must be finite and positive."
        ) from exc

    if not np.isfinite(selection_tolerance) or selection_tolerance <= 0.0:
        raise ValueError("selection_tolerance must be finite and positive.")

    previous = _validate_previous_weights(previous_weights, n_assets)
    costs = _validate_costs(transaction_costs, n_assets)

    # Mixed-integer decision variables:
    # w = continuous portfolio weights
    # z = binary asset-selection variables
    weights = cp.Variable(n_assets, name="portfolio_weights")
    selected = cp.Variable(n_assets, boolean=True, name="selected_assets")

    trade_amounts = cp.abs(weights - previous)

    expected_return_expression = mu @ weights

    variance_expression = cp.quad_form(
        weights,
        cp.psd_wrap(sigma),
    )

    transaction_cost_expression = costs @ trade_amounts

    objective = cp.Maximize(
        expected_return_expression
        - 0.5 * risk_aversion * variance_expression
        - transaction_cost_expression
    )

    problem_constraints = [
        cp.sum(weights) == 1.0,
        weights >= 0.0,
        weights <= max_weight * selected,
        cp.sum(selected) <= max_holdings,
        0.5 * cp.sum(trade_amounts) <= max_turnover,
    ]

    problem = cp.Problem(objective, problem_constraints)

    try:
        if solver is None:
            problem.solve()
        else:
            problem.solve(solver=solver)
    except cp.error.SolverError as exc:
        installed = cp.installed_solvers()

        raise RuntimeError(
            "The hard portfolio problem could not be solved by the "
            "configured CVXPY solver. This formulation requires a solver "
            "that supports mixed-integer quadratic optimization with "
            "absolute-value trading terms. "
            f"Requested solver: {solver!r}. "
            f"Installed solvers: {installed}. "
            "Install and configure a compatible solver before running "
            "the cardinality-constrained optimizer."
        ) from exc

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            "Hard portfolio optimization did not produce a usable solution. "
            f"Solver status: {problem.status!r}."
        )

    if weights.value is None or selected.value is None:
        raise RuntimeError("The solver returned incomplete decision variables.")

    optimized_weights = np.asarray(weights.value, dtype=float).reshape(-1)
    selection_values = np.asarray(selected.value, dtype=float).reshape(-1)

    if not np.isfinite(optimized_weights).all():
        raise RuntimeError("The solver returned non-finite weights.")

    if not np.isfinite(selection_values).all():
        raise RuntimeError("The solver returned non-finite selection values.")

    # Binary variables should be close to either zero or one.
    rounded_selection = np.rint(selection_values)

    if not np.allclose(selection_values, rounded_selection, atol=1e-4):
        raise RuntimeError(
            "The solver returned selection variables that are not "
            "sufficiently close to binary values."
        )

    selected_mask = rounded_selection.astype(bool)

    # Validate the actual weights, including cardinality and turnover.
    if int(np.count_nonzero(optimized_weights > selection_tolerance)) > max_holdings:
        raise RuntimeError(
            "The solution exceeds max_holdings when evaluated using "
            "selection_tolerance."
        )

    if np.any(optimized_weights > max_weight + 1e-5):
        raise RuntimeError("The solution violates the maximum position weight.")

    if np.any(optimized_weights < -1e-5):
        raise RuntimeError("The solution violates the long-only constraint.")

    if not np.isclose(optimized_weights.sum(), 1.0, atol=1e-5):
        raise RuntimeError("The solution violates the fully invested constraint.")

    actual_turnover = calculate_turnover(optimized_weights, previous)

    if actual_turnover > max_turnover + 1e-5:
        raise RuntimeError("The solution violates the turnover limit.")

    actual_transaction_cost = float(
        costs @ np.abs(optimized_weights - previous)
    )

    expected_portfolio_return = float(mu @ optimized_weights)
    portfolio_variance = float(optimized_weights @ sigma @ optimized_weights)

    if portfolio_variance < -1e-10:
        raise RuntimeError("Computed portfolio variance is negative.")

    portfolio_variance = max(0.0, portfolio_variance)

    # Validate the core continuous constraints using the shared validator.
    validate_portfolio_weights(
        optimized_weights,
        constraints=PortfolioConstraints(
            fully_invested=True,
            long_only=True,
            max_weight=max_weight,
            max_turnover=max_turnover,
            max_holdings=max_holdings,
            budget_tolerance=1e-5,
            weight_tolerance=selection_tolerance,
        ),
        previous_weights=previous,
        tolerance=1e-5,
    )

    return ConstrainedPortfolioResult(
        tickers=ordered_tickers,
        weights=optimized_weights.copy(),
        selected=selected_mask.copy(),
        expected_return=expected_portfolio_return,
        portfolio_variance=portfolio_variance,
        portfolio_volatility=float(np.sqrt(portfolio_variance)),
        turnover=float(actual_turnover),
        transaction_cost=actual_transaction_cost,
        objective_value=float(problem.value),
        risk_aversion=float(risk_aversion),
        solver=str(problem.solver_stats.solver_name),
        status=str(problem.status),
    )