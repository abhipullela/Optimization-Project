"""
Continuous mean-variance portfolio optimization.

Maximizes:
    expected_return @ weights
    - 0.5 * risk_aversion * portfolio_variance

Subject to:
    - Fully invested portfolio
    - Long-only positions
    - Optional maximum position weight

This is the Easy Problem. Cardinality and transaction-cost-aware
optimization will be handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np

from src.optimization.constraints import (
    PortfolioConstraints,
    validate_portfolio_weights,
)


@dataclass(frozen=True)
class MeanVarianceResult:
    """Results returned by the continuous optimizer."""

    tickers: tuple[str, ...]
    weights: np.ndarray
    expected_return: float
    portfolio_variance: float
    portfolio_volatility: float
    objective_value: float
    risk_aversion: float
    solver: str
    status: str

    def weights_frame(self):
        """Return portfolio weights as a ticker-indexed DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "ticker": list(self.tickers),
                "weight": self.weights.copy(),
            }
        ).set_index("ticker")


def _validate_inputs(
    expected_returns: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    tickers: Sequence[str],
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate and normalize optimizer inputs."""

    if isinstance(expected_returns, (str, bytes)):
        raise TypeError("expected_returns must be a numeric vector.")

    mu = np.asarray(expected_returns, dtype=float)

    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("expected_returns must be a non-empty 1D vector.")

    if not np.isfinite(mu).all():
        raise ValueError("expected_returns contains non-finite values.")

    if isinstance(tickers, (str, bytes)):
        raise TypeError("tickers must be a sequence of ticker strings.")

    normalized_tickers = []

    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Every ticker must be a non-empty string.")

        normalized_tickers.append(ticker.strip().upper())

    if len(normalized_tickers) != mu.size:
        raise ValueError(
            "The number of tickers must match the expected-return vector."
        )

    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError("Duplicate tickers are not allowed.")

    if isinstance(risk_aversion, (bool, np.bool_)):
        raise ValueError("risk_aversion must be a finite number >= 0.")

    try:
        risk_aversion = float(risk_aversion)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk_aversion must be a finite number >= 0.") from exc

    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("risk_aversion must be a finite number >= 0.")

    sigma = np.asarray(covariance, dtype=float)

    expected_shape = (mu.size, mu.size)

    if sigma.shape != expected_shape:
        raise ValueError(
            f"covariance must have shape {expected_shape}; got {sigma.shape}."
        )

    if not np.isfinite(sigma).all():
        raise ValueError("covariance contains non-finite values.")

    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=1e-8):
        raise ValueError("covariance must be symmetric.")

    # Numerical tolerance for small floating-point eigenvalue errors.
    sigma = (sigma + sigma.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(sigma)

    scale = max(1.0, float(np.max(np.abs(sigma))))
    psd_tolerance = 1e-10 * scale

    if float(eigenvalues.min()) < -psd_tolerance:
        raise ValueError(
            "covariance must be positive semidefinite. "
            "Check the historical risk-model output."
        )

    # Remove tiny negative eigenvalues caused only by floating-point noise.
    if float(eigenvalues.min()) < 0:
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sigma = (eigenvectors * eigenvalues) @ eigenvectors.T
        sigma = (sigma + sigma.T) / 2.0

    return mu, sigma, tuple(normalized_tickers)


def optimize_mean_variance(
    expected_returns: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    tickers: Sequence[str],
    risk_aversion: float = 5.0,
    constraints: PortfolioConstraints | None = None,
    solver: str | None = None,
) -> MeanVarianceResult:
    """
    Solve the continuous long-only mean-variance portfolio problem.

    Parameters
    ----------
    expected_returns:
        Predicted returns for the decision date, in ticker order.
    covariance:
        Historical covariance matrix in exactly the same ticker order.
    tickers:
        Ticker labels corresponding to expected_returns and covariance.
    risk_aversion:
        Non-negative penalty on portfolio variance. Must be calibrated
        consistently with the return and covariance units.
    constraints:
        Shared portfolio constraints. Only fully-invested, long-only,
        and maximum-weight settings are used by this continuous problem.
        Turnover and cardinality are reserved for the hard problem.
    solver:
        Optional CVXPY solver name. If omitted, CVXPY selects an installed
        compatible solver.

    Returns
    -------
    MeanVarianceResult
        Optimized weights and associated portfolio statistics.
    """

    mu, sigma, ordered_tickers = _validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        tickers=tickers,
        risk_aversion=risk_aversion,
    )

    if constraints is None:
        constraints = PortfolioConstraints()

    if not isinstance(constraints, PortfolioConstraints):
        raise TypeError("constraints must be a PortfolioConstraints instance.")

    n_assets = mu.size
    weights = cp.Variable(n_assets, name="portfolio_weights")

    portfolio_variance_expression = cp.quad_form(
        weights,
        cp.psd_wrap(sigma),
    )

    objective = cp.Maximize(
        mu @ weights
        - 0.5 * risk_aversion * portfolio_variance_expression
    )

    problem_constraints = []

    if constraints.fully_invested:
        problem_constraints.append(cp.sum(weights) == 1.0)

    if constraints.long_only:
        problem_constraints.append(weights >= 0.0)

    if constraints.max_weight is not None:
        problem_constraints.append(weights <= constraints.max_weight)

    problem = cp.Problem(objective, problem_constraints)

    try:
        if solver is None:
            problem.solve()
        else:
            problem.solve(solver=solver)
    except cp.error.SolverError as exc:
        installed = cp.installed_solvers()

        raise RuntimeError(
            "The mean-variance problem could not be solved. "
            f"Requested solver: {solver!r}. "
            f"Installed CVXPY solvers: {installed}. "
            "Check that a compatible solver is installed."
        ) from exc

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            "Mean-variance optimization did not produce a usable solution. "
            f"Solver status: {problem.status!r}. "
            "Check the constraints and solver configuration."
        )

    if weights.value is None:
        raise RuntimeError("The solver returned no portfolio weights.")

    optimized_weights = np.asarray(weights.value, dtype=float).reshape(-1)

    if not np.isfinite(optimized_weights).all():
        raise RuntimeError("The solver returned non-finite portfolio weights.")

    # Validate the solution against the constraints that this problem supports.
    validation_constraints = PortfolioConstraints(
        fully_invested=constraints.fully_invested,
        long_only=constraints.long_only,
        max_weight=constraints.max_weight,
        max_turnover=None,
        max_holdings=None,
        budget_tolerance=max(constraints.budget_tolerance, 1e-5),
        weight_tolerance=max(constraints.weight_tolerance, 1e-5),
    )

    validate_portfolio_weights(
        optimized_weights,
        constraints=validation_constraints,
        tolerance=max(constraints.budget_tolerance, 1e-5),
    )

    expected_portfolio_return = float(mu @ optimized_weights)
    portfolio_variance = float(optimized_weights @ sigma @ optimized_weights)

    # Protect against tiny negative numerical variance.
    if portfolio_variance < -1e-10:
        raise RuntimeError(
            f"Computed portfolio variance is unexpectedly negative: "
            f"{portfolio_variance}."
        )

    portfolio_variance = max(0.0, portfolio_variance)
    portfolio_volatility = float(np.sqrt(portfolio_variance))

    return MeanVarianceResult(
        tickers=ordered_tickers,
        weights=optimized_weights.copy(),
        expected_return=expected_portfolio_return,
        portfolio_variance=portfolio_variance,
        portfolio_volatility=portfolio_volatility,
        objective_value=float(problem.value),
        risk_aversion=float(risk_aversion),
        solver=str(problem.solver_stats.solver_name),
        status=str(problem.status),
    )