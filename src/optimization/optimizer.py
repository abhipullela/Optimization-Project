"""Orchestration layer for portfolio optimization.

Connects ML predictions, historical risk estimates, and the Easy and Hard
portfolio optimizers. This module does not retrain the ML model.

Timing assumption: the portfolio is formed after the decision date's close,
so returns dated on or before that date may be used for covariance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.optimization.constraints import (
    PortfolioConstraints,
    validate_portfolio_weights,
)
from src.optimization.constrained_portfolio import (
    optimize_constrained_portfolio,
)
from src.optimization.data_interface import prepare_prediction_slice
from src.optimization.mean_variance import optimize_mean_variance
from src.optimization.risk_model import estimate_covariance

OptimizationMode = Literal["easy", "hard"]


@dataclass(frozen=True)
class PortfolioOptimizationRun:
    """Combined inputs, estimates, and results for one decision date."""

    mode: OptimizationMode
    decision_date: pd.Timestamp
    tickers: tuple[str, ...]
    expected_returns: np.ndarray
    covariance: np.ndarray
    covariance_observations: int
    prediction_model_versions: tuple[str, ...]
    weights: np.ndarray
    expected_portfolio_return: float
    portfolio_variance: float
    portfolio_volatility: float
    objective_value: float
    turnover: float | None
    transaction_cost: float | None
    solver: str
    status: str

    def weights_frame(self) -> pd.DataFrame:
        """Return ticker weights and expected returns, sorted by weight."""
        frame = pd.DataFrame({
            "ticker": self.tickers,
            "expected_return": self.expected_returns,
            "weight": self.weights,
        })
        return frame.sort_values("weight", ascending=False, ignore_index=True)


def optimize_portfolio(
    predictions: pd.DataFrame,
    historical_data: pd.DataFrame,
    decision_date: str | pd.Timestamp,
    mode: OptimizationMode = "hard",
    eligible_tickers: Sequence[str] | None = None,
    covariance_window: int = 252,
    min_covariance_observations: int | None = None,
    risk_aversion: float = 5.0,
    max_weight: float = 0.20,
    max_holdings: int = 10,
    previous_weights: Sequence[float] | None = None,
    max_turnover: float = 0.50,
    transaction_costs: float | Sequence[float] = 0.001,
    solver: str | None = None,
) -> PortfolioOptimizationRun:
    """Prepare inputs and run Easy or Hard portfolio optimization.

    `previous_weights` must follow the same ticker order as the prediction
    slice. If omitted, an equal-weight starting portfolio is assumed.
    """
    if mode not in ("easy", "hard"):
        raise ValueError("mode must be either 'easy' or 'hard'.")

    prediction_slice = prepare_prediction_slice(
        predictions=predictions,
        decision_date=decision_date,
        eligible_tickers=eligible_tickers,
    )

    covariance_estimate = estimate_covariance(
        historical_data=historical_data,
        decision_date=prediction_slice.decision_date,
        tickers=prediction_slice.tickers,
        window=covariance_window,
        min_observations=min_covariance_observations,
    )

    if covariance_estimate.tickers != prediction_slice.tickers:
        raise RuntimeError(
            "Ticker-order mismatch between predictions and covariance matrix."
        )
    if covariance_estimate.decision_date != prediction_slice.decision_date:
        raise RuntimeError(
            "Decision-date mismatch between predictions and covariance estimate."
        )

    tickers = prediction_slice.tickers
    mu = prediction_slice.expected_returns
    sigma = covariance_estimate.covariance
    n_assets = len(tickers)

    if previous_weights is None:
        previous = np.full(n_assets, 1.0 / n_assets)
    else:
        previous = np.asarray(previous_weights, dtype=float)

    if previous.shape != (n_assets,):
        raise ValueError(
            "previous_weights must have one value per ticker in prediction order."
        )
    if not np.isfinite(previous).all():
        raise ValueError("previous_weights must contain finite values.")
    if np.any(previous < -1e-8):
        raise ValueError("previous_weights cannot contain negative values.")
    if not np.isclose(previous.sum(), 1.0, atol=1e-6):
        raise ValueError("previous_weights must sum to 1.")

    if mode == "easy":
        easy_constraints = PortfolioConstraints(
            fully_invested=True,
            long_only=True,
            max_weight=max_weight,
            max_turnover=None,
            max_holdings=None,
        )

        result = optimize_mean_variance(
            expected_returns=mu,
            covariance=sigma,
            tickers=tickers,
            risk_aversion=risk_aversion,
            constraints=easy_constraints,
            solver=solver,
        )
        weights = np.asarray(result.weights, dtype=float)

        validate_portfolio_weights(
            weights=weights,
            constraints=PortfolioConstraints(
                fully_invested=True,
                long_only=True,
                max_weight=max_weight,
            ),
        )
        turnover = float(0.5 * np.abs(weights - previous).sum())
        transaction_cost = None

    else:
        result = optimize_constrained_portfolio(
            expected_returns=mu,
            covariance=sigma,
            tickers=tickers,
            previous_weights=previous,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            max_holdings=max_holdings,
            max_turnover=max_turnover,
            transaction_costs=transaction_costs,
            solver=solver,
        )
        weights = np.asarray(result.weights, dtype=float)
        turnover = float(result.turnover)
        transaction_cost = float(result.transaction_cost)

        validate_portfolio_weights(
            weights=weights,
            constraints=PortfolioConstraints(
                fully_invested=True,
                long_only=True,
                max_weight=max_weight,
                max_turnover=max_turnover,
                max_holdings=max_holdings,
            ),
            previous_weights=previous,
        )

    expected_return = float(mu @ weights)
    variance = float(weights @ sigma @ weights)
    if variance < -1e-10:
        raise RuntimeError(f"Calculated portfolio variance is negative: {variance}.")
    variance = max(0.0, variance)

    return PortfolioOptimizationRun(
        mode=mode,
        decision_date=prediction_slice.decision_date,
        tickers=tickers,
        expected_returns=mu.copy(),
        covariance=sigma.copy(),
        covariance_observations=covariance_estimate.observations,
        prediction_model_versions=prediction_slice.model_versions,
        weights=weights.copy(),
        expected_portfolio_return=expected_return,
        portfolio_variance=variance,
        portfolio_volatility=float(np.sqrt(variance)),
        objective_value=float(result.objective_value),
        turnover=turnover,
        transaction_cost=transaction_cost,
        solver=str(result.solver),
        status=str(result.status),
    )
