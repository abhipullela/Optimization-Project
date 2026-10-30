"""
Shared portfolio constraints and validation utilities.

Portfolio conventions:
    - Fully invested portfolio by default.
    - Long-only positions.
    - Optional maximum position size.
    - Optional turnover limit.
    - Optional cardinality limit.

This module validates portfolio inputs and solutions. It does not solve
an optimization problem; the optimization modules will use these rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioConstraints:
    # Configuration for portfolio feasibility constraints.
    
    # Parameters :
    fully_invested: bool = True #If True, portfolio weights must sum to 1.
    long_only: bool = True #If True, short positions are forbidden.
    max_weight: float | None = 1.0 #Maximum weight allowed for any individual stock.
                                   #Use None for no additional upper bound beyond the budget.
    max_turnover: float | None = None #Optional maximum one-way turnover, defined as [0.5 * sum(abs(new_weights - old_weights))]
    max_holdings: int | None = None #Optional maximum number of nonzero holdings
    budget_tolerance: float = 1e-6 # Numerical tolerance for the budget constraint
    weight_tolerance: float = 1e-8 # Numerical tolerance when validating individual weights

    def __post_init__(self) -> None:
        if self.max_weight is not None:
            if not np.isfinite(self.max_weight):
                raise ValueError("max_weight must be finite or None.")

            if self.max_weight <= 0:
                raise ValueError("max_weight must be greater than zero.")

            if self.max_weight > 1:
                raise ValueError(
                    "For this long-only, fully invested formulation, "
                    "max_weight cannot exceed 1."
                )

        if self.max_turnover is not None:
            if not np.isfinite(self.max_turnover):
                raise ValueError("max_turnover must be finite or None.")

            if self.max_turnover < 0:
                raise ValueError("max_turnover cannot be negative.")

        if self.max_holdings is not None:
            if (
                not isinstance(self.max_holdings, int)
                or isinstance(self.max_holdings, bool)
                or self.max_holdings < 1
            ):
                raise ValueError(
                    "max_holdings must be a positive integer or None."
                )

        if self.budget_tolerance <= 0:
            raise ValueError("budget_tolerance must be positive.")

        if self.weight_tolerance <= 0:
            raise ValueError("weight_tolerance must be positive.")


def _validate_weight_vector(weights: np.ndarray) -> np.ndarray:
    """Validate and return a one-dimensional finite weight vector."""
    values = np.asarray(weights, dtype=float)

    if values.ndim != 1:
        raise ValueError("Portfolio weights must be a one-dimensional vector.")

    if values.size == 0:
        raise ValueError("Portfolio weights cannot be empty.")

    if not np.isfinite(values).all():
        raise ValueError("Portfolio weights must contain only finite values.")

    return values


def validate_previous_weights(
    previous_weights: np.ndarray,
    n_assets: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Validate the previous portfolio weights.

    Previous weights are expected to represent a fully invested,
    long-only portfolio over the same ordered asset universe.
    """
    values = _validate_weight_vector(previous_weights)

    if values.size != n_assets:
        raise ValueError(
            f"Expected {n_assets} previous weights, got {values.size}."
        )

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")

    if np.any(values < -tolerance):
        raise ValueError("Previous weights cannot contain short positions.")

    if not np.isclose(values.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Previous portfolio weights must sum to 1. "
            f"Received {values.sum():.10f}."
        )

    # Remove tiny numerical negatives within tolerance.
    values = np.where(values < 0, 0.0, values)

    # Do not silently normalize a materially invalid portfolio.
    if not np.isclose(values.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Previous weights are invalid after numerical tolerance handling."
        )

    return values


def calculate_turnover(
    new_weights: np.ndarray,
    previous_weights: np.ndarray,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Definition:
        turnover = 0.5 * sum(abs(new_weights - previous_weights))

    Both weight vectors must use exactly the same asset order.
    """
    new_values = _validate_weight_vector(new_weights)
    old_values = _validate_weight_vector(previous_weights)

    if new_values.shape != old_values.shape:
        raise ValueError(
            "New and previous weight vectors must have the same shape."
        )

    return float(0.5 * np.abs(new_values - old_values).sum())


def validate_portfolio_weights(
    weights: np.ndarray,
    constraints: PortfolioConstraints | None = None,
    previous_weights: np.ndarray | None = None,
    tolerance: float | None = None,
) -> dict[str, float | int | bool]:
    """
    Validate a portfolio weight vector against configured constraints.

    Returns a summary of the portfolio's budget, minimum and maximum
    weights, number of holdings, and turnover when applicable.

    Raises
    ------
    ValueError
        If the portfolio violates any configured constraint.
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    values = _validate_weight_vector(weights)

    weight_tolerance = (
        constraints.weight_tolerance if tolerance is None else tolerance
    )

    if not np.isfinite(weight_tolerance) or weight_tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")

    budget = float(values.sum())
    minimum_weight = float(values.min())
    maximum_weight = float(values.max())

    # Budget constraint.
    if constraints.fully_invested:
        if not np.isclose(
            budget,
            1.0,
            atol=constraints.budget_tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                "Portfolio weights must sum to 1. "
                f"Received {budget:.10f}."
            )

    # Long-only constraint.
    if constraints.long_only and np.any(values < -weight_tolerance):
        raise ValueError(
            "Long-only constraint violated: at least one weight is negative."
        )

    # Position-size constraint.
    if constraints.max_weight is not None:
        if np.any(values > constraints.max_weight + weight_tolerance):
            raise ValueError(
                "Maximum position weight exceeded. "
                f"Configured maximum: {constraints.max_weight:.6f}; "
                f"observed maximum: {maximum_weight:.6f}."
            )

    # Holdings count.
    holdings = int(np.count_nonzero(np.abs(values) > weight_tolerance))

    if constraints.max_holdings is not None:
        if holdings > constraints.max_holdings:
            raise ValueError(
                "Cardinality constraint violated: "
                f"{holdings} holdings exceed the maximum of "
                f"{constraints.max_holdings}."
            )

    # Turnover constraint.
    turnover = None

    if constraints.max_turnover is not None:
        if previous_weights is None:
            raise ValueError(
                "previous_weights must be supplied when max_turnover "
                "is configured."
            )

        old_values = validate_previous_weights(
            previous_weights,
            n_assets=values.size,
            tolerance=constraints.budget_tolerance,
        )

        turnover = calculate_turnover(values, old_values)

        if turnover > constraints.max_turnover + weight_tolerance:
            raise ValueError(
                "Turnover constraint violated. "
                f"Observed: {turnover:.6f}; "
                f"maximum: {constraints.max_turnover:.6f}."
            )

    return {
        "budget": budget,
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
        "number_of_holdings": holdings,
        "turnover": turnover,
        "is_long_only": bool(np.all(values >= -weight_tolerance)),
    }