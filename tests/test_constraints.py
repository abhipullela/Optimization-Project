import numpy as np
import pytest

from src.optimization.constraints import (
    PortfolioConstraints,
    calculate_turnover,
    validate_portfolio_weights,
    validate_previous_weights,
)


def test_valid_fully_invested_portfolio():
    weights = np.array([0.4, 0.35, 0.25])

    result = validate_portfolio_weights(
        weights,
        PortfolioConstraints(),
    )

    assert result["budget"] == pytest.approx(1.0)
    assert result["number_of_holdings"] == 3
    assert result["is_long_only"] is True


def test_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_portfolio_weights(
            np.array([0.4, 0.3, 0.2]),
            PortfolioConstraints(),
        )


def test_rejects_negative_weights_in_long_only_portfolio():
    with pytest.raises(ValueError, match="negative"):
        validate_portfolio_weights(
            np.array([0.7, 0.4, -0.1]),
            PortfolioConstraints(),
        )


def test_rejects_position_limit_violation():
    constraints = PortfolioConstraints(max_weight=0.40)

    with pytest.raises(ValueError, match="Maximum position"):
        validate_portfolio_weights(
            np.array([0.5, 0.3, 0.2]),
            constraints,
        )


def test_rejects_too_many_holdings():
    constraints = PortfolioConstraints(max_holdings=2)

    with pytest.raises(ValueError, match="Cardinality"):
        validate_portfolio_weights(
            np.array([0.4, 0.35, 0.25]),
            constraints,
        )


def test_calculates_one_way_turnover():
    previous = np.array([0.5, 0.3, 0.2])
    current = np.array([0.3, 0.4, 0.3])

    assert calculate_turnover(current, previous) == pytest.approx(0.20)


def test_rejects_turnover_above_limit():
    previous = np.array([0.5, 0.3, 0.2])
    current = np.array([0.3, 0.4, 0.3])

    constraints = PortfolioConstraints(max_turnover=0.10)

    with pytest.raises(ValueError, match="Turnover constraint"):
        validate_portfolio_weights(
            current,
            constraints,
            previous_weights=previous,
        )


def test_turnover_requires_previous_weights():
    constraints = PortfolioConstraints(max_turnover=0.20)

    with pytest.raises(ValueError, match="previous_weights"):
        validate_portfolio_weights(
            np.array([0.5, 0.3, 0.2]),
            constraints,
        )


def test_previous_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_previous_weights(
            np.array([0.5, 0.3, 0.1]),
            n_assets=3,
        )


def test_rejects_non_finite_weights():
    with pytest.raises(ValueError, match="finite"):
        validate_portfolio_weights(
            np.array([0.5, np.nan, 0.5]),
            PortfolioConstraints(),
        )


def test_rejects_invalid_position_cap():
    with pytest.raises(ValueError, match="cannot exceed 1"):
        PortfolioConstraints(max_weight=1.2)


def test_rejects_duplicate_or_invalid_holdings_limit():
    with pytest.raises(ValueError, match="positive integer"):
        PortfolioConstraints(max_holdings=0)

# Run the test :
# python -m pytest -q tests/test_data_interface.py tests/test_risk_model.py tests/test_constraints.py