import numpy as np
import pandas as pd
import pytest

from src.optimization.data_interface import prepare_prediction_slice


@pytest.fixture
def predictions():
    return pd.DataFrame([
        {"date": "2025-01-02", "ticker": "BBB", "predicted_return": 0.01,
         "predicted_horizon": "1_trading_day", "return_type": "simple_return", "model_version": "ridge_v1"},
        {"date": "2025-01-02", "ticker": "AAA", "predicted_return": -0.02,
         "predicted_horizon": "1_trading_day", "return_type": "simple_return", "model_version": "ridge_v1"},
        {"date": "2025-01-03", "ticker": "AAA", "predicted_return": 0.03,
         "predicted_horizon": "1_trading_day", "return_type": "simple_return", "model_version": "ridge_v1"},
    ])


def test_orders_predictions_to_requested_ticker_order(predictions):
    result = prepare_prediction_slice(predictions, "2025-01-02", ["AAA", "BBB"])
    assert result.tickers == ("AAA", "BBB")
    np.testing.assert_allclose(result.expected_returns, [-0.02, 0.01])


def test_default_order_is_deterministic(predictions):
    result = prepare_prediction_slice(predictions, "2025-01-02")
    assert result.tickers == ("AAA", "BBB")


def test_missing_eligible_prediction_raises_instead_of_zero_fill(predictions):
    with pytest.raises(ValueError, match="Missing predictions"):
        prepare_prediction_slice(predictions, "2025-01-02", ["AAA", "BBB", "CCC"])


def test_missing_decision_date_raises(predictions):
    with pytest.raises(ValueError, match="No ML predictions"):
        prepare_prediction_slice(predictions, "2025-01-04")


def test_duplicate_date_ticker_raises(predictions):
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate date/ticker"):
        prepare_prediction_slice(duplicated, "2025-01-02")


def test_non_finite_prediction_raises(predictions):
    bad = predictions.copy()
    bad.loc[0, "predicted_return"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        prepare_prediction_slice(bad, "2025-01-02")


def test_wrong_return_type_raises(predictions):
    bad = predictions.copy()
    bad.loc[0, "return_type"] = "log_return"
    with pytest.raises(ValueError, match="return type"):
        prepare_prediction_slice(bad, "2025-01-02")


def test_ticker_normalization_and_duplicate_eligible_tickers(predictions):
    result = prepare_prediction_slice(predictions, "2025-01-02", [" aaa ", "bbb"])
    assert result.tickers == ("AAA", "BBB")
    with pytest.raises(ValueError, match="Ticker list contains duplicates"):
        prepare_prediction_slice(predictions, "2025-01-02", ["aaa", " AAA "])

# Run the test :
# python -m pytest -q tests/test_data_interface.py