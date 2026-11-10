"""
End-to-End Portfolio Optimization Runner

Connects the walk-forward ML predictions with the historical risk model
and the portfolio constraints to generate target weights for a specific date.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure the src module can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.optimization.optimizer import optimize_portfolio


def load_previous_weights(filepath: Path, expected_tickers: tuple[str, ...]) -> np.ndarray:
    """Load and perfectly align previous portfolio weights to the optimizer's ticker order."""
    if not filepath.exists():
        raise FileNotFoundError(f"Previous weights file not found: {filepath}")
        
    df = pd.read_csv(filepath)
    if not {'ticker', 'weight'}.issubset(df.columns):
        raise ValueError("Previous weights file must contain 'ticker' and 'weight' columns.")
        
    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0.0)
    
    weight_map = dict(zip(df['ticker'], df['weight']))
    
    aligned_weights = []
    for ticker in expected_tickers:
        aligned_weights.append(weight_map.get(ticker, 0.0))
        
    aligned_array = np.array(aligned_weights, dtype=float)
    
    # Strictly enforce fully invested previous weights for turnover calculations
    total_weight = aligned_array.sum()
    if total_weight > 1e-6:
        aligned_array = aligned_array / total_weight
    else:
        # Fallback to an equal-weight baseline if the previous portfolio is empty
        aligned_array = np.full(len(expected_tickers), 1.0 / len(expected_tickers))
        
    return aligned_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end portfolio optimization.")
    parser.add_argument("--decision-date", required=True, help="Decision date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["easy", "hard"], default="hard", 
                        help="Optimization mode ('easy' is continuous, 'hard' includes cardinality/turnover constraints)")
    parser.add_argument("--predictions", type=Path, 
                        default=PROJECT_ROOT / "data" / "processed" / "predictions.csv",
                        help="Path to ML predictions")
    parser.add_argument("--historical-data", type=Path, 
                        default=PROJECT_ROOT / "data" / "processed" / "adjusted_primary_master.csv",
                        help="Path to cleaned/adjusted historical prices")
    parser.add_argument("--previous-weights", type=Path, default=None,
                        help="Path to CSV containing 'ticker' and 'weight' from the prior rebalance")
    parser.add_argument("--output", type=Path, 
                        default=PROJECT_ROOT / "data" / "processed" / "target_weights.csv",
                        help="Output path for new portfolio weights")
    
    # Portfolio Constraints
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--max-weight", type=float, default=0.20)
    parser.add_argument("--max-holdings", type=int, default=10)
    parser.add_argument("--max-turnover", type=float, default=0.50)
    
    args = parser.parse_args()
    
    target_date = pd.Timestamp(args.decision_date).normalize()
    
    print(f"Loading predictions from {args.predictions.name}...")
    predictions = pd.read_csv(args.predictions)
    
    print(f"Loading historical risk data from {args.historical_data.name}...")
    historical_data = pd.read_csv(args.historical_data)
    
    # Calculate daily returns for the risk model if missing
    if 'daily_return' not in historical_data.columns:
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        historical_data = historical_data.sort_values(by=['Symbol', 'Date'])
        historical_data['daily_return'] = historical_data.groupby('Symbol')['close'].pct_change()
    
    # The optimizer explicitly requires previous weights to strictly match its internal ticker order.
    # We slice the predictions purely to extract the eligible universe for the decision date.
    pred_dated = predictions[pd.to_datetime(predictions['date']).dt.normalize() == target_date]
    if pred_dated.empty:
        print(f"Error: No predictions found for decision date {target_date.date()}.")
        sys.exit(1)
        
    expected_tickers = tuple(sorted(pred_dated['ticker'].astype(str).str.strip().str.upper().unique()))
    
    prev_weights = None
    if args.previous_weights:
        print(f"Loading previous portfolio constraints from {args.previous_weights.name}...")
        prev_weights = load_previous_weights(args.previous_weights, expected_tickers)
        
    print(f"\nRunning {args.mode.upper()} optimization for {target_date.date()}...")
    
    try:
        run_result = optimize_portfolio(
            predictions=predictions,
            historical_data=historical_data,
            decision_date=target_date,
            mode=args.mode,
            risk_aversion=args.risk_aversion,
            max_weight=args.max_weight,
            max_holdings=args.max_holdings,
            max_turnover=args.max_turnover,
            previous_weights=prev_weights
        )
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        sys.exit(1)
        
    # Save target weights
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = run_result.weights_frame()
    out_df.to_csv(args.output)
    
    active_holdings = (out_df['weight'] > 1e-4).sum()
    
    print("\n--- Optimization Complete ---")
    print(f"Solver Status:     {run_result.status} (via {run_result.solver})")
    print(f"Expected Return:   {run_result.expected_portfolio_return:.6f}")
    print(f"Risk (Volatility): {run_result.portfolio_volatility:.6f}")
    print(f"Turnover:          {run_result.turnover:.4f}")
    if run_result.transaction_cost is not None:
        print(f"Transaction Cost:  {run_result.transaction_cost:.6f}")
    print(f"Active Holdings:   {active_holdings} / {args.max_holdings} allowed")
    print(f"Saved targets to:  {args.output}")


if __name__ == '__main__':
    main()