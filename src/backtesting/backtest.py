"""
Chronological Portfolio Backtester.
Simulates periodic rebalancing with turnover constraints, transaction costs, and portfolio drift.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.optimization.optimizer import optimize_portfolio
from src.backtesting.metrics import calculate_portfolio_metrics

def align_weights(weights_df: pd.DataFrame, all_tickers: list[str]) -> np.ndarray:
    weight_map = dict(zip(weights_df['ticker'], weights_df['weight']))
    return np.array([weight_map.get(t, 0.0) for t in all_tickers])

def main():
    parser = argparse.ArgumentParser(description="Chronological Portfolio Backtester")
    parser.add_argument("--mode", choices=["easy", "hard"], default="hard")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "data" / "processed" / "predictions.csv")
    parser.add_argument("--historical-data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "adjusted_primary_master.csv")
    parser.add_argument("--benchmark", type=Path, default=PROJECT_ROOT / "data" / "processed" / "nifty_benchmark.csv")
    parser.add_argument("--rebalance-freq", type=int, default=21)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--max-turnover", type=float, default=0.50)
    parser.add_argument("--output-returns", type=Path, default=PROJECT_ROOT / "data" / "processed" / "backtest_returns.csv")
    parser.add_argument("--output-metrics", type=Path, default=PROJECT_ROOT / "data" / "processed" / "backtest_metrics.json")
    args = parser.parse_args()

    print("Loading data for backtest...")
    preds = pd.read_csv(args.predictions)
    hist = pd.read_csv(args.historical_data)
    
    if 'daily_return' not in hist.columns:
        hist['Date'] = pd.to_datetime(hist['Date'])
        hist = hist.sort_values(by=['Symbol', 'Date'])
        hist['daily_return'] = hist.groupby('Symbol')['close'].pct_change()
        
    hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
    preds['date'] = pd.to_datetime(preds['date']).dt.normalize()
    
    returns_wide = hist.pivot(index='Date', columns='Symbol', values='daily_return').fillna(0.0)
    all_tickers = sorted(returns_wide.columns)
    
    valid_dates = sorted(preds['date'].unique())
    rebalance_dates = valid_dates[::args.rebalance_freq]
    
    print(f"Total Trading Days: {len(valid_dates)}")
    print(f"Rebalance events scheduled: {len(rebalance_dates)} (Every {args.rebalance_freq} days)")
    
    portfolio_history = []
    current_weights = np.zeros(len(all_tickers))
    
    for i in tqdm(range(len(rebalance_dates)), desc="Running Walk-Forward Simulation"):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else valid_dates[-1]
        
        prev_w = current_weights if current_weights.sum() > 0 else None
        allowable_turnover = 1.0 if prev_w is None else args.max_turnover
        
        try:
            res = optimize_portfolio(
                predictions=preds,
                historical_data=hist,
                decision_date=reb_date,
                mode=args.mode,
                max_turnover=allowable_turnover,
                previous_weights=prev_w
            )
            target_weights = align_weights(res.weights_frame(), all_tickers)
            turnover = res.turnover
            tcost = turnover * args.transaction_cost
        except Exception:
            target_weights = current_weights.copy()
            turnover = 0.0
            tcost = 0.0
            
        period_dates = [d for d in valid_dates if reb_date < d <= next_reb_date]
        current_weights = target_weights.copy()
        
        if period_dates:
            first_day = period_dates[0]
            day_returns = returns_wide.loc[first_day].reindex(all_tickers).fillna(0.0).values
            port_return = np.dot(current_weights, day_returns) - tcost
            
            portfolio_history.append({'date': first_day, 'portfolio_return': port_return})
            
            current_weights = current_weights * (1 + day_returns)
            current_weights /= current_weights.sum() if current_weights.sum() > 0 else 1.0
            
            for d in period_dates[1:]:
                day_returns = returns_wide.loc[d].reindex(all_tickers).fillna(0.0).values
                port_return = np.dot(current_weights, day_returns)
                portfolio_history.append({'date': d, 'portfolio_return': port_return})
                
                current_weights = current_weights * (1 + day_returns)
                current_weights /= current_weights.sum() if current_weights.sum() > 0 else 1.0
                
    results_df = pd.DataFrame(portfolio_history)
    args.output_returns.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_returns, index=False)
    
    returns_series = results_df.set_index('date')['portfolio_return']
    
    # Load Benchmark
    benchmark_series = None
    if args.benchmark.exists():
        bench_df = pd.read_csv(args.benchmark)
        bench_df['Date'] = pd.to_datetime(bench_df['Date']).dt.normalize()
        benchmark_series = bench_df.set_index('Date')['daily_return']
    else:
        print(f"\nWarning: Benchmark data not found at {args.benchmark}. Alpha and Beta will not be calculated.")

    metrics = calculate_portfolio_metrics(returns_series, benchmark_returns=benchmark_series)
    
    with open(args.output_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Backtest Complete ---")
    print(f"Annualized Return:     {metrics.get('annualized_return', 0.0):.2%}")
    print(f"Annualized Volatility: {metrics.get('annualized_volatility', 0.0):.2%}")
    print(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"Max Drawdown:          {metrics.get('max_drawdown', 0.0):.2%}")
    if 'annualized_alpha' in metrics:
        print(f"Annualized Alpha:      {metrics['annualized_alpha']:.2%}")
        print(f"Beta vs Benchmark:     {metrics['beta']:.2f}")
    
    print(f"\nSaved returns array: {args.output_returns}")
    print(f"Saved metrics JSON:  {args.output_metrics}")

if __name__ == '__main__':
    main()