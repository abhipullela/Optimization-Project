"""Generate performance visualizations for the portfolio."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot portfolio vs benchmark returns.")
    parser.add_argument("--returns", type=Path, default=PROJECT_ROOT / "data" / "processed" / "backtest_returns.csv")
    parser.add_argument("--benchmark", type=Path, default=PROJECT_ROOT / "data" / "processed" / "nifty_benchmark.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "figures" / "cumulative_performance.png")
    args = parser.parse_args()

    print("Loading data...")
    port_df = pd.read_csv(args.returns)
    port_df['date'] = pd.to_datetime(port_df['date'])
    port_df = port_df.sort_values('date')

    bench_df = pd.read_csv(args.benchmark)
    bench_df['Date'] = pd.to_datetime(bench_df['Date'])
    bench_df = bench_df.rename(columns={'Date': 'date', 'daily_return': 'benchmark_return'})

    # Merge to ensure dates align perfectly
    merged = pd.merge(port_df, bench_df[['date', 'benchmark_return']], on='date', how='inner')

    # Calculate cumulative growth (starting at 1.0)
    merged['Portfolio'] = (1 + merged['portfolio_return']).cumprod()
    merged['NIFTY 50'] = (1 + merged['benchmark_return']).cumprod()

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Portfolio Line (Blue, thicker)
    plt.plot(merged['date'], merged['Portfolio'], label='Optimized 10-Stock Portfolio', color='#1f77b4', linewidth=2.5)
    
    # Benchmark Line (Orange, thinner, slightly transparent)
    plt.plot(merged['date'], merged['NIFTY 50'], label='NIFTY 50 Benchmark', color='#ff7f0e', linewidth=1.5, alpha=0.8)

    plt.title('Walk-Forward Portfolio Performance vs NIFTY 50 (Dec 2021 - Sep 2026)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Growth of ₹1)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully to: {args.output}")

if __name__ == '__main__':
    main()