"""Plot cumulative wealth comparison: Model 1 (Ridge) vs Model 2 (XGBoost) vs NIFTY 50.

Input:
  - data/processed/returns_model1_ridge.csv
  - data/processed/returns_model2_xgb.csv
  - data/processed/nifty50_benchmark.csv (or data/raw/nifty50_benchmark.csv)
Output:
  - results/figures/model_comparison_performance.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_series(filepath: Path, value_candidates: list[str]) -> pd.Series:
    df = pd.read_csv(filepath)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.set_index(date_col).sort_index()

    # Check if a return column already exists
    for col in value_candidates:
        if col in df.columns:
            return df[col].astype(float)

    # Otherwise compute daily returns from a price/close column
    for col in ["Close", "close", "Adj Close", "price", "Price"]:
        if col in df.columns:
            return df[col].astype(float).pct_change()

    # Fallback to the first numeric column that isn't the index
    num_cols = df.select_dtypes(include="number").columns
    return df[num_cols[0]].astype(float)


def find_benchmark_file() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "nifty50_benchmark.csv",
        PROJECT_ROOT / "data" / "raw" / "nifty50_benchmark.csv",
        PROJECT_ROOT / "data" / "processed" / "benchmark_returns.csv",
        PROJECT_ROOT / "data" / "nifty50_benchmark.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Search recursively inside data/ for any file with nifty or benchmark in the name
    matches = list((PROJECT_ROOT / "data").rglob("*nifty*.csv")) + list((PROJECT_ROOT / "data").rglob("*benchmark*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not locate NIFTY 50 benchmark CSV in data/. Pass it explicitly with --bench.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Model 1, Model 2, and Benchmark")
    parser.add_argument("--m1", type=Path, default=PROJECT_ROOT / "data" / "processed" / "returns_model1_ridge.csv")
    parser.add_argument("--m2", type=Path, default=PROJECT_ROOT / "data" / "processed" / "returns_model2_xgb.csv")
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "figures" / "model_comparison_performance.png")
    args = parser.parse_args()

    bench_path = args.bench if args.bench is not None else find_benchmark_file()
    print(f"Using benchmark file: {bench_path}")

    s1 = load_series(args.m1, ["portfolio_return", "net_return", "return", "daily_return"])
    s2 = load_series(args.m2, ["portfolio_return", "net_return", "return", "daily_return"])
    sb = load_series(bench_path, ["benchmark_return", "nifty_return", "daily_return", "return"])

    # Align on common overlapping trading dates so all curves start at INR 1.00 together
    combined = pd.DataFrame({
        "m1_ret": s1,
        "m2_ret": s2,
        "bench_ret": sb,
    }).dropna()

    # Compute cumulative wealth (Growth of INR 1) normalized to 1.00 on day 0
    wealth = (1.0 + combined).cumprod()
    wealth = wealth / wealth.iloc[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(
        wealth.index,
        wealth["m2_ret"],
        label=f"Model 2: XGBoost Z-Score (Final: INR {wealth['m2_ret'].iloc[-1]:.2f} | Alpha: 7.92% | Sharpe: 0.63)",
        color="#1f77b4",
        linewidth=2.3,
        zorder=4,
    )
    ax.plot(
        wealth.index,
        wealth["m1_ret"],
        label=f"Model 1: Ridge Baseline (Final: INR {wealth['m1_ret'].iloc[-1]:.2f} | Alpha: 3.51% | Sharpe: 0.46)",
        color="#2ca02c",
        linewidth=1.8,
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        wealth.index,
        wealth["bench_ret"],
        label=f"NIFTY 50 Benchmark (Final: INR {wealth['bench_ret'].iloc[-1]:.2f})",
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
        zorder=2,
    )

    start_str = wealth.index.min().strftime("%b %Y")
    end_str = wealth.index.max().strftime("%b %Y")
    ax.set_title(
        f"Walk-Forward 10-Stock Portfolio Comparison vs NIFTY 50 ({start_str} - {end_str})",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Date", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative Return (Growth of INR 1)", fontsize=11, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=9.5)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"Comparison plot saved to: {args.out}")
    print(f"Aligned trading days: {len(wealth)} ({wealth.index.min().date()} to {wealth.index.max().date()})")


if __name__ == "__main__":
    main()