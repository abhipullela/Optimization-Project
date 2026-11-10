"""
Calculates standard financial performance metrics for the portfolio.
"""
import numpy as np
import pandas as pd

def calculate_drawdowns(returns: pd.Series) -> pd.Series:
    """Calculate the drawdowns from the running maximum."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max

def calculate_portfolio_metrics(
    daily_returns: pd.Series, 
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.05,
    trading_days: int = 252
) -> dict:
    """Calculate annualized return, volatility, Sharpe, Sortino, and Drawdowns."""
    if daily_returns.empty:
        return {}

    cumulative_return = (1 + daily_returns).prod() - 1
    years = len(daily_returns) / trading_days
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0.0

    daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(trading_days)

    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = daily_returns - daily_rf
    
    sharpe_ratio = (excess_returns.mean() / daily_vol) * np.sqrt(trading_days) if daily_vol > 0 else 0.0

    downside_returns = excess_returns[excess_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days)
    sortino_ratio = (excess_returns.mean() * trading_days) / downside_vol if downside_vol > 0 else 0.0

    drawdowns = calculate_drawdowns(daily_returns)
    max_drawdown = drawdowns.min()

    metrics = {
        "total_trading_days": len(daily_returns),
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown)
    }
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned_returns, aligned_bench = daily_returns.align(benchmark_returns, join='inner')
        if not aligned_returns.empty:
            cov = np.cov(aligned_returns, aligned_bench)[0, 1]
            var = np.var(aligned_bench)
            beta = cov / var if var > 0 else 1.0
            alpha = (aligned_returns.mean() - daily_rf) - beta * (aligned_bench.mean() - daily_rf)
            metrics["beta"] = float(beta)
            metrics["annualized_alpha"] = float(alpha * trading_days)
            
    return metrics