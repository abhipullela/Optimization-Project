"""Evaluate the ML ranker using Information Coefficient (IC)."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ranker IC")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "data" / "processed" / "predictions_ranker.csv")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "processed" / "ml_features_primary.csv")
    args = parser.parse_args()

    print("Loading predictions and actuals...")
    preds = pd.read_csv(args.predictions)
    preds['date'] = pd.to_datetime(preds['date'])

    actuals = pd.read_csv(args.features)
    actuals['date'] = pd.to_datetime(actuals['date'])
    
    # Merge predictions with actual next day returns
    df = pd.merge(
        preds[['date', 'ticker', 'predicted_return']],
        actuals[['date', 'Symbol', 'target_next_day_return']],
        left_on=['date', 'ticker'],
        right_on=['date', 'Symbol'],
        how='inner'
    ).dropna()

    daily_ic = []
    
    # Calculate Spearman Rank Correlation for each day
    for date, group in df.groupby('date'):
        if len(group) > 5:  # Need a minimum number of stocks to rank
            corr, _ = spearmanr(group['predicted_return'], group['target_next_day_return'])
            daily_ic.append({'date': date, 'ic': corr})

    ic_df = pd.DataFrame(daily_ic).dropna()
    
    mean_ic = ic_df['ic'].mean()
    ic_ir = mean_ic / ic_df['ic'].std()  # Information Ratio of the IC
    hit_rate = (ic_df['ic'] > 0).mean()

    print("\n--- Model 2 (XGBRanker) Signal Evaluation ---")
    print(f"Total Trading Days Evaluated: {len(ic_df)}")
    print(f"Mean Information Coefficient (IC): {mean_ic:.4f}")
    print(f"IC Information Ratio (IC IR):      {ic_ir:.4f}")
    print(f"Hit Rate (% of days with IC > 0):  {hit_rate:.2%}")
    
    # Rule of thumb for daily quant models:
    # IC > 0.02 is good. IC > 0.05 is excellent.

if __name__ == '__main__':
    main()