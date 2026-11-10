import pandas as pd
import numpy as np
from pathlib import Path

# Phase 1 Corporate Actions Dictionary
# Multi-events on a single ticker (e.g., RELIANCE) will be compounded backward.
CORPORATE_ACTIONS = [
    {"symbol": "TATASTEEL", "ex_date": "2022-07-28", "divisor": 10.0, "event_type": "split_bonus"},
    {"symbol": "BEL", "ex_date": "2022-09-15", "divisor": 3.0, "event_type": "split_bonus"},
    {"symbol": "POWERGRID", "ex_date": "2023-09-12", "divisor": 4/3, "event_type": "split_bonus"},
    {"symbol": "RELIANCE", "ex_date": "2023-07-20", "divisor": 1.10149, "event_type": "demerger"},
    {"symbol": "RELIANCE", "ex_date": "2024-10-28", "divisor": 2.0, "event_type": "split_bonus"},
    {"symbol": "DRREDDY", "ex_date": "2024-10-28", "divisor": 5.0, "event_type": "split_bonus"},
    {"symbol": "WIPRO", "ex_date": "2024-12-03", "divisor": 2.0, "event_type": "split_bonus"},
    {"symbol": "ITC", "ex_date": "2025-01-06", "divisor": 1.05707, "event_type": "demerger"},
    {"symbol": "HDFCBANK", "ex_date": "2025-08-26", "divisor": 2.0, "event_type": "split_bonus"},
    {"symbol": "HINDUNILVR", "ex_date": "2025-12-05", "divisor": 1.01576, "event_type": "demerger"},
    {"symbol": "KOTAKBANK", "ex_date": "2026-01-14", "divisor": 5.0, "event_type": "split_bonus"},
]

PRICE_COLUMNS = ['prev_close', 'open', 'high', 'low', 'last', 'close', 'average_price']
VOLUME_COLUMNS = ['volume']

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

def apply_backward_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    df_adjusted = df.copy()
    
    # Initialize divisor tracking columns
    df_adjusted['price_divisor'] = 1.0
    df_adjusted['volume_divisor'] = 1.0
    
    events_df = pd.DataFrame(CORPORATE_ACTIONS)
    events_df['ex_date'] = pd.to_datetime(events_df['ex_date'])
    
    # Process each symbol independently to prevent cross-contamination
    for symbol in df_adjusted['Symbol'].unique():
        symbol_events = events_df[events_df['symbol'] == symbol].sort_values(by='ex_date', ascending=False)
        
        if symbol_events.empty:
            continue
            
        mask_symbol = df_adjusted['Symbol'] == symbol
        
        # Apply multipliers backward from each ex-date
        for _, event in symbol_events.iterrows():
            # Only adjust dates strictly before the ex-date
            mask_date = df_adjusted['Date'] < event['ex_date']
            mask_apply = mask_symbol & mask_date
            
            # Compound the price divisor
            df_adjusted.loc[mask_apply, 'price_divisor'] *= event['divisor']
            
            # Compound volume divisor only for splits/bonuses
            if event['event_type'] == 'split_bonus':
                df_adjusted.loc[mask_apply, 'volume_divisor'] *= event['divisor']
                
    # Execute the adjustments
    for col in PRICE_COLUMNS:
        if col in df_adjusted.columns:
            df_adjusted[col] = df_adjusted[col] / df_adjusted['price_divisor']
            
    for col in VOLUME_COLUMNS:
        if col in df_adjusted.columns:
            # Volume multiplies by the divisor (e.g. 1:10 split = 10x volume before ex-date)
            df_adjusted[col] = df_adjusted[col] * df_adjusted['volume_divisor']
            
    # Clean up tracking columns to match original schema
    df_adjusted = df_adjusted.drop(columns=['price_divisor', 'volume_divisor'])
    
    return df_adjusted

def main():
    # Dynamically resolve project root based on the location of this script (src/data/adjust_prices.py)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    
    # Map to the data/processed/ directory
    processed_dir = project_root / "data" / "processed"
    input_path = processed_dir / "cleaned_primary_master.csv"
    output_path = processed_dir / "adjusted_primary_master.csv"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")
        
    print(f"Loading raw data from {input_path}...")
    df = load_data(str(input_path))
    
    print("Applying corporate action backward adjustments...")
    df_adjusted = apply_backward_adjustment(df)
    
    print(f"Saving adjusted dataset to {output_path}...")
    df_adjusted.to_csv(output_path, index=False)
    print("Phase 1 adjustment complete.")

if __name__ == "__main__":
    main()