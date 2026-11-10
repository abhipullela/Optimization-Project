"""Extract and clean NIFTY benchmark data for portfolio evaluation."""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract NIFTY benchmark data")
    parser.add_argument("--primary", type=Path, default=PROJECT_ROOT / "data" / "raw" / "OPT_DATA.zip")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "nifty_benchmark.csv")
    args = parser.parse_args()

    if not args.primary.exists():
        raise FileNotFoundError(f"Primary zip not found at {args.primary}")

    frames = []
    with zipfile.ZipFile(args.primary) as archive:
        candidates = [
            f for f in archive.namelist() 
            if "NIFTY_5_YRS" in f.upper() and f.upper().endswith(".CSV") and "NIFTY 50" in f.upper()
        ]
        
        if not candidates:
            raise FileNotFoundError("Could not find NIFTY 50 CSVs in the zip.")
            
        print(f"Found {len(candidates)} NIFTY 50 files. Extracting and merging...")
        
        for target in candidates:
            df = pd.read_csv(archive.open(target), encoding='ISO-8859-1')
            df.columns = [str(c).strip() for c in df.columns]
            
            # Safely find the required columns
            date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
            close_col = next((c for c in df.columns if 'CLOSE' in c.upper()), None)
            
            if not date_col or not close_col:
                print(f"Skipping {target}: Missing Date/Close columns. Found headers: {df.columns.tolist()}")
                continue
                
            df['Date'] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce').dt.normalize()
            df['close'] = pd.to_numeric(df[close_col].astype(str).str.replace(',', ''), errors='coerce')
            
            frames.append(df[['Date', 'close']].dropna())
    
    if not frames:
        raise ValueError("Failed to extract valid data from any NIFTY 50 file. Check the CSV headers printed above.")

    # Combine all chunks
    master_df = pd.concat(frames, ignore_index=True)
    
    # Sort chronologically and drop duplicates in case of overlapping files
    master_df = master_df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Calculate daily returns for the index
    master_df['daily_return'] = master_df['close'].pct_change()
    master_df = master_df.dropna(subset=['daily_return'])
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master_df[['Date', 'close', 'daily_return']].to_csv(args.output, index=False)
    print(f"Saved complete benchmark data ({len(master_df)} days) to: {args.output.name}")

if __name__ == '__main__':
    main()