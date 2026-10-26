"""
Load source datasets for Optimization-Project.

Primary: OPT_DATA.zip
Secondary: nifty500_stocks.csv

This module only loads/normalizes source data. Use preprocess.py for filtering,
deduplication, secondary-source enrichment, and feature preparation.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


PRIMARY_RENAME = {
    "Prev Close": "prev_close",
    "Open Price": "open",
    "High Price": "high",
    "Low Price": "low",
    "Last Price": "last",
    "Close Price": "close",
    "Average Price": "average_price",
    "Total Traded Quantity": "volume",
    "Turnover ₹": "turnover_inr",
    "No. of Trades": "number_of_trades",
    "Deliverable Qty": "deliverable_quantity",
    "% Dly Qt to Traded Qty": "deliverable_pct",
}


def parse_numeric(series: pd.Series) -> pd.Series:
    """Convert formatted numeric strings to numeric values."""
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
        .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def symbol_from_filename(filename: str) -> str:
    match = re.search(r"TO-\d{2}-\d{2}-\d{4}-(.+?)-ALL-N\.csv$", filename)
    return match.group(1).strip() if match else Path(filename).stem


def load_primary_zip(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stock CSVs from OPT_DATA.zip; also return a file inventory."""
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []

    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and "/NIFTY_5_YRS/" not in name
            and "/VIX_5_YRS/" not in name
        ]

        for member in members:
            filename = Path(member).name
            symbol = symbol_from_filename(filename)
            try:
                df = pd.read_csv(archive.open(member))
            except Exception as exc:
                inventory.append({
                    "file": member, "symbol": symbol, "status": "READ_ERROR",
                    "rows": 0, "error": str(exc),
                })
                continue

            df.columns = [str(c).strip() for c in df.columns]
            if "Date" not in df.columns or "Series" not in df.columns:
                inventory.append({
                    "file": member, "symbol": symbol,
                    "status": "MISSING_DATE_OR_SERIES", "rows": len(df), "error": "",
                })
                continue

            df["Date"] = pd.to_datetime(
                df["Date"].astype(str).str.strip(),
                format="%d-%b-%Y",
                errors="coerce",
            )
            df["Symbol"] = (
                df["Symbol"].astype("string").str.strip()
                if "Symbol" in df.columns else symbol
            )
            df["Series"] = df["Series"].astype("string").str.strip()
            df["source_primary_file"] = filename
            df["source_archive_member"] = member
            frames.append(df)

            inventory.append({
                "file": member, "symbol": symbol, "status": "LOADED",
                "rows": len(df), "eq_rows": int((df["Series"] == "EQ").sum()),
                "invalid_dates": int(df["Date"].isna().sum()), "error": "",
            })

    if not frames:
        raise ValueError("No readable stock CSVs found in the primary ZIP.")
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(inventory)


def load_secondary_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize the secondary historical stock CSV."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = {"Date", "Symbol"} - set(df.columns)
    if missing:
        raise ValueError(f"Secondary CSV missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str).str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )
    df["Symbol"] = df["Symbol"].astype("string").str.strip()
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = parse_numeric(df[col])

    df = df.dropna(subset=["Date", "Symbol"])
    return (
        df.sort_values(["Symbol", "Date"])
          .drop_duplicates(["Symbol", "Date"], keep="last")
          .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--output", default="data/interim")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    primary, inventory = load_primary_zip(args.primary)
    secondary = load_secondary_csv(args.secondary)

    primary.to_csv(output / "primary_loaded.csv", index=False)
    secondary.to_csv(output / "secondary_loaded.csv", index=False)
    inventory.to_csv(output / "source_inventory.csv", index=False)
    print(f"Loaded primary rows: {len(primary):,}")
    print(f"Loaded secondary rows: {len(secondary):,}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
