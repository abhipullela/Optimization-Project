"""
Clean and enrich stock data for Optimization-Project.

Primary source: OPT_DATA.zip
Secondary source: nifty500_stocks.csv

Outputs under data/processed:
- cleaned_primary_master.csv (cleaned/consolidated OHLCV data)
- supplemental_dummy_live_replay.csv
- data_quality_report.csv

ML features are generated separately by src/features/build_features.py.

Only exact Symbol+Date matches are used for imputation. Existing primary
values are preserved. Synthetic replay timestamps are NOT real market times.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
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
SECONDARY_FIELDS = {
    "Open": "secondary_open",
    "High": "secondary_high",
    "Low": "secondary_low",
    "Close": "secondary_close",
    "Adj Close": "secondary_adj_close",
    "Volume": "secondary_volume",
}
FILL_MAP = {
    "open": "secondary_open",
    "high": "secondary_high",
    "low": "secondary_low",
    "close": "secondary_close",
    "volume": "secondary_volume",
}
NUMERIC_FIELDS = [
    "prev_close", "open", "high", "low", "last", "close", "average_price",
    "volume", "turnover_inr", "number_of_trades", "deliverable_quantity",
    "deliverable_pct",
]


def parse_numeric(series: pd.Series) -> pd.Series:
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


def load_primary_equities(primary_zip: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audit: list[dict] = []

    with zipfile.ZipFile(primary_zip) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and "/NIFTY_5_YRS/" not in name
            and "/VIX_5_YRS/" not in name
        ]
        for member in members:
            filename = Path(member).name
            inferred_symbol = symbol_from_filename(filename)
            try:
                df = pd.read_csv(archive.open(member))
            except Exception as exc:
                audit.append({
                    "file": filename, "symbol": inferred_symbol, "status": "READ_ERROR",
                    "raw_rows": 0, "eq_rows": 0, "notes": str(exc),
                })
                continue

            df.columns = [str(c).strip() for c in df.columns]
            if "Date" not in df or "Series" not in df:
                audit.append({
                    "file": filename, "symbol": inferred_symbol,
                    "status": "EXCLUDED_MISSING_REQUIRED_COLUMNS",
                    "raw_rows": len(df), "eq_rows": 0,
                    "notes": "Missing Date or Series column.",
                })
                continue

            raw_rows = len(df)
            df["Date"] = pd.to_datetime(
                df["Date"].astype(str).str.strip(),
                format="%d-%b-%Y",
                errors="coerce",
            )
            df["Series"] = df["Series"].astype("string").str.strip()
            if "Symbol" not in df:
                df["Symbol"] = inferred_symbol
            df["Symbol"] = df["Symbol"].astype("string").str.strip()

            invalid_dates = int(df["Date"].isna().sum())
            df = df.dropna(subset=["Date", "Symbol"])
            df = df[df["Series"] == "EQ"].copy()
            if df.empty:
                audit.append({
                    "file": filename, "symbol": inferred_symbol,
                    "status": "EXCLUDED_NO_VALID_EQ_ROWS",
                    "raw_rows": raw_rows, "eq_rows": 0,
                    "invalid_dates": invalid_dates,
                    "duplicate_symbol_dates_removed": 0,
                    "missing_core_numeric_before_fill": "",
                    "secondary_matched_rows": 0, "cells_filled_from_secondary": 0,
                    "notes": "No valid EQ rows; manually verify instrument/series.",
                })
                continue

            duplicate_count = int(df.duplicated(["Symbol", "Date"]).sum())
            df = df.sort_values(["Symbol", "Date"]).drop_duplicates(
                ["Symbol", "Date"], keep="last"
            )
            df = df.rename(columns={k: v for k, v in PRIMARY_RENAME.items() if k in df})

            missing_core = 0
            for col in NUMERIC_FIELDS:
                if col in df:
                    df[col] = parse_numeric(df[col])
                    missing_core += int(df[col].isna().sum())

            df["source_primary_file"] = filename
            df["source_archive_member"] = member
            frames.append(df)
            audit.append({
                "file": filename, "symbol": inferred_symbol, "status": "INCLUDED_EQ",
                "raw_rows": raw_rows, "eq_rows": len(df),
                "invalid_dates": invalid_dates,
                "duplicate_symbol_dates_removed": duplicate_count,
                "missing_core_numeric_before_fill": missing_core,
                "secondary_matched_rows": "", "cells_filled_from_secondary": "",
                "notes": "",
            })

    if not frames:
        raise ValueError("No valid EQ rows found in OPT_DATA.zip.")
    primary = (
        pd.concat(frames, ignore_index=True, sort=False)
          .sort_values(["Symbol", "Date"])
          .reset_index(drop=True)
    )
    return primary, pd.DataFrame(audit)


def load_secondary(secondary_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(secondary_csv)
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
        if col in df:
            df[col] = parse_numeric(df[col])
    return (
        df.dropna(subset=["Date", "Symbol"])
          .sort_values(["Symbol", "Date"])
          .drop_duplicates(["Symbol", "Date"], keep="last")
          .reset_index(drop=True)
    )


def enrich_from_secondary(primary: pd.DataFrame, secondary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep = ["Date", "Symbol"] + [c for c in SECONDARY_FIELDS if c in secondary]
    sec = secondary[keep].rename(columns={
        c: SECONDARY_FIELDS[c] for c in keep if c in SECONDARY_FIELDS
    })
    merged = primary.merge(sec, on=["Date", "Symbol"], how="left", validate="one_to_one")
    matched_rows = int(merged["secondary_close"].notna().sum()) if "secondary_close" in merged else 0

    fill_flags = []
    cells_filled = 0
    for primary_col, secondary_col in FILL_MAP.items():
        flag_col = f"{primary_col}_filled_from_secondary"
        if primary_col in merged and secondary_col in merged:
            can_fill = merged[primary_col].isna() & merged[secondary_col].notna()
            merged[flag_col] = can_fill
            cells_filled += int(can_fill.sum())
            merged.loc[can_fill, primary_col] = merged.loc[can_fill, secondary_col]
        else:
            merged[flag_col] = False
        fill_flags.append(flag_col)

    merged["secondary_date_match"] = (
        merged["secondary_close"].notna()
        if "secondary_close" in merged else False
    )
    merged["primary_value_filled_from_secondary"] = merged[fill_flags].any(axis=1)
    merged["data_source"] = np.where(
        merged["primary_value_filled_from_secondary"],
        "OPT_DATA primary; blank fields filled from secondary",
        np.where(merged["secondary_date_match"],
                 "OPT_DATA primary; secondary match available",
                 "OPT_DATA primary only"),
    )
    return merged, {
        "secondary_matched_rows": matched_rows,
        "cells_filled_from_secondary": cells_filled,
    }


def make_supplemental_replay(
    secondary: pd.DataFrame,
    primary: pd.DataFrame,
    replay_start: str = "2026-10-27T09:15:00+05:30",
) -> pd.DataFrame:
    primary_keys = set(zip(primary["Symbol"].astype(str), primary["Date"]))
    matched_symbols = set(primary["Symbol"].astype(str))
    supplemental = secondary[
        secondary["Symbol"].astype(str).isin(matched_symbols)
    ].copy()
    keys = list(zip(supplemental["Symbol"].astype(str), supplemental["Date"]))
    supplemental = supplemental.loc[[key not in primary_keys for key in keys]].copy()

    supplemental = supplemental.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    g = supplemental.groupby("Symbol", sort=False)
    supplemental["daily_return"] = g["Close"].pct_change()
    g = supplemental.groupby("Symbol", sort=False)
    supplemental["return_lag_1"] = g["daily_return"].shift(1)
    supplemental["return_lag_5"] = g["daily_return"].shift(5)
    supplemental["rolling_volatility_5"] = g["daily_return"].transform(
        lambda s: s.rolling(5, min_periods=5).std()
    )
    supplemental["rolling_volatility_20"] = g["daily_return"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    supplemental["rolling_mean_return_5"] = g["daily_return"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    ma5 = g["Close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma20 = g["Close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    supplemental["moving_average_ratio_5_20"] = ma5 / ma20 - 1
    supplemental["volume_change_1"] = (
        supplemental.groupby("Symbol", sort=False)["Volume"].pct_change()
                    .replace([np.inf, -np.inf], np.nan)
    )
    supplemental["target_next_day_return"] = (
        supplemental.groupby("Symbol", sort=False)["daily_return"].shift(-1)
    )
    supplemental["event_timestamp_ist"] = (
        supplemental["Date"].dt.strftime("%Y-%m-%d") + "T15:30:00+05:30"
    )
    supplemental["event_type"] = "HISTORICAL_REPLAY_RECORD"
    supplemental["data_origin"] = "nifty500_stocks.csv; Symbol+Date absent from OPT_DATA"
    supplemental = supplemental.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    supplemental["replay_sequence"] = np.arange(1, len(supplemental) + 1)

    start = pd.Timestamp(replay_start)
    supplemental["simulated_ingestion_timestamp_ist"] = [
        (start + pd.Timedelta(minutes=i)).isoformat()
        for i in range(len(supplemental))
    ]
    return supplemental



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Path to OPT_DATA.zip")
    parser.add_argument("--secondary", required=True, help="Path to nifty500_stocks.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--replay-start", default="2026-10-27T09:15:00+05:30",
        help="Synthetic ingestion timestamp start; not a market timestamp.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    primary, audit = load_primary_equities(args.primary)
    secondary = load_secondary(args.secondary)
    enriched, stats = enrich_from_secondary(primary, secondary)

    # This output contains cleaned/consolidated market data only.
    # ML features are created separately by src/features/build_features.py.
    cleaned_path = out / "cleaned_primary_master.csv"
    enriched.to_csv(cleaned_path, index=False)

    # Keep the source-level audit report from loading the primary archive.
    audit_path = out / "data_quality_report.csv"
    audit.to_csv(audit_path, index=False)

    # Keep supplemental replay generation separate from the primary dataset.
    replay = make_supplemental_replay(secondary, primary, args.replay_start)
    replay_path = out / "supplemental_dummy_live_replay.csv"
    replay.to_csv(replay_path, index=False)

    print(f"Primary EQ rows: {len(primary):,}; symbols: {primary['Symbol'].nunique()}")
    print(f"Secondary matched rows: {stats['secondary_matched_rows']:,}")
    print(f"Cells filled from secondary: {stats['cells_filled_from_secondary']:,}")
    print(f"Supplemental replay rows: {len(replay):,}")
    print(f"Cleaned master saved to: {cleaned_path.resolve()}")
    print(f"Data quality report saved to: {audit_path.resolve()}")
    print(f"Supplemental replay saved to: {replay_path.resolve()}")
    print(
        "ML feature output was not generated by preprocess.py; "
        "run src/features/build_features.py separately."
    )


if __name__ == "__main__":
    main()
