"""Load, normalize, and audit source datasets for Optimization-Project.

Primary: OPT_DATA.zip
Secondary: nifty500_stocks.csv

This module performs loading and basic type normalization. Filtering primary
records to EQ, resolving duplicates, cross-source imputation, and feature
engineering belong in preprocess.py.
"""
from __future__ import annotations

import argparse
import re
import warnings
import zipfile
from pathlib import Path, PurePosixPath

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

PRIMARY_NUMERIC_FIELDS = list(PRIMARY_RENAME.values())
SECONDARY_NUMERIC_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
EXCLUDED_ARCHIVE_DIRS = {"NIFTY_5_YRS", "VIX_5_YRS"}


def parse_numeric(series: pd.Series) -> pd.Series:
    """Convert formatted numeric strings to numbers; invalid/missing values -> NA."""
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
        .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def symbol_from_filename(filename: str) -> str:
    """Infer a stock symbol from a standard NSE bhavcopy filename."""
    basename = PurePosixPath(filename.replace("\\", "/")).name
    match = re.search(r"TO-\d{2}-\d{2}-\d{4}-(.+?)-ALL-N\.csv$", basename)
    return match.group(1).strip() if match else Path(basename).stem


def _is_candidate_stock_csv(member: str) -> bool:
    """Exclude metadata and index/volatility history folders by path component."""
    normalized = member.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.suffix.lower() != ".csv":
        return False
    parts = {part.upper() for part in path.parts}
    if "__MACOSX" in parts:
        return False
    return not bool(parts & EXCLUDED_ARCHIVE_DIRS)


def _normalize_symbol_column(df: pd.DataFrame, inferred_symbol: str) -> pd.Series:
    """Use the source Symbol when populated, falling back to the filename symbol."""
    if "Symbol" not in df.columns:
        return pd.Series(inferred_symbol, index=df.index, dtype="string")
    symbols = df["Symbol"].astype("string").str.strip()
    return symbols.mask(symbols.isna() | symbols.eq(""), inferred_symbol)


def load_primary_zip(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs from OPT_DATA.zip and return normalized rows plus file inventory.

    Invalid dates are retained as NaT in the primary rows so they remain visible
    for auditing. Their counts are recorded in the inventory rather than being
    silently dropped here.
    """
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if _is_candidate_stock_csv(name)]
        if not members:
            raise ValueError(
                "No candidate stock CSV files found in the primary ZIP after exclusions."
            )

        for member in members:
            filename = PurePosixPath(member.replace("\\", "/")).name
            inferred_symbol = symbol_from_filename(filename)
            try:
                df = pd.read_csv(archive.open(member))
            except Exception as exc:
                inventory.append({
                    "file": member,
                    "symbol": inferred_symbol,
                    "status": "READ_ERROR",
                    "rows": 0,
                    "eq_rows": 0,
                    "invalid_dates": 0,
                    "missing_numeric_cells": 0,
                    "error": str(exc),
                })
                continue

            df.columns = [str(column).strip() for column in df.columns]
            missing_required = {"Date", "Series"} - set(df.columns)
            if missing_required:
                inventory.append({
                    "file": member,
                    "symbol": inferred_symbol,
                    "status": "MISSING_REQUIRED_COLUMNS",
                    "rows": len(df),
                    "eq_rows": 0,
                    "invalid_dates": 0,
                    "missing_numeric_cells": 0,
                    "error": f"Missing columns: {sorted(missing_required)}",
                })
                continue

            raw_rows = len(df)
            df["Date"] = pd.to_datetime(
                df["Date"].astype("string").str.strip(),
                format="%d-%b-%Y",
                errors="coerce",
            )
            invalid_dates = int(df["Date"].isna().sum())
            df["Symbol"] = _normalize_symbol_column(df, inferred_symbol)
            df["Series"] = df["Series"].astype("string").str.strip().str.upper()

            # Standardize known numeric column names and their data types.
            rename_map = {old: new for old, new in PRIMARY_RENAME.items() if old in df.columns}
            df = df.rename(columns=rename_map)
            duplicate_columns = df.columns[df.columns.duplicated()].unique().tolist()
            if duplicate_columns:
                raise ValueError(
                    f"Column normalization created duplicate column names in {member}: "
                    f"{duplicate_columns}"
                )

            missing_numeric_cells = 0
            for column in PRIMARY_NUMERIC_FIELDS:
                if column in df.columns:
                    df[column] = parse_numeric(df[column])
                    missing_numeric_cells += int(df[column].isna().sum())

            df["source_primary_file"] = filename
            df["source_archive_member"] = member
            frames.append(df)

            inventory.append({
                "file": member,
                "symbol": inferred_symbol,
                "status": "LOADED",
                "rows": raw_rows,
                "eq_rows": int((df["Series"] == "EQ").sum()),
                "invalid_dates": invalid_dates,
                "missing_numeric_cells": missing_numeric_cells,
                "error": "",
            })

    if not frames:
        raise ValueError(
            "No readable stock CSVs with required Date and Series columns were found. "
            "Inspect the ZIP contents and source_inventory logic."
        )

    primary = pd.concat(frames, ignore_index=True, sort=False)
    if not primary["Date"].notna().any():
        warnings.warn(
            "No valid dates were parsed in any primary CSV. Check the expected date format %d-%b-%Y.",
            RuntimeWarning,
        )
    invalid_total = int(primary["Date"].isna().sum())
    if invalid_total:
        warnings.warn(
            f"Primary data contains {invalid_total:,} rows with invalid dates; "
            "they were retained as NaT and listed in source_inventory.csv.",
            RuntimeWarning,
        )

    primary = primary.sort_values(["Symbol", "Date"], kind="stable", na_position="last").reset_index(drop=True)
    return primary, pd.DataFrame(inventory)


def load_secondary_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize the secondary historical stock CSV.

    Invalid key rows are dropped with a warning. Duplicate Symbol+Date keys are
    reported and collapsed to the last row in source-file order.
    """
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    missing = {"Date", "Symbol"} - set(df.columns)
    if missing:
        raise ValueError(f"Secondary CSV missing required columns: {sorted(missing)}")

    expected_data_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing_data_fields = expected_data_fields - set(df.columns)
    if missing_data_fields:
        warnings.warn(
            "Secondary CSV is missing expected price/volume columns: "
            f"{sorted(missing_data_fields)}. Filling will be limited to available columns.",
            RuntimeWarning,
        )

    raw_rows = len(df)
    df["Date"] = pd.to_datetime(
        df["Date"].astype("string").str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )
    symbols = df["Symbol"].astype("string").str.strip()
    df["Symbol"] = symbols.mask(symbols.isna() | symbols.eq(""), pd.NA)

    invalid_dates = int(df["Date"].isna().sum())
    invalid_symbols = int(df["Symbol"].isna().sum())
    if invalid_dates or invalid_symbols:
        warnings.warn(
            f"Secondary CSV: dropping {int((df['Date'].isna() | df['Symbol'].isna()).sum()):,} rows "
            f"with invalid keys (invalid dates={invalid_dates:,}, blank symbols={invalid_symbols:,}).",
            RuntimeWarning,
        )
    df = df.dropna(subset=["Date", "Symbol"]).copy()

    for column in SECONDARY_NUMERIC_FIELDS:
        if column in df.columns:
            df[column] = parse_numeric(df[column])

    
    key_cols = ["Symbol", "Date"]
    duplicate_mask = df.duplicated(key_cols, keep=False)
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        duplicate_df = df.loc[duplicate_mask]

        grouped = duplicate_df.groupby(key_cols, sort=False)
        duplicate_group_count = grouped.ngroups

        compare_cols = [
            col for col in SECONDARY_NUMERIC_FIELDS
            if col in df.columns
        ]

        if compare_cols:
            variation = grouped[compare_cols].nunique(dropna=True)

            # Differences among non-missing values are real conflicts.
            ohlcv_cols = [
                col for col in ["Open", "High", "Low", "Close", "Volume"]
                if col in variation.columns
            ]

            if ohlcv_cols:
                conflict_mask = variation[ohlcv_cols].gt(1).any(axis=1)
                conflicting = variation.loc[conflict_mask]
            else:
                conflicting = variation.iloc[0:0]

            conflict_count = len(conflicting)

            # Stop rather than arbitrarily selecting a conflicting price.
            if conflict_count:
                examples = [
                    (symbol, str(date.date()))
                    for symbol, date in conflicting.index[:5]
                ]

                raise ValueError(
                    f"Found {conflict_count:,} duplicate Symbol+Date "
                    "groups with conflicting OHLCV values. Resolve these "
                    f"before proceeding. Examples: {examples}"
                )

            # Adjusted-close differences are reported separately.
            if "Adj Close" in variation.columns:
                adj_conflicts = int(
                    variation["Adj Close"].gt(1).sum()
                )

                if adj_conflicts:
                    warnings.warn(
                        f"{adj_conflicts:,} duplicate Symbol+Date groups "
                        "have differing Adj Close values, while the "
                        "available OHLCV values agree. Keeping the last "
                        "source row for each key.",
                        RuntimeWarning,
                    )

            print(
                "Duplicate audit: "
                f"{duplicate_count:,} rows in "
                f"{duplicate_group_count:,} duplicate groups; "
                f"{conflict_count:,} OHLCV-conflicting groups."
            )

        else:
            warnings.warn(
                f"Found {duplicate_count:,} rows with duplicate "
                "Symbol+Date keys, but no numeric market columns are "
                "available to validate the duplicates.",
                RuntimeWarning,
            )

    # Stable sorting preserves original source order within duplicate keys.
    df = (
        df.sort_values(["Symbol", "Date"], kind="stable")
          .drop_duplicates(["Symbol", "Date"], keep="last")
          .reset_index(drop=True)
    )


    if df.empty:
        raise ValueError(
            "No valid secondary records remain after date/symbol validation. "
            "Check the source columns and expected date format %d-%m-%Y."
        )

    print(
        "Secondary load summary: "
        f"raw rows={raw_rows:,}, valid unique rows={len(df):,}, "
        f"invalid dates={invalid_dates:,}, blank symbols={invalid_symbols:,}, "
        f"duplicate-key rows={duplicate_count:,}"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Path to OPT_DATA.zip")
    parser.add_argument("--secondary", required=True, help="Path to nifty500_stocks.csv")
    parser.add_argument("--output", default="data/interim", help="Output directory")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    primary, inventory = load_primary_zip(args.primary)
    secondary = load_secondary_csv(args.secondary)

    primary.to_csv(output / "primary_loaded.csv", index=False)
    secondary.to_csv(output / "secondary_loaded.csv", index=False)
    inventory.to_csv(output / "source_inventory.csv", index=False)

    statuses = inventory["status"].value_counts().to_dict()
    invalid_primary_dates = int(primary["Date"].isna().sum())
    print(f"Primary rows loaded: {len(primary):,}")
    print(f"Primary symbols: {primary['Symbol'].nunique(dropna=True):,}")
    print(f"Primary invalid dates retained for audit: {invalid_primary_dates:,}")
    print(f"Primary archive-file statuses: {statuses}")
    print(f"Secondary rows loaded: {len(secondary):,}")
    print(f"Output directory: {output.resolve()}")


if __name__ == "__main__":
    main()
