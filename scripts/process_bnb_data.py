#!/usr/bin/env python3
"""
ETL Pipeline: Resample 1-minute Raw BNB/USDT Data to 1-Hour Continuous Candles.

Reads raw 1m BNB/USDT data from data/ or dataset/csv/bnb/, standardizes columns,
resamples to 1-Hour OHLCV, validates data completeness, and serializes clean
continuous parquet to data/bnb_1h_continuous.parquet (and dataset/processed/bnb/1h/).

Author: Adaptive Trading System Team
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ProcessBNBData")

RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def discover_bnb_raw_files() -> List[Path]:
    """Finds raw 1m BNB data files in data/ or dataset/csv/bnb/."""
    search_dirs = [
        Path("data"),
        Path("dataset/csv/bnb"),
        Path("dataset/csv/BNB"),
        Path("dataset/raw/bnb"),
    ]
    files: List[Path] = []
    for d in search_dirs:
        if d.exists():
            files.extend(list(d.glob("*BNB*.csv")) + list(d.glob("*bnb*.csv")))
            files.extend(list(d.glob("*BNB*.parquet")) + list(d.glob("*bnb*.parquet")))

    # Deduplicate by resolved absolute path
    unique_files = sorted(list({f.resolve() for f in files}))
    return [Path(p) for p in unique_files]


def load_raw_1m_data(file_paths: List[Path]) -> pd.DataFrame:
    """Loads all raw 1m data files into a unified chronologically sorted DataFrame."""
    if not file_paths:
        raise FileNotFoundError("No raw BNB 1m CSV or Parquet files found in data/ or dataset/csv/bnb/.")

    logger.info(f"Discovered {len(file_paths)} raw BNB 1m files. Loading...")
    dfs: List[pd.DataFrame] = []

    for f in file_paths:
        try:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                # Check if file has header
                first_line = pd.read_csv(f, nrows=1)
                if "open_time" in first_line.columns or "timestamp" in first_line.columns:
                    df = pd.read_csv(f)
                else:
                    df = pd.read_csv(f, names=RAW_COLUMNS[:first_line.shape[1]])
            
            # Standardize column naming
            rename_map = {}
            for col in df.columns:
                lower_col = str(col).lower().strip()
                if lower_col in ("open_time", "opentime", "time", "timestamp"):
                    rename_map[col] = "timestamp"
                elif lower_col in ("open", "high", "low", "close", "volume"):
                    rename_map[col] = lower_col
            df.rename(columns=rename_map, inplace=True)
            
            if "timestamp" not in df.columns:
                logger.warning(f"Skipping {f.name}: missing timestamp column")
                continue

            dfs.append(df[["timestamp", "open", "high", "low", "close", "volume"]])
        except Exception as e:
            logger.warning(f"Error loading {f.name}: {e}")

    if not dfs:
        raise ValueError("Failed to parse valid OHLCV data from raw BNB files.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} total 1-minute rows.")

    # Convert numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Timestamp conversion to UTC datetime
    if np.issubdtype(combined["timestamp"].dtype, np.number):
        # Determine if milliseconds or microseconds or seconds
        sample_ts = combined["timestamp"].dropna().iloc[0]
        if sample_ts > 1e14:  # microseconds
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], unit="us", utc=True)
        elif sample_ts > 1e11:  # milliseconds
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)
        else:  # seconds
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], unit="s", utc=True)
    else:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)

    # Sort and remove duplicates
    combined.sort_values("timestamp", inplace=True)
    combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    combined.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def resample_to_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resamples 1-minute OHLCV data to 1-Hour continuous bars."""
    logger.info("Resampling 1-minute candles to 1-Hour OHLCV...")
    df = df_1m.set_index("timestamp")

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_1h = df.resample("1h", closed="left", label="left").agg(agg_rules)

    # Drop bars with missing data
    df_1h.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df_1h.reset_index(inplace=True)

    # Add open_time in ms for system compatibility
    df_1h["open_time"] = df_1h["timestamp"].astype("int64") // 10**6
    df_1h["datetime"] = df_1h["timestamp"]

    # Reorder columns
    col_order = ["timestamp", "open", "high", "low", "close", "volume", "open_time", "datetime"]
    df_1h = df_1h[[c for c in col_order if c in df_1h.columns]]
    df_1h.sort_values("timestamp", inplace=True)
    df_1h.reset_index(drop=True, inplace=True)

    return df_1h


def main():
    raw_files = discover_bnb_raw_files()
    if not raw_files:
        logger.error("No BNB 1m raw files found. Exiting.")
        sys.exit(1)

    df_1m = load_raw_1m_data(raw_files)
    df_1h = resample_to_1h(df_1m)

    # Target output paths
    out_path_data = Path("data/bnb_1h_continuous.parquet")
    out_path_processed = Path("dataset/processed/bnb/1h/bnb_1h_continuous.parquet")

    out_path_data.parent.mkdir(parents=True, exist_ok=True)
    out_path_processed.parent.mkdir(parents=True, exist_ok=True)

    df_1h.to_parquet(out_path_data, index=False)
    df_1h.to_parquet(out_path_processed, index=False)

    logger.info(f"Saved {len(df_1h)} 1H candles to {out_path_data}")
    logger.info(f"Saved mirror copy to {out_path_processed}")

    # Print verification metrics
    start_date = df_1h["timestamp"].iloc[0]
    end_date = df_1h["timestamp"].iloc[-1]
    total_candles = len(df_1h)

    print("\n" + "=" * 70)
    print("               BNB/USDT 1H RESAMPLING VERIFICATION METRICS           ")
    print("=" * 70)
    print(f"Start Timestamp   : {start_date} UTC")
    print(f"End Timestamp     : {end_date} UTC")
    print(f"Total 1H Candles  : {total_candles:,}")
    print(f"File Saved Path   : {out_path_data.resolve()}")
    print("-" * 70)
    print("\nFirst 5 Candles (Snapshot):")
    print(df_1h[["timestamp", "open", "high", "low", "close", "volume"]].head(5).to_string(index=False))
    print("\nLast 5 Candles (Snapshot):")
    print(df_1h[["timestamp", "open", "high", "low", "close", "volume"]].tail(5).to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
