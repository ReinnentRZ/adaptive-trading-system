#!/usr/bin/env python3
"""
Vectorized Data Processing Pipeline for Binance Historical 1-Minute Datasets.

This script processes raw 1-minute OHLCV CSV datasets located in `dataset/csv/{coin}/`
and resamples them into multi-timeframe continuous series (5m, 15m, 30m) saved as
both Parquet and CSV files in `dataset/processed/{coin}/{timeframe}/`.

Author: Adaptive Trading System Team
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetProcessor")

# Raw Binance 1-minute Kline columns definition
RAW_COLUMNS: List[str] = [
    "open_time",       # Microseconds timestamp
    "open",            # Open price (float)
    "high",            # High price (float)
    "low",             # Low price (float)
    "close",           # Close price (float)
    "volume",          # Base asset volume (float)
    "close_time",      # Close timestamp (microseconds)
    "quote_volume",    # Quote asset volume (float)
    "trades_count",    # Number of trades (int)
    "taker_buy_base",  # Taker buy base asset volume (float)
    "taker_buy_quote", # Taker buy quote asset volume (float)
    "ignore",          # Ignore column
]

RAW_DTYPES: Dict[str, str] = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_volume": "float64",
    "trades_count": "int64",
    "taker_buy_base": "float64",
    "taker_buy_quote": "float64",
    "ignore": "str",
}

# Mapping between timeframe strings and pandas resample offset aliases
TIMEFRAME_OFFSET_MAP: Dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}

# OHLCV Resampling Aggregation Rules
RESAMPLE_AGG_RULES: Dict[str, str] = {
    "open_time": "first",
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "close_time": "last",
    "quote_volume": "sum",
    "trades_count": "sum",
    "taker_buy_base": "sum",
    "taker_buy_quote": "sum",
}


@dataclass(frozen=True)
class ProcessingSummary:
    """Summary of data processing results for a specific coin and timeframe."""
    coin: str
    timeframe: str
    row_count: int
    start_date: str
    end_date: str
    parquet_path: Path
    parquet_size_mb: float
    csv_path: Path
    csv_size_mb: float
    elapsed_seconds: float


class DatasetProcessor:
    """
    Vectorized data processing engine for cryptocurrency OHLCV datasets.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        timeframes: Sequence[str] = ("5m", "15m", "30m"),
        parquet_compression: str = "snappy",
    ) -> None:
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeframes = list(timeframes)
        self.parquet_compression = parquet_compression

        # Validate timeframes
        for tf in self.timeframes:
            if tf not in TIMEFRAME_OFFSET_MAP:
                raise ValueError(
                    f"Unsupported timeframe '{tf}'. Supported: {list(TIMEFRAME_OFFSET_MAP.keys())}"
                )

    def discover_coins(self) -> List[str]:
        """
        Discovers all coin subdirectories present in the input directory.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        coins = [
            d.name for d in self.input_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(coins)

    def load_and_concatenate_raw(self, coin: str) -> pd.DataFrame:
        """
        Loads and concatenates all monthly CSV files for a given coin into a continuous 1m DataFrame.
        Ensures chronological sorting, deduplication, and datetime index assignment.
        """
        coin_dir = self.input_dir / coin
        csv_pattern = str(coin_dir / "*.csv")
        csv_files = sorted(glob.glob(csv_pattern))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for coin '{coin}' in {coin_dir}")

        logger.info(f"[{coin.upper()}] Loading {len(csv_files)} monthly 1m CSV files...")
        start_time = time.perf_counter()

        dataframes: List[pd.DataFrame] = []
        total_raw_rows = 0

        for file_path in csv_files:
            try:
                # Fast vectorized CSV reading with explicit types
                df_month = pd.read_csv(
                    file_path,
                    header=None,
                    names=RAW_COLUMNS,
                    dtype=RAW_DTYPES,
                    usecols=list(RAW_DTYPES.keys()),
                )
                total_raw_rows += len(df_month)
                dataframes.append(df_month)
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                raise

        # Vectorized concatenation
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Drop the unused 'ignore' column
        if "ignore" in combined_df.columns:
            combined_df.drop(columns=["ignore"], inplace=True)

        # Deduplicate and sort chronologically by open_time
        initial_len = len(combined_df)
        combined_df.drop_duplicates(subset=["open_time"], keep="first", inplace=True)
        combined_df.sort_values(by="open_time", ascending=True, inplace=True)
        dedup_len = len(combined_df)

        if initial_len != dedup_len:
            logger.warning(
                f"[{coin.upper()}] Removed {initial_len - dedup_len} duplicate records."
            )

        # Convert timestamp to DatetimeIndex (Binance raw timestamps are in microseconds: 16 digits)
        # Unit detection: > 1e14 implies microseconds, > 1e11 implies milliseconds
        sample_ts = int(combined_df["open_time"].iloc[0])
        ts_unit = "us" if sample_ts > 10**14 else "ms"
        
        datetime_series = pd.to_datetime(combined_df["open_time"], unit=ts_unit, utc=True)
        combined_df["datetime"] = datetime_series
        combined_df.set_index("datetime", inplace=True)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[{coin.upper()}] Loaded & concatenated {len(combined_df):,} 1m records in {elapsed:.2f}s "
            f"({combined_df.index[0]} to {combined_df.index[-1]})."
        )

        return combined_df

    def resample_ohlcv(self, df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resamples a 1-minute continuous DataFrame into the target timeframe using
        standard financial OHLCV aggregation rules.
        """
        offset = TIMEFRAME_OFFSET_MAP[timeframe]

        # Vectorized resample with left-closed and left-labeled intervals
        resampled = (
            df_1m.resample(offset, closed="left", label="left")
            .agg(RESAMPLE_AGG_RULES)
            .dropna(subset=["open"])
        )

        # Reset datetime index to column for persistence
        resampled = resampled.reset_index()

        # Enforce strict column order and data types
        column_order = [
            "datetime",
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
        ]
        resampled = resampled[column_order]

        # Enforce integer types on trade counts and timestamps
        resampled["open_time"] = resampled["open_time"].astype("int64")
        resampled["close_time"] = resampled["close_time"].astype("int64")
        resampled["trades_count"] = resampled["trades_count"].astype("int64")

        return resampled

    def export_dataset(
        self, df: pd.DataFrame, coin: str, timeframe: str
    ) -> Tuple[Path, float, Path, float]:
        """
        Exports the processed DataFrame into both Parquet and CSV formats.
        Returns file paths and their respective sizes in MB.
        """
        target_dir = self.output_dir / coin / timeframe
        target_dir.mkdir(parents=True, exist_ok=True)

        base_filename = f"{coin}_{timeframe}_continuous"
        parquet_file = target_dir / f"{base_filename}.parquet"
        csv_file = target_dir / f"{base_filename}.csv"

        # Vectorized Parquet Export
        df.to_parquet(
            parquet_file,
            engine="pyarrow",
            compression=self.parquet_compression,
            index=False,
        )
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)

        # Vectorized CSV Export
        df.to_csv(csv_file, index=False)
        csv_size = csv_file.stat().st_size / (1024 * 1024)

        return parquet_file, parquet_size, csv_file, csv_size

    def process_coin(self, coin: str) -> List[ProcessingSummary]:
        """
        Processes a single coin: loads 1m raw data and generates all requested multi-timeframe datasets.
        """
        logger.info(f"=== Processing Coin: {coin.upper()} ===")
        df_1m = self.load_and_concatenate_raw(coin)

        summaries: List[ProcessingSummary] = []

        for tf in self.timeframes:
            tf_start = time.perf_counter()
            logger.info(f"[{coin.upper()}] Resampling to {tf}...")
            
            df_resampled = self.resample_ohlcv(df_1m, tf)

            parquet_path, parquet_size, csv_path, csv_size = self.export_dataset(
                df_resampled, coin, tf
            )
            tf_elapsed = time.perf_counter() - tf_start

            start_str = str(df_resampled["datetime"].iloc[0])
            end_str = str(df_resampled["datetime"].iloc[-1])

            summary = ProcessingSummary(
                coin=coin,
                timeframe=tf,
                row_count=len(df_resampled),
                start_date=start_str,
                end_date=end_str,
                parquet_path=parquet_path,
                parquet_size_mb=parquet_size,
                csv_path=csv_path,
                csv_size_mb=csv_size,
                elapsed_seconds=tf_elapsed,
            )
            summaries.append(summary)

            logger.info(
                f"[{coin.upper()}] {tf} generated: {len(df_resampled):,} rows "
                f"| Parquet: {parquet_size:.2f} MB | CSV: {csv_size:.2f} MB "
                f"in {tf_elapsed:.2f}s"
            )

        return summaries

    def run(self, selected_coins: Optional[Sequence[str]] = None) -> List[ProcessingSummary]:
        """
        Executes the data processing pipeline across all selected coins.
        """
        total_start = time.perf_counter()
        available_coins = self.discover_coins()

        coins_to_process = (
            [c.lower() for c in selected_coins if c.lower() in available_coins]
            if selected_coins
            else available_coins
        )

        if not coins_to_process:
            logger.error(
                f"No matching coins found to process. Available: {available_coins}"
            )
            return []

        logger.info(
            f"Starting dataset processing for {len(coins_to_process)} coins: {coins_to_process} "
            f"into timeframes: {self.timeframes}"
        )

        all_summaries: List[ProcessingSummary] = []
        for coin in coins_to_process:
            coin_summaries = self.process_coin(coin)
            all_summaries.extend(coin_summaries)

        total_elapsed = time.perf_counter() - total_start
        logger.info(
            f"=== Completed processing {len(all_summaries)} datasets in {total_elapsed:.2f}s ==="
        )

        return all_summaries


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vectorized OHLCV Multi-Timeframe Dataset Generator"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/csv",
        help="Path to raw 1-minute CSV directory (default: dataset/csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/processed",
        help="Path to save processed datasets (default: dataset/processed)",
    )
    parser.add_argument(
        "--coins",
        type=str,
        nargs="+",
        default=None,
        help="Specific coins to process (e.g. btc eth sol). Default: all available",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["5m", "15m", "30m"],
        help="Timeframes to resample (default: 5m 15m 30m)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        help="Parquet compression algorithm (default: snappy)",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    
    processor = DatasetProcessor(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        timeframes=args.timeframes,
        parquet_compression=None if args.compression == "none" else args.compression,
    )

    summaries = processor.run(selected_coins=args.coins)

    # Print summary table
    print("\n" + "=" * 105)
    print(
        f"{'Coin':<6} | {'Timeframe':<10} | {'Rows':<10} | {'Start Date':<20} | {'End Date':<20} | {'Parquet (MB)':<12} | {'CSV (MB)':<10}"
    )
    print("-" * 105)
    for s in summaries:
        print(
            f"{s.coin.upper():<6} | {s.timeframe:<10} | {s.row_count:<10,} | {s.start_date[:19]:<20} | {s.end_date[:19]:<20} | {s.parquet_size_mb:<12.2f} | {s.csv_size_mb:<10.2f}"
        )
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
