#!/usr/bin/env python3
"""
Vectorized Feature Generation & Dynamic Triple-Barrier Ground-Truth Labeling Pipeline.

Designed for Spot Trading (Long-Only) Meta-Labeling Dedication Systems.
Processes continuous multi-timeframe Parquet datasets from `dataset/processed/`
into ML-ready training datasets in `dataset/features/`.

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

# Try importing TA-Lib, raise clear error if missing
try:
    import talib
except ImportError as err:
    raise ImportError(
        "TA-Lib is required for high-speed indicator calculation. "
        "Please ensure TA-Lib C-library and Python bindings are installed."
    ) from err

# Numba for high-performance JIT and parallel vectorization
from numba import njit, prange

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FeatureBuilder")

# Final target output schema
FINAL_FEATURE_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "cci",
    "adx",
    "wt_diff",
    "atr",
    "volatility_ratio",
    "normalized_atr",
    "lorentzian_signal",
    "target_label",
]


@dataclass(frozen=True)
class FeatureSummary:
    """Statistical summary of processed features and ground truth labels."""
    coin: str
    timeframe: str
    total_raw_rows: int
    valid_output_rows: int
    label_0_count: int
    label_1_count: int
    imbalance_ratio: float
    win_rate_pct: float
    output_path: Path
    file_size_mb: float
    elapsed_seconds: float


# ==============================================================================
# NUMBA JIT ACCELERATED ENGINES (ZERO LOOKAHEAD BIAS)
# ==============================================================================

@njit(parallel=True, fastmath=True)
def _compute_lorentzian_signals_parallel(
    features: np.ndarray,      # shape (N, 4): [rsi, adx, cci, wt1]
    close_prices: np.ndarray,  # shape (N,)
    neighbors_count: int = 8,
    max_bars_back: int = 2000,
    label_horizon: int = 4,
) -> np.ndarray:
    """
    Numba-parallelized Lorentzian Nearest Neighbor classification.
    Strictly enforces zero lookahead bias: at bar t, only candidate bars
    i <= t - label_horizon whose forward outcomes are known are used.
    """
    n = len(close_prices)
    signals = np.zeros(n, dtype=np.float64)

    # Historical outcome direction labels: +1 (LONG), -1 (SHORT), 0 (NEUTRAL)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - label_horizon):
        if close_prices[i + label_horizon] > close_prices[i]:
            labels[i] = 1
        elif close_prices[i + label_horizon] < close_prices[i]:
            labels[i] = -1
        else:
            labels[i] = 0

    cutoff_idx = int(np.floor(neighbors_count * 0.75 + 0.5))

    for t in prange(n):
        cand_f0 = features[t, 0]
        cand_f1 = features[t, 1]
        cand_f2 = features[t, 2]
        cand_f3 = features[t, 3]

        if np.isnan(cand_f0) or np.isnan(cand_f1) or np.isnan(cand_f2) or np.isnan(cand_f3):
            signals[t] = 0.0
            continue

        start_idx = max(0, t - max_bars_back)
        end_idx = t - label_horizon
        if end_idx <= start_idx:
            signals[t] = 0.0
            continue

        distances_buf = np.zeros(neighbors_count + 2, dtype=np.float64)
        preds_buf = np.zeros(neighbors_count + 2, dtype=np.int8)
        last_distance = -1.0
        buf_len = 0

        for i in range(start_idx, end_idx + 1):
            if i % label_horizon == 0:
                continue

            hf0 = features[i, 0]
            hf1 = features[i, 1]
            hf2 = features[i, 2]
            hf3 = features[i, 3]

            if np.isnan(hf0) or np.isnan(hf1) or np.isnan(hf2) or np.isnan(hf3):
                continue

            # Lorentzian Distance Metric: sum(ln(1 + |a_k - b_k|))
            d = (
                np.log1p(np.abs(cand_f0 - hf0))
                + np.log1p(np.abs(cand_f1 - hf1))
                + np.log1p(np.abs(cand_f2 - hf2))
                + np.log1p(np.abs(cand_f3 - hf3))
            )

            if d < last_distance:
                continue

            last_distance = d
            distances_buf[buf_len] = d
            preds_buf[buf_len] = labels[i]
            buf_len += 1

            if buf_len > neighbors_count:
                last_distance = distances_buf[cutoff_idx]
                for k in range(buf_len - 1):
                    distances_buf[k] = distances_buf[k + 1]
                    preds_buf[k] = preds_buf[k + 1]
                buf_len -= 1

        vote_sum = 0
        for k in range(buf_len):
            vote_sum += preds_buf[k]

        if vote_sum > 0:
            signals[t] = 1.0
        elif vote_sum < 0:
            signals[t] = -1.0
        else:
            signals[t] = 0.0

    return signals


@njit(parallel=True, fastmath=True)
def _compute_triple_barrier_labels_parallel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_multiplier: float,
    sl_multiplier: float,
) -> np.ndarray:
    """
    Simulates dynamic forward triple-barrier for Spot Long-Only trades.
    - Label 1 (Successful Buy): High touches TP before Low touches SL within horizon.
    - Label 0 (Failed / Avoid): Low touches SL first OR TP not reached before horizon expiry.
    - Intra-bar conservative fail-safe: if both TP & SL breached in same candle, SL hit first (Label 0).
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)

    for t in prange(n - horizon):
        atr_t = atr[t]
        close_t = close[t]

        if np.isnan(atr_t) or atr_t <= 0:
            labels[t] = 0
            continue

        tp_price = close_t + (tp_multiplier * atr_t)
        sl_price = close_t - (sl_multiplier * atr_t)

        assigned_label = 0

        for k in range(1, horizon + 1):
            bar_idx = t + k
            h = high[bar_idx]
            l = low[bar_idx]

            hit_tp = h >= tp_price
            hit_sl = l <= sl_price

            if hit_tp and hit_sl:
                # Conservative fail-safe: assume SL triggered first
                assigned_label = 0
                break
            elif hit_tp:
                assigned_label = 1
                break
            elif hit_sl:
                assigned_label = 0
                break

        labels[t] = assigned_label

    return labels


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================

class FeatureExtractor:
    """Vectorized calculation of Layer-1 technical indicators and market regimes."""

    @staticmethod
    def extract_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes RSI, CCI, ADX, WaveTrend, ATR, Volatility Ratio, and Normalized ATR.
        Fully vectorized with zero row-by-row iteration.
        """
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)

        # 1. Momentum & Oscillators
        rsi = talib.RSI(close, timeperiod=14)
        cci = talib.CCI(high, low, close, timeperiod=20)
        adx = talib.ADX(high, low, close, timeperiod=14)

        # WaveTrend Oscillator (Channel: 10, Avg: 21, SMA: 4)
        hlc3 = (high + low + close) / 3.0
        esa = talib.EMA(hlc3, timeperiod=10)
        d = talib.EMA(np.abs(hlc3 - esa), timeperiod=10)
        divisor = 0.015 * d
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = np.where(divisor != 0, (hlc3 - esa) / divisor, 0.0)
        wt1 = talib.EMA(ci, timeperiod=21)
        wt2 = talib.SMA(wt1, timeperiod=4)
        wt_diff = wt1 - wt2

        # 2. Volatility & Market Regime
        atr = talib.ATR(high, low, close, timeperiod=14)
        atr_sma50 = talib.SMA(atr, timeperiod=50)

        with np.errstate(divide="ignore", invalid="ignore"):
            volatility_ratio = np.where(
                (atr_sma50 != 0) & (~np.isnan(atr_sma50)), atr / atr_sma50, np.nan
            )
            normalized_atr = np.where(close != 0, atr / close, np.nan)

        # Assign back into DataFrame
        df["rsi"] = rsi
        df["cci"] = cci
        df["adx"] = adx
        df["wt1"] = wt1
        df["wt2"] = wt2
        df["wt_diff"] = wt_diff
        df["atr"] = atr
        df["volatility_ratio"] = volatility_ratio
        df["normalized_atr"] = normalized_atr

        return df


class FeaturePipeline:
    """
    End-to-end dataset feature generation and ground truth labeling engine.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        timeframes: Sequence[str] = ("5m", "15m", "30m"),
        horizon: int = 12,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        lorentzian_k: int = 8,
        lorentzian_max_bars: int = 2000,
        lorentzian_label_horizon: int = 4,
        compression: str = "snappy",
    ) -> None:
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeframes = list(timeframes)
        self.horizon = horizon
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.lorentzian_k = lorentzian_k
        self.lorentzian_max_bars = lorentzian_max_bars
        self.lorentzian_label_horizon = lorentzian_label_horizon
        self.compression = compression

    def discover_coins(self) -> List[str]:
        """Discovers available coins in input processed directory."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        coins = [
            d.name for d in self.input_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(coins)

    def load_processed_parquet(self, coin: str, timeframe: str) -> pd.DataFrame:
        """
        TAHAP 1: Ingests continuous Parquet dataset and validates temporal order and schema.
        """
        file_path = (
            self.input_dir / coin / timeframe / f"{coin}_{timeframe}_continuous.parquet"
        )
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df = pd.read_parquet(file_path)

        # Validate required columns
        base_cols = ["open", "high", "low", "close", "volume"]
        for col in base_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required base column '{col}' in {file_path}")
            df[col] = df[col].astype(np.float64)

        # Validate temporal ordering
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df.sort_values(by="datetime", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
        elif "open_time" in df.columns:
            df.sort_values(by="open_time", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    def process_dataset(self, coin: str, timeframe: str) -> FeatureSummary:
        """
        Executes feature extraction, Lorentzian signal generation, and Triple Barrier labeling.
        """
        t0 = time.perf_counter()
        logger.info(f"[{coin.upper()}] Loading continuous {timeframe} parquet dataset...")

        df = self.load_processed_parquet(coin, timeframe)
        total_raw_rows = len(df)

        # TAHAP 2: Feature Extraction
        logger.info(f"[{coin.upper()}] [{timeframe}] Extracting Layer-1 technical indicators...")
        df = FeatureExtractor.extract_indicators(df)

        # Calculate Lorentzian Classification signals via Numba JIT
        logger.info(f"[{coin.upper()}] [{timeframe}] Computing Lorentzian classification signals...")
        lorentz_features = np.column_stack((
            df["rsi"].to_numpy(dtype=np.float64),
            df["adx"].to_numpy(dtype=np.float64),
            df["cci"].to_numpy(dtype=np.float64),
            df["wt1"].to_numpy(dtype=np.float64),
        ))
        close_np = df["close"].to_numpy(dtype=np.float64)

        lorentzian_signals = _compute_lorentzian_signals_parallel(
            features=lorentz_features,
            close_prices=close_np,
            neighbors_count=self.lorentzian_k,
            max_bars_back=self.lorentzian_max_bars,
            label_horizon=self.lorentzian_label_horizon,
        )
        df["lorentzian_signal"] = lorentzian_signals

        # TAHAP 3: Dynamic Triple-Barrier Ground-Truth Labeling
        logger.info(
            f"[{coin.upper()}] [{timeframe}] Generating ground-truth labels "
            f"(TP={self.tp_multiplier}x ATR, SL={self.sl_multiplier}x ATR, Horizon={self.horizon})..."
        )
        high_np = df["high"].to_numpy(dtype=np.float64)
        low_np = df["low"].to_numpy(dtype=np.float64)
        atr_np = df["atr"].to_numpy(dtype=np.float64)

        labels = _compute_triple_barrier_labels_parallel(
            high=high_np,
            low=low_np,
            close=close_np,
            atr=atr_np,
            horizon=self.horizon,
            tp_multiplier=self.tp_multiplier,
            sl_multiplier=self.sl_multiplier,
        )
        df["target_label"] = labels

        # Data Cleaning:
        # 1. Drop the final `horizon` rows (uncompleted forward horizon)
        df = df.iloc[: -self.horizon].copy()

        # 2. Drop initial warmup NaN rows
        df.dropna(subset=FINAL_FEATURE_COLUMNS, inplace=True)

        # 3. Pure Meta-Labeling Event Filtering: Keep ONLY rows with Primary Buy Trigger Events
        # Trigger condition: lorentzian_signal == 1 OR (rsi < 40 AND wt_diff > 0)
        primary_buy_events = (df["lorentzian_signal"] == 1.0) | (
            (df["rsi"] < 40.0) & (df["wt_diff"] > 0.0)
        )
        logger.info(
            f"[{coin.upper()}] [{timeframe}] Filtering primary events: "
            f"{int(np.count_nonzero(primary_buy_events)):,} of {len(df):,} bars selected"
        )
        df = df[primary_buy_events].copy().reset_index(drop=True)

        # Ensure correct datatypes and exact final column ordering (including datetime for temporal tracking)
        df["target_label"] = df["target_label"].astype(np.int32)
        export_cols = (["datetime"] if "datetime" in df.columns else []) + FINAL_FEATURE_COLUMNS
        final_df = df[export_cols].copy()

        # TAHAP 4: Persistence
        out_dir = self.output_dir / coin / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{coin}_{timeframe}_features.parquet"
        flat_out_file = self.output_dir / f"{coin}_{timeframe}_features.parquet"

        final_df.to_parquet(
            out_file,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )
        final_df.to_parquet(
            flat_out_file,
            engine="pyarrow",
            compression=self.compression,
            index=False,
        )
        file_size_mb = out_file.stat().st_size / (1024 * 1024)

        # Statistics
        valid_rows = len(final_df)
        label_0_cnt = int(np.count_nonzero(final_df["target_label"] == 0))
        label_1_cnt = int(np.count_nonzero(final_df["target_label"] == 1))
        imbalance_ratio = (label_0_cnt / label_1_cnt) if label_1_cnt > 0 else float("inf")
        win_rate_pct = (label_1_cnt / valid_rows * 100.0) if valid_rows > 0 else 0.0
        elapsed = time.perf_counter() - t0

        logger.info(
            f"[{coin.upper()}] [{timeframe}] Saved {valid_rows:,} labeled rows to {out_file.name} "
            f"({file_size_mb:.2f} MB) | Win Rate: {win_rate_pct:.2f}% | Imbalance: {imbalance_ratio:.2f}:1 "
            f"in {elapsed:.2f}s"
        )

        return FeatureSummary(
            coin=coin,
            timeframe=timeframe,
            total_raw_rows=total_raw_rows,
            valid_output_rows=valid_rows,
            label_0_count=label_0_cnt,
            label_1_count=label_1_cnt,
            imbalance_ratio=imbalance_ratio,
            win_rate_pct=win_rate_pct,
            output_path=out_file,
            file_size_mb=file_size_mb,
            elapsed_seconds=elapsed,
        )

    def run(self, selected_coins: Optional[Sequence[str]] = None) -> List[FeatureSummary]:
        """Runs the pipeline across all selected coins and timeframes."""
        available_coins = self.discover_coins()
        coins_to_process = (
            [c.lower() for c in selected_coins if c.lower() in available_coins]
            if selected_coins
            else available_coins
        )

        if not coins_to_process:
            logger.error(f"No coins to process. Available: {available_coins}")
            return []

        logger.info(
            f"Starting feature & label generation for {len(coins_to_process)} coins: {coins_to_process} "
            f"| Timeframes: {self.timeframes} | Horizon: {self.horizon} | TP: {self.tp_multiplier}x | SL: {self.sl_multiplier}x"
        )

        total_t0 = time.perf_counter()
        summaries: List[FeatureSummary] = []

        for coin in coins_to_process:
            for tf in self.timeframes:
                summary = self.process_dataset(coin, tf)
                summaries.append(summary)

        total_elapsed = time.perf_counter() - total_t0
        logger.info(
            f"=== Finished feature pipeline for {len(summaries)} datasets in {total_elapsed:.2f}s ==="
        )

        return summaries


# ==============================================================================
# CLI INTERFACE & REPORTING
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vectorized Feature Engineering & Ground-Truth Labeling Pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/processed",
        help="Path to continuous processed parquet datasets (default: dataset/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/features",
        help="Path to output feature datasets (default: dataset/features)",
    )
    parser.add_argument(
        "--coins",
        type=str,
        nargs="+",
        default=None,
        help="Coins to process (e.g. btc eth sol). Default: all available",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["5m", "15m", "30m"],
        help="Timeframes to process (default: 5m 15m 30m)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward evaluation horizon in candles (default: 12)",
    )
    parser.add_argument(
        "--tp_mult",
        "--tp-mult",
        type=float,
        default=2.0,
        help="Take profit ATR multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--sl_mult",
        "--sl-mult",
        type=float,
        default=1.0,
        help="Stop loss ATR multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "zstd", "gzip", "none"],
        help="Parquet compression algorithm (default: snappy)",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution flow."""
    args = parse_args()

    pipeline = FeaturePipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        timeframes=args.timeframes,
        horizon=args.horizon,
        tp_multiplier=args.tp_mult,
        sl_multiplier=args.sl_mult,
        compression=None if args.compression == "none" else args.compression,
    )

    summaries = pipeline.run(selected_coins=args.coins)

    # Statistical Reporting Table
    print("\n" + "=" * 115)
    print(
        f"{'Coin':<6} | {'Timeframe':<10} | {'Total Rows':<11} | {'Valid Rows':<11} | "
        f"{'Label 0':<10} | {'Label 1':<10} | {'Win Rate (%)':<13} | {'Imbalance':<11} | {'Size (MB)':<9}"
    )
    print("-" * 115)
    for s in summaries:
        print(
            f"{s.coin.upper():<6} | {s.timeframe:<10} | {s.total_raw_rows:<11,} | {s.valid_output_rows:<11,} | "
            f"{s.label_0_count:<10,} | {s.label_1_count:<10,} | {s.win_rate_pct:<13.2f} | "
            f"{s.imbalance_ratio:<8.2f}:1 | {s.file_size_mb:<9.2f}"
        )
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
