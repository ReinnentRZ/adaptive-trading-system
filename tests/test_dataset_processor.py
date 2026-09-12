"""
Unit tests for the vectorized dataset processing pipeline.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.process_datasets import DatasetProcessor, TIMEFRAME_OFFSET_MAP


class TestDatasetProcessor(unittest.TestCase):
    """Test suite for DatasetProcessor resampling and export capabilities."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "csv"
        self.output_dir = Path(self.test_dir) / "processed"
        self.coin = "testcoin"

        coin_csv_dir = self.input_dir / self.coin
        coin_csv_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic 1m Binance-like candles (60 minutes = 60 rows)
        # 2025-06-01 00:00:00 UTC start
        start_us = 1748736000000000
        step_us = 60 * 1000 * 1000

        rows = []
        for i in range(60):
            open_t = start_us + i * step_us
            close_t = open_t + step_us - 1
            o = 100.0 + i
            h = o + 2.0
            l = o - 1.0
            c = o + 0.5
            v = 10.0
            qv = v * c
            trades = 50
            tb_b = 5.0
            tb_q = 5.0 * c
            ignore = 0
            rows.append([open_t, o, h, l, c, v, close_t, qv, trades, tb_b, tb_q, ignore])

        self.df_synthetic = pd.DataFrame(rows)
        self.csv_path = coin_csv_dir / "TESTCOINUSDT-1m-2025-06.csv"
        self.df_synthetic.to_csv(self.csv_path, header=False, index=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_discover_coins(self) -> None:
        processor = DatasetProcessor(input_dir=self.input_dir, output_dir=self.output_dir)
        coins = processor.discover_coins()
        self.assertEqual(coins, [self.coin])

    def test_load_and_resample_5m(self) -> None:
        processor = DatasetProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            timeframes=["5m", "15m", "30m"],
        )
        df_1m = processor.load_and_concatenate_raw(self.coin)
        self.assertEqual(len(df_1m), 60)

        # 60 1m candles -> 12 5m candles
        df_5m = processor.resample_ohlcv(df_1m, "5m")
        self.assertEqual(len(df_5m), 12)
        # First candle open should match first 1m open
        self.assertEqual(df_5m["open"].iloc[0], 100.0)
        # First 5m candle high should be max of first 5 candles: max(102, 103, 104, 105, 106) = 106.0
        self.assertEqual(df_5m["high"].iloc[0], 106.0)
        # First 5m candle low should be min of first 5 candles: min(99, 100, 101, 102, 103) = 99.0
        self.assertEqual(df_5m["low"].iloc[0], 99.0)
        # First 5m candle close should be close of 5th candle (index 4): 104 + 0.5 = 104.5
        self.assertEqual(df_5m["close"].iloc[0], 104.5)
        # Volume should be sum of 5 candles * 10 = 50.0
        self.assertEqual(df_5m["volume"].iloc[0], 50.0)

    def test_full_pipeline_execution(self) -> None:
        processor = DatasetProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            timeframes=["5m", "15m", "30m"],
        )
        summaries = processor.run()
        self.assertEqual(len(summaries), 3)

        for s in summaries:
            self.assertTrue(s.parquet_path.exists())
            self.assertTrue(s.csv_path.exists())
            self.assertGreater(s.parquet_size_mb, 0)
            self.assertGreater(s.csv_size_mb, 0)

            # Verify Parquet and CSV row parity
            df_parquet = pd.read_parquet(s.parquet_path)
            df_csv = pd.read_csv(s.csv_path)
            self.assertEqual(len(df_parquet), s.row_count)
            self.assertEqual(len(df_csv), s.row_count)


if __name__ == "__main__":
    unittest.main()
