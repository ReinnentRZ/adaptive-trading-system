"""
Unit tests for the feature extraction and ground-truth labeling pipeline.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_features import (
    FeatureExtractor,
    FeaturePipeline,
    FINAL_FEATURE_COLUMNS,
    _compute_triple_barrier_labels_parallel,
    _compute_lorentzian_signals_parallel,
)


class TestBuildFeaturesPipeline(unittest.TestCase):
    """Test suite for feature generation and dynamic triple-barrier labeling."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "processed"
        self.output_dir = Path(self.test_dir) / "features"
        self.coin = "testcoin"
        self.timeframe = "5m"

        coin_tf_dir = self.input_dir / self.coin / self.timeframe
        coin_tf_dir.mkdir(parents=True, exist_ok=True)

        # Generate 200 synthetic candles
        np.random.seed(42)
        n = 200
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5) + 0.1
        low = close - np.abs(np.random.randn(n) * 0.5) - 0.1
        open_ = (high + low) / 2.0
        volume = np.random.rand(n) * 100.0

        dates = pd.date_range("2025-06-01", periods=n, freq="5min", tz="UTC")
        self.df_synthetic = pd.DataFrame({
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        self.parquet_path = coin_tf_dir / f"{self.coin}_{self.timeframe}_continuous.parquet"
        self.df_synthetic.to_parquet(self.parquet_path, engine="pyarrow")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_feature_extractor_columns(self) -> None:
        df_copy = self.df_synthetic.copy()
        df_features = FeatureExtractor.extract_indicators(df_copy)

        expected_new_cols = [
            "rsi", "cci", "adx", "wt1", "wt2", "wt_diff",
            "atr", "volatility_ratio", "normalized_atr"
        ]
        for col in expected_new_cols:
            self.assertIn(col, df_features.columns)
            # Ensure calculations produce numeric values after warmup
            self.assertFalse(np.isnan(df_features[col].iloc[-1]))

    def test_triple_barrier_labeler(self) -> None:
        # Construct deterministic scenario
        high = np.array([100.0, 105.0, 101.0, 100.0])
        low = np.array([99.0, 99.5, 95.0, 99.0])
        close = np.array([100.0, 101.0, 96.0, 99.0])
        atr = np.array([1.0, 1.0, 1.0, 1.0])

        # Bar 0: TP = 100 + 4.5*1 = 104.5, SL = 100 - 1.5*1 = 98.5
        # Bar 1 has high=105.0 (>= 104.5) and low=99.5 (> 98.5) -> Label 1
        labels = _compute_triple_barrier_labels_parallel(
            high=high,
            low=low,
            close=close,
            atr=atr,
            horizon=2,
            tp_multiplier=4.5,
            sl_multiplier=1.5,
        )
        self.assertEqual(labels[0], 1)

    def test_full_feature_pipeline(self) -> None:
        pipeline = FeaturePipeline(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            timeframes=[self.timeframe],
            horizon=5,
            tp_multiplier=4.5,
            sl_multiplier=1.5,
        )
        summaries = pipeline.run(selected_coins=[self.coin])
        self.assertEqual(len(summaries), 1)

        summary = summaries[0]
        self.assertTrue(summary.output_path.exists())
        self.assertGreater(summary.valid_output_rows, 0)

        df_out = pd.read_parquet(summary.output_path)
        self.assertTrue(set(FINAL_FEATURE_COLUMNS).issubset(set(df_out.columns)))
        self.assertFalse(df_out.isna().any().any())
        self.assertTrue(set(df_out["target_label"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
