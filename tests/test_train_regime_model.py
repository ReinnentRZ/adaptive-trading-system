"""
Unit tests for Unsupervised Market Regime Model and Backtesting Pipeline.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_engine import run_regime_backtest
from scripts.train_regime_model import (
    MicrostructureFeatureExtractor,
    RegimeModelTrainer,
)


class TestTrainRegimeModel(unittest.TestCase):
    """Test suite for RegimeModelTrainer and RegimeBacktester."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "processed"
        self.models_dir = Path(self.test_dir) / "models"
        self.logs_dir = Path(self.test_dir) / "logs"
        self.coin = "testbtc"
        self.timeframe = "15m"

        coin_tf_dir = self.input_dir / self.coin / self.timeframe
        coin_tf_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic dataset (500 candles)
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2026-06-01", periods=n, freq="15min", tz="UTC")
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.uniform(0.5, 2.0, n)
        low = close - np.random.uniform(0.5, 2.0, n)
        open_ = (high + low) / 2.0
        volume = np.random.uniform(100.0, 1000.0, n)

        data = {
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        df = pd.DataFrame(data)
        cont_path = coin_tf_dir / f"{self.coin}_{self.timeframe}_continuous.parquet"
        df.to_parquet(cont_path, engine="pyarrow")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_feature_extraction(self) -> None:
        df = pd.DataFrame({
            "high": [10.0, 11.0, 12.0] * 30,
            "low": [9.0, 9.5, 10.0] * 30,
            "close": [9.5, 10.5, 11.5] * 30,
            "volume": [100.0, 150.0, 200.0] * 30,
        })
        extracted = MicrostructureFeatureExtractor.extract_features(df)
        self.assertIn("log_return", extracted.columns)
        self.assertIn("volatility", extracted.columns)
        self.assertIn("volume_intensity", extracted.columns)

    def test_training_and_backtest(self) -> None:
        trainer = RegimeModelTrainer(
            coin=self.coin,
            timeframe=self.timeframe,
            n_components=3,
            train_ratio=0.70,
            processed_dir=self.input_dir,
            models_dir=self.models_dir,
            random_state=42,
        )
        res = trainer.train_and_profile()
        self.assertEqual(len(res.state_profiles), 3)
        self.assertTrue(res.model_path.exists())
        self.assertTrue(res.metadata_path.exists())

        # Test Regime Backtester
        backtest_res, exit_reasons = run_regime_backtest(
            coin=self.coin,
            timeframe=self.timeframe,
            initial_capital=100.0,
            trade_allocation=20.0,
            processed_dir=self.input_dir,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
        )
        self.assertEqual(backtest_res.coin, self.coin.upper())
        self.assertEqual(backtest_res.timeframe, self.timeframe)
        self.assertTrue(backtest_res.trades_log_path.exists())


if __name__ == "__main__":
    unittest.main()
