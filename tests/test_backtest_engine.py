"""
Unit tests for the Event-Driven Backtesting Engine.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from scripts.backtest_engine import EventDrivenBacktester, FEATURE_COLUMNS

TARGET_COLUMN = "target_label"


class TestBacktestEngine(unittest.TestCase):
    """Test suite for EventDrivenBacktester."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "features"
        self.models_dir = Path(self.test_dir) / "models"
        self.logs_dir = Path(self.test_dir) / "logs"
        self.coin = "testcoin"
        self.timeframe = "5m"

        coin_tf_dir = self.input_dir / self.coin / self.timeframe
        coin_tf_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic feature dataset (600 candles)
        np.random.seed(42)
        n = 600
        dates = pd.date_range("2026-07-01", periods=n, freq="5min", tz="UTC")
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 1.0
        low = close - 0.5
        open_ = (high + low) / 2.0
        volume = np.ones(n) * 10.0

        data = {
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "rsi": np.random.uniform(30, 70, n),
            "cci": np.random.uniform(-100, 100, n),
            "adx": np.random.uniform(15, 40, n),
            "wt_diff": np.random.randn(n),
            "atr": np.ones(n) * 1.5,
            "volatility_ratio": np.ones(n),
            "normalized_atr": np.ones(n) * 0.015,
            "lorentzian_signal": np.random.choice([-1.0, 0.0, 1.0], n),
            TARGET_COLUMN: np.random.choice([0, 1], n),
        }
        df_feat = pd.DataFrame(data)
        cont_path = coin_tf_dir / f"{self.coin}_{self.timeframe}_continuous.parquet"
        df_feat.to_parquet(cont_path, engine="pyarrow")

        # Create dummy trained model
        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.zeros((10, len(FEATURE_COLUMNS))), np.array([0, 1] * 5))
        model_path = self.models_dir / f"{self.coin}_{self.timeframe}_dedication_lgbm.joblib"
        joblib.dump(dummy, model_path)

        meta_path = self.models_dir / f"{self.coin}_{self.timeframe}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({"recommended_threshold": 0.50}, f)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_backtester_execution(self) -> None:
        engine = EventDrivenBacktester(
            coin=self.coin,
            timeframe=self.timeframe,
            initial_capital=100.0,
            trade_allocation=25.0,
            threshold=0.50,
            processed_dir=self.input_dir,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
        )

        res = engine.run()
        self.assertEqual(res.coin, self.coin.upper())
        self.assertEqual(res.timeframe, self.timeframe)
        self.assertTrue(res.trades_log_path.exists())
        self.assertTrue(res.daily_log_path.exists())

        # Check daily summary CSV
        df_daily = pd.read_csv(res.daily_log_path)
        self.assertFalse(df_daily.empty)
        self.assertIn("ending_balance", df_daily.columns)
        self.assertIn("daily_return_pct", df_daily.columns)


if __name__ == "__main__":
    unittest.main()
