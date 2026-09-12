"""
Unit tests for LightGBM Dedication Model Trainer.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.train_dedication_model import (
    DedicationModelTrainer,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TimeSeriesSplitter,
)


class TestDedicationTrainer(unittest.TestCase):
    """Test suite for TimeSeriesSplitter and DedicationModelTrainer."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "features"
        self.models_dir = Path(self.test_dir) / "models"
        self.coin = "testcoin"
        self.timeframe = "5m"

        coin_tf_dir = self.input_dir / self.coin / self.timeframe
        coin_tf_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic feature dataset (500 rows)
        np.random.seed(42)
        n = 500
        data = {col: np.random.randn(n) for col in FEATURE_COLUMNS}
        data[TARGET_COLUMN] = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
        self.df_synthetic = pd.DataFrame(data)

        self.parquet_path = coin_tf_dir / f"{self.coin}_{self.timeframe}_features.parquet"
        self.df_synthetic.to_parquet(self.parquet_path, engine="pyarrow")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_time_series_splitter_purge_gap(self) -> None:
        n = 500
        purge_buffer = 12
        df_train, df_val, df_test = TimeSeriesSplitter.split(
            self.df_synthetic,
            train_ratio=0.70,
            val_ratio=0.15,
            purge_buffer=purge_buffer,
        )

        self.assertEqual(len(df_train), 350)
        # val_start = 350 + 12 = 362, val_end = 362 + 75 = 437
        self.assertEqual(len(df_val), 75)
        # test_start = 437 + 12 = 449, test_end = 500
        self.assertEqual(len(df_test), 51)

    def test_train_and_evaluate_pipeline(self) -> None:
        trainer = DedicationModelTrainer(
            coin=self.coin,
            timeframe=self.timeframe,
            input_dir=self.input_dir,
            models_dir=self.models_dir,
            purge_buffer=12,
        )
        metrics = trainer.train_and_evaluate()

        self.assertTrue(metrics.model_path.exists())
        self.assertTrue(metrics.metadata_path.exists())
        self.assertGreater(metrics.roc_auc, 0.0)
        self.assertGreater(len(metrics.threshold_results), 0)

        # Test model reload
        loaded_model = joblib.load(metrics.model_path)
        sample_X = np.random.randn(5, len(FEATURE_COLUMNS))
        preds = loaded_model.predict_proba(sample_X)
        self.assertEqual(preds.shape, (5, 2))

        # Test metadata content
        with open(metrics.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.assertEqual(metadata["coin"], self.coin)
            self.assertEqual(metadata["timeframe"], self.timeframe)
            self.assertEqual(metadata["feature_columns"], FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
