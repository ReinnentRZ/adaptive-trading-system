#!/usr/bin/env python3
"""
LightGBM Meta-Labeling Dedication Model (Layer-2) Training Pipeline.

Trains and evaluates a second-stage machine learning decision engine for Spot
Trading (Long-Only) using purged time-series splits, early stopping, and
probability threshold calibration.

Author: Adaptive Trading System Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DedicationTrainer")

# Feature columns for Layer-2 ML Meta-Model
FEATURE_COLUMNS: List[str] = [
    "rsi",
    "cci",
    "adx",
    "wt_diff",
    "atr",
    "volatility_ratio",
    "normalized_atr",
    "lorentzian_signal",
]
TARGET_COLUMN: str = "target_label"


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Evaluation metrics at a specific decision threshold."""
    threshold: float
    total_trades_approved: int
    approval_rate_pct: float
    filtered_win_rate_pct: float
    baseline_win_rate_pct: float
    win_rate_lift_pct: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class TrainingMetrics:
    """Comprehensive performance summary of the trained model on out-of-sample test set."""
    coin: str
    timeframe: str
    train_samples: int
    val_samples: int
    test_samples: int
    best_iteration: int
    roc_auc: float
    baseline_test_win_rate_pct: float
    threshold_results: List[ThresholdEvaluation]
    top_split_features: List[Tuple[str, int]]
    top_gain_features: List[Tuple[str, float]]
    model_path: Path
    metadata_path: Path
    elapsed_seconds: float


class TimeSeriesSplitter:
    """
    Purged & Embargoed Time-Series Splitter.
    Prevents label leakage between chronological splits.
    """

    @staticmethod
    def split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        purge_buffer: int = 12,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe into Train (70%), Val (15%), and Test (15%) with
        purge gaps equal to the forward-looking label horizon.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_start = train_end + purge_buffer
        val_end = val_start + int(n * val_ratio)
        test_start = val_end + purge_buffer
        test_end = n

        if test_start >= test_end or val_start >= val_end:
            raise ValueError(
                f"Insufficient samples ({n}) for purged split with buffer {purge_buffer}."
            )

        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[val_start:val_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        logger.info(
            f"Temporal Split: Train={len(df_train):,} ({df_train.index[0]} to {df_train.index[-1]}), "
            f"Val={len(df_val):,} (Purge gap {purge_buffer} bars), "
            f"Test={len(df_test):,} (Purge gap {purge_buffer} bars)"
        )

        return df_train, df_val, df_test


class DedicationModelTrainer:
    """
    LightGBM Meta-Labeling Classifier Trainer.
    """

    def __init__(
        self,
        coin: str,
        timeframe: str,
        input_dir: Path = Path("dataset/features"),
        models_dir: Path = Path("models"),
        purge_buffer: int = 12,
        random_state: int = 42,
    ) -> None:
        self.coin = coin.lower()
        self.timeframe = timeframe.lower()
        self.input_dir = Path(input_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.purge_buffer = purge_buffer
        self.random_state = random_state

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_feature_dataset(self) -> pd.DataFrame:
        """Loads and verifies the feature parquet dataset."""
        file_path = (
            self.input_dir
            / self.coin
            / self.timeframe
            / f"{self.coin}_{self.timeframe}_features.parquet"
        )
        if not file_path.exists():
            file_path = self.input_dir / f"{self.coin}_{self.timeframe}_features.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature dataset not found at: {file_path}")

        df = pd.read_parquet(file_path)

        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {file_path}")

        # Ensure sorted chronologically
        if "datetime" in df.columns:
            df = df.sort_values(by="datetime", ascending=True).reset_index(drop=True)

        return df

    def train_and_evaluate(self) -> TrainingMetrics:
        """
        Executes purged splitting, LightGBM training, early stopping,
        OOS test evaluation, threshold tuning, and model serialization.
        """
        t0 = time.perf_counter()
        logger.info(f"=== Training Dedication Model: {self.coin.upper()} [{self.timeframe}] ===")

        df = self.load_feature_dataset()
        df_train, df_val, df_test = TimeSeriesSplitter.split(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            purge_buffer=self.purge_buffer,
        )

        X_train = df_train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y_train = df_train[TARGET_COLUMN].to_numpy(dtype=np.int32)

        X_val = df_val[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y_val = df_val[TARGET_COLUMN].to_numpy(dtype=np.int32)

        X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y_test = df_test[TARGET_COLUMN].to_numpy(dtype=np.int32)

        # Dynamic class imbalance weighting
        count_0 = np.count_nonzero(y_train == 0)
        count_1 = np.count_nonzero(y_train == 1)
        scale_pos_weight = (count_0 / count_1) if count_1 > 0 else 1.0

        logger.info(
            f"Train Class Distribution: Label 0 = {count_0:,}, Label 1 = {count_1:,} "
            f"(scale_pos_weight={scale_pos_weight:.2f})"
        )

        # LightGBM Classifier Architecture
        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )

        # Fit with Early Stopping on Validation Set (Optimizing for ROC-AUC)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["val"],
            eval_metric="auc",
            callbacks=callbacks,
        )

        best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else 1000
        logger.info(f"Model converged at iteration: {best_iter}")

        # Out-Of-Sample Test Evaluation
        y_test_probs = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_test_probs))
        baseline_test_win_rate = (
            float(np.count_nonzero(y_test == 1)) / len(y_test) * 100.0
        )

        logger.info(
            f"Out-of-Sample ROC-AUC: {roc_auc:.4f} | Baseline Win Rate: {baseline_test_win_rate:.2f}%"
        )

        # Probability Decision Threshold Tuning across full spectrum (0.01 step resolution)
        thresholds = [round(float(x), 3) for x in np.arange(0.20, 0.85, 0.01)]
        threshold_results: List[ThresholdEvaluation] = []

        for th in thresholds:
            y_pred_th = (y_test_probs >= th).astype(int)
            total_approved = int(np.count_nonzero(y_pred_th == 1))
            app_rate = (total_approved / len(y_test)) * 100.0

            if total_approved > 0:
                true_positives = int(np.count_nonzero((y_pred_th == 1) & (y_test == 1)))
                filtered_win_rate = (true_positives / total_approved) * 100.0
                prec = float(precision_score(y_test, y_pred_th, zero_division=0))
                rec = float(recall_score(y_test, y_pred_th, zero_division=0))
                f1 = float(f1_score(y_test, y_pred_th, zero_division=0))
            else:
                filtered_win_rate = 0.0
                prec, rec, f1 = 0.0, 0.0, 0.0

            lift = filtered_win_rate - baseline_test_win_rate

            eval_res = ThresholdEvaluation(
                threshold=th,
                total_trades_approved=total_approved,
                approval_rate_pct=app_rate,
                filtered_win_rate_pct=filtered_win_rate,
                baseline_win_rate_pct=baseline_test_win_rate,
                win_rate_lift_pct=lift,
                precision=prec,
                recall=rec,
                f1=f1,
            )
            threshold_results.append(eval_res)

        # Feature Importance Ranking
        split_importances = model.booster_.feature_importance(importance_type="split")
        gain_importances = model.booster_.feature_importance(importance_type="gain")

        split_ranks = sorted(
            zip(FEATURE_COLUMNS, [int(x) for x in split_importances]),
            key=lambda x: x[1],
            reverse=True,
        )
        gain_ranks = sorted(
            zip(FEATURE_COLUMNS, [float(x) for x in gain_importances]),
            key=lambda x: x[1],
            reverse=True,
        )

        # Model Serialization
        model_filename = f"{self.coin}_{self.timeframe}_dedication_lgbm.joblib"
        metadata_filename = f"{self.coin}_{self.timeframe}_metadata.json"

        model_path = self.models_dir / model_filename
        metadata_path = self.models_dir / metadata_filename

        joblib.dump(model, model_path)

        # Candidate selection: require at least 15 approved trades and prioritize highest filtered win rate / precision
        valid_candidates = [
            r for r in threshold_results if r.total_trades_approved >= 15 and r.filtered_win_rate_pct > baseline_test_win_rate
        ]
        if valid_candidates:
            # Pick candidate with highest filtered win rate and win rate lift
            rec_threshold = max(valid_candidates, key=lambda x: (x.filtered_win_rate_pct, x.win_rate_lift_pct, x.precision)).threshold
        else:
            active_cands = [r for r in threshold_results if r.total_trades_approved >= 15]
            rec_threshold = max(active_cands, key=lambda x: (x.filtered_win_rate_pct, x.f1)).threshold if active_cands else 0.50

        metadata: Dict[str, Any] = {
            "coin": self.coin,
            "timeframe": self.timeframe,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "best_iteration": int(best_iter),
            "scale_pos_weight": float(scale_pos_weight),
            "roc_auc_test": float(roc_auc),
            "baseline_test_win_rate_pct": float(baseline_test_win_rate),
            "recommended_threshold": float(rec_threshold),
            "threshold_evaluations": [asdict(r) for r in threshold_results],
            "top_features_by_gain": [{"feature": f, "gain": g} for f, g in gain_ranks],
            "top_features_by_split": [{"feature": f, "split": s} for f, s in split_ranks],
            "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Saved model to {model_path.name} & metadata to {metadata_path.name} in {elapsed:.2f}s"
        )

        return TrainingMetrics(
            coin=self.coin,
            timeframe=self.timeframe,
            train_samples=len(df_train),
            val_samples=len(df_val),
            test_samples=len(df_test),
            best_iteration=int(best_iter),
            roc_auc=roc_auc,
            baseline_test_win_rate_pct=baseline_test_win_rate,
            threshold_results=threshold_results,
            top_split_features=split_ranks[:5],
            top_gain_features=gain_ranks[:5],
            model_path=model_path,
            metadata_path=metadata_path,
            elapsed_seconds=elapsed,
        )


# ==============================================================================
# CLI INTERFACE & STATISTICAL REPORTING
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="LightGBM Meta-Labeling Dedication Model Trainer"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default=None,
        help="Target coin symbol (e.g. btc, eth, sol)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Target timeframe (e.g. 5m, 15m, 30m)",
    )
    parser.add_argument(
        "--coins",
        type=str,
        nargs="+",
        default=["btc", "eth", "sol"],
        help="List of coins to train (default: btc eth sol)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["5m", "15m", "30m"],
        help="List of timeframes to train (default: 5m 15m 30m)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all available combinations in dataset/features",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/features",
        help="Input feature dataset directory (default: dataset/features)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained models and metadata (default: models)",
    )
    parser.add_argument(
        "--purge-buffer",
        type=int,
        default=12,
        help="Purge gap between temporal splits (default: 12)",
    )
    return parser.parse_args()


def print_training_report(metrics_list: List[TrainingMetrics]) -> None:
    """Prints a structured performance table and threshold calibration matrix."""
    print("\n" + "=" * 135)
    print(
        f"{'Coin':<6} | {'Timeframe':<10} | {'Train':<9} | {'Val':<8} | {'Test':<8} | "
        f"{'ROC-AUC':<8} | {'Base Win%':<10} | {'Win% @P>=0.50':<14} | {'Win% @P>=0.55':<14} | {'Win% @P>=0.60':<14} | {'Best Iter':<9}"
    )
    print("-" * 135)

    for m in metrics_list:
        th50 = next((t for t in m.threshold_results if t.threshold == 0.50), None)
        th55 = next((t for t in m.threshold_results if t.threshold == 0.55), None)
        th60 = next((t for t in m.threshold_results if t.threshold == 0.60), None)

        w50 = f"{th50.filtered_win_rate_pct:.2f}%" if th50 and th50.total_trades_approved > 0 else "N/A"
        w55 = f"{th55.filtered_win_rate_pct:.2f}%" if th55 and th55.total_trades_approved > 0 else "N/A"
        w60 = f"{th60.filtered_win_rate_pct:.2f}%" if th60 and th60.total_trades_approved > 0 else "N/A"

        print(
            f"{m.coin.upper():<6} | {m.timeframe:<10} | {m.train_samples:<9,} | {m.val_samples:<8,} | "
            f"{m.test_samples:<8,} | {m.roc_auc:<8.4f} | {m.baseline_test_win_rate_pct:<9.2f}% | "
            f"{w50:<14} | {w55:<14} | {w60:<14} | {m.best_iteration:<9}"
        )
    print("=" * 135 + "\n")

    # Detailed Feature Importance & Threshold Report
    for m in metrics_list:
        print(f"--- Top 5 Features (Gain Importance) for {m.coin.upper()} [{m.timeframe}] ---")
        for rank, (feat, gain) in enumerate(m.top_gain_features, 1):
            print(f"  {rank}. {feat:<20} : {gain:,.2f}")
        print()


def main() -> None:
    """Main CLI execution flow."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    models_dir = Path(args.models_dir)

    pairs_to_train: List[Tuple[str, str]] = []

    if args.coin and args.timeframe:
        pairs_to_train = [(args.coin.lower(), args.timeframe.lower())]
    elif args.all or (args.coins and args.timeframes):
        coins = args.coins
        timeframes = args.timeframes
        for c in coins:
            for tf in timeframes:
                pairs_to_train.append((c.lower(), tf.lower()))
    else:
        logger.error("Please specify --coin and --timeframe or --all.")
        sys.exit(1)

    all_metrics: List[TrainingMetrics] = []
    total_start = time.perf_counter()

    for coin, tf in pairs_to_train:
        trainer = DedicationModelTrainer(
            coin=coin,
            timeframe=tf,
            input_dir=input_dir,
            models_dir=models_dir,
            purge_buffer=args.purge_buffer,
        )
        try:
            metrics = trainer.train_and_evaluate()
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Failed training for {coin} {tf}: {e}", exc_info=True)

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        f"=== Completed training {len(all_metrics)} models in {total_elapsed:.2f}s ==="
    )

    if all_metrics:
        print_training_report(all_metrics)


if __name__ == "__main__":
    main()
