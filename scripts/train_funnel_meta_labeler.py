#!/usr/bin/env python3
"""
Dedicated Meta-Labeler (Layer-2 AI) Training Pipeline for Market Regime Funnel Strategy.

Specifically designed for BTC/USDT 1H:
  1. Causal Extraction of Primary Quantitative Candidate Signals (Pilar 1 & 2: HMM + Kalman).
  2. Ground Truth Triple Barrier / Adaptive Bracket Simulation (Label y=1 if TP1/TP2 covers roundtrip fees, y=0 if SL).
  3. Purged Time-Series Split training and Probability Calibration (CalibratedClassifierCV).
  4. Saves calibrated artifact to `models/btc_1h_funnel_metalabeler.joblib`.

Authors: Adaptive Trading System Quant Team
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
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Project path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.strategy import AdaptiveBracketParams, RegimeFunnelConfig
from src.strategies.meta_labeler import (
    DEFAULT_FEATURE_COLUMNS,
    MULTI_ASSET_FEATURE_COLUMNS,
    MetaLabelingGate,
    PurgedTimeSeriesSplit,
)
from src.strategies.regime_funnel import RegimeFunnelStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FunnelMetaTrainer")


class FunnelMetaLabelingTrainer:
    """
    End-to-end Trainer & Calibrator for the 1H Corong 3 Pilar Meta-Labeler.
    Supports single-asset or pooled multi-asset training (BTC, ETH, SOL).
    """

    def __init__(
        self,
        coin: str = "btc",
        coins: Optional[Sequence[str]] = None,
        timeframe: str = "1h",
        processed_dir: Path = Path("dataset/processed"),
        models_dir: Path = Path("models"),
        use_kalman: bool = True,
        single_shot: bool = False,
        fee_rate: float = 0.0002,  # 0.02% maker per side -> 0.04% roundtrip
        max_holding_bars: int = 12,
        purge_buffer: int = 5,
        cv_splits: int = 4,
        cv_mode: str = "purged_ts",  # "purged_ts" or "prefit"
        output_model_path: Optional[Path] = None,
        random_state: int = 42,
    ) -> None:
        self.coin = coin.lower()
        self.coins = [c.lower() for c in coins] if coins else [self.coin]
        self.is_multi_asset = len(self.coins) > 1
        self.timeframe = timeframe.lower()
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.use_kalman = use_kalman
        self.single_shot = single_shot
        self.fee_rate = fee_rate
        self.max_holding_bars = max_holding_bars
        self.purge_buffer = purge_buffer
        self.cv_splits = cv_splits
        self.cv_mode = cv_mode.lower()
        self.random_state = random_state

        if self.is_multi_asset:
            self.feature_columns = list(MULTI_ASSET_FEATURE_COLUMNS)
        else:
            self.feature_columns = list(DEFAULT_FEATURE_COLUMNS)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        if output_model_path:
            self.output_model_path = Path(output_model_path)
        elif self.is_multi_asset:
            self.output_model_path = self.models_dir / f"multi_asset_{self.timeframe}_funnel_metalabeler.joblib"
        else:
            self.output_model_path = self.models_dir / f"{self.coin}_{self.timeframe}_funnel_metalabeler.joblib"

    def _load_raw_df(self, coin: str) -> pd.DataFrame:
        """Loads continuous OHLCV dataset for a given coin."""
        data_path = self.processed_dir / coin / self.timeframe / f"{coin}_{self.timeframe}_continuous.parquet"
        if not data_path.exists():
            data_path = self.processed_dir / f"{coin}_{self.timeframe}_continuous.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Continuous dataset not found at: {data_path}")

        logger.info(f"[{coin.upper()}] Loading dataset from: {data_path}")
        df = pd.read_parquet(data_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df.sort_values(by="datetime", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def prepare_dataset_for_coin(
        self, coin: str, btc_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, RegimeFunnelStrategy, RegimeFunnelConfig]:
        """Prepares strategy indicators, HMM states, and features for a coin."""
        df = self._load_raw_df(coin)

        hmm_model_path = self.models_dir / f"{coin}_{self.timeframe}_regime_hmm.joblib"
        if not hmm_model_path.exists():
            hmm_model_path = self.models_dir / f"{coin}_{self.timeframe}_regime_model.joblib"
        if not hmm_model_path.exists():
            hmm_model_path = self.models_dir / "btc_1h_regime_hmm.joblib"

        cfg = RegimeFunnelConfig(
            hmm_model_path=str(hmm_model_path),
            use_kalman_filter=self.use_kalman,
            use_volume_filter=False,
            single_shot_per_episode=self.single_shot,
            use_meta_labeler=False,
        )
        strategy = RegimeFunnelStrategy(cfg)

        logger.info(f"[{coin.upper()}] Computing Strategy Indicators (Kalman={self.use_kalman})...")
        df_ind = strategy.compute_indicators(df)
        df_ind.dropna(subset=strategy.feature_columns + ["atr", "ema_9", "rsi_14", "ema_200"], inplace=True)
        df_ind.reset_index(drop=True, inplace=True)

        # Extract features (including cross-asset btc_return_1h and relative_strength_vs_btc) on full series before slicing
        logger.info(f"[{coin.upper()}] Extracting Microstructure & Cross-Asset Features...")
        df_features = MetaLabelingGate.extract_features(df_ind, btc_df=btc_df)
        df_features["coin"] = coin

        # Slice after 200-bar warm-up
        df_test = df_features.iloc[200:].copy().reset_index(drop=True)
        causal_states, _ = strategy.compute_causal_states(df_test)
        df_test["regime_state"] = causal_states

        return df_test, strategy, cfg

    def extract_candidates_and_labels(
        self,
        df: pd.DataFrame,
        strategy: Optional[RegimeFunnelStrategy] = None,
        config: Optional[RegimeFunnelConfig] = None,
        coin: Optional[str] = None,
        btc_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extracts causal primary signals where Pilar 1 & Pilar 2 trigger,
        and simulates Triple Barrier / Adaptive Bracket ground truth labels.
        """
        strat = strategy or getattr(self, "strategy", None)
        cfg = config or getattr(self, "config", None)
        coin_name = (coin or getattr(self, "coin", "btc")).upper()

        logger.info(f"[{coin_name}] Simulating Ground Truth Triple Barrier Outcomes for Primary Candidates...")
        records: List[Dict[str, Any]] = []
        state_age = 0
        traded_in_episode = False
        n = len(df)

        for i in range(n - self.max_holding_bars - 1):
            curr_state = int(df["regime_state"].iloc[i])
            if curr_state == strat.bullish_state_id:
                state_age += 1
            else:
                state_age = 0
                traded_in_episode = False

            sig, details = strat.evaluate_gates(
                df, i, curr_state, state_age, traded_in_episode, btc_df=btc_df
            )
            if not sig:
                continue

            if self.single_shot:
                traded_in_episode = True

            # Next-bar open entry (causal execution at t+1)
            entry_idx = i + 1
            entry_price = float(df["open"].iloc[entry_idx])
            atr_val = float(df["atr"].iloc[i])
            sig_close = float(df["close"].iloc[i])
            sig_trend = float(
                df["kalman_trend"].iloc[i]
                if (self.use_kalman and "kalman_trend" in df)
                else df["ema_200"].iloc[i]
            )
            sig_rsi = float(df["rsi_14"].iloc[i])
            sig_natr = float(df["normalized_atr"].iloc[i])
            sig_natr_med = float(
                df["normalized_atr"].iloc[max(0, i - 100) : i + 1].median()
            )

            brackets = strat.calculate_adaptive_brackets(
                entry_price=entry_price,
                atr_val=atr_val,
                current_close=sig_close,
                ema200_val=sig_trend,
                rsi_val=sig_rsi,
                natr_val=sig_natr,
                natr_median=sig_natr_med,
                params=cfg.adaptive_brackets,
            )
            tp1_price = brackets["tp1_price"]
            tp2_price = brackets["tp2_price"]
            sl_price = brackets["sl_price"]
            tp1_ratio = brackets["tp1_ratio"]
            tp2_ratio = brackets["tp2_ratio"]

            # Intra-bar simulation up to max_holding_bars
            tp1_taken = False
            curr_sl = sl_price
            curr_qty = 1.0
            realized_gross = 0.0
            realized_fee = entry_price * self.fee_rate
            trade_finished = False
            net_return = 0.0
            exit_reason = "TIME_EXPIRY"

            for h_step in range(1, self.max_holding_bars + 1):
                b_idx = entry_idx + h_step
                if b_idx >= n:
                    break
                h_b = float(df["high"].iloc[b_idx])
                l_b = float(df["low"].iloc[b_idx])

                # Stop loss worst-case check
                if l_b <= curr_sl:
                    exit_price = curr_sl
                    exit_val = curr_qty * exit_price
                    exit_fee = exit_val * self.fee_rate
                    gross = (exit_val - curr_qty * entry_price) + realized_gross
                    total_fee = realized_fee + exit_fee
                    net_return = (gross - total_fee) / entry_price
                    exit_reason = "BREAK_EVEN_SL" if tp1_taken else "STOP_LOSS"
                    trade_finished = True
                    break

                # TP1 scaling out check
                if not tp1_taken and h_b >= tp1_price:
                    tp1_qty = 1.0 * tp1_ratio
                    tp1_val = tp1_qty * tp1_price
                    tp1_fee = tp1_val * self.fee_rate
                    realized_gross += tp1_qty * (tp1_price - entry_price)
                    realized_fee += tp1_fee
                    curr_qty -= tp1_qty
                    tp1_taken = True
                    # Lock Break-Even SL (+0.25% buffer)
                    curr_sl = max(curr_sl, entry_price * cfg.risk_be_buffer)

                # TP2 exit check
                if tp1_taken and h_b >= tp2_price:
                    exit_price = tp2_price
                    exit_val = curr_qty * exit_price
                    exit_fee = exit_val * self.fee_rate
                    gross = (exit_val - curr_qty * entry_price) + realized_gross
                    total_fee = realized_fee + exit_fee
                    net_return = (gross - total_fee) / entry_price
                    exit_reason = "TAKE_PROFIT"
                    trade_finished = True
                    break

            if not trade_finished:
                # Time horizon expiration
                exit_price = float(df["close"].iloc[min(entry_idx + self.max_holding_bars, n - 1)])
                exit_val = curr_qty * exit_price
                exit_fee = exit_val * self.fee_rate
                gross = (exit_val - curr_qty * entry_price) + realized_gross
                total_fee = realized_fee + exit_fee
                net_return = (gross - total_fee) / entry_price
                exit_reason = "TIME_EXPIRY"

            # Binary Label: y = 1 if net profit > 0 (strictly covers 0.04% roundtrip fees)
            target_label = 1 if net_return > 0.0 else 0

            row = {col: float(df[col].iloc[i]) if col in df.columns else 0.0 for col in self.feature_columns}
            row["coin"] = coin_name
            row["datetime"] = df["datetime"].iloc[i]
            row["bar_idx"] = i
            row["entry_price"] = entry_price
            row["exit_reason"] = exit_reason
            row["net_return"] = net_return
            row["target_label"] = target_label
            records.append(row)

        cand_df = pd.DataFrame(records)
        wins = int((cand_df["target_label"] == 1).sum()) if len(cand_df) > 0 else 0
        losses = int((cand_df["target_label"] == 0).sum()) if len(cand_df) > 0 else 0
        win_rate = (wins / len(cand_df) * 100.0) if len(cand_df) > 0 else 0.0

        logger.info(
            f"[{coin_name}] Extracted {len(cand_df):,} Candidate Signals -> "
            f"Ground Truth Wins: {wins:,} ({win_rate:.2f}%), Losses: {losses:,} ({100.0 - win_rate:.2f}%)"
        )
        return cand_df

    def load_and_prepare_all_candidates(self) -> pd.DataFrame:
        """Loads and prepares candidates across all configured coins (BTC, ETH, SOL)."""
        btc_raw = self._load_raw_df("btc")
        all_candidates: List[pd.DataFrame] = []

        for c in self.coins:
            df_feat, strat, cfg = self.prepare_dataset_for_coin(c, btc_df=btc_raw)
            c_cand = self.extract_candidates_and_labels(
                df=df_feat, strategy=strat, config=cfg, coin=c, btc_df=btc_raw
            )
            all_candidates.append(c_cand)

        pooled_df = pd.concat(all_candidates, ignore_index=True)
        if "datetime" in pooled_df.columns:
            pooled_df.sort_values(by="datetime", ascending=True, inplace=True)
            pooled_df.reset_index(drop=True, inplace=True)

        logger.info(
            f"=== Multi-Asset Pooling Complete: {len(pooled_df):,} total candidates across {self.coins} ==="
        )
        return pooled_df

    def train_and_calibrate(self, cand_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits LightGBM and applies probability calibration via CalibratedClassifierCV.
        """
        X = cand_df[self.feature_columns].to_numpy(dtype=np.float64)
        y = cand_df["target_label"].to_numpy(dtype=np.int32)
        n = len(X)

        count_0 = int(np.count_nonzero(y == 0))
        count_1 = int(np.count_nonzero(y == 1))
        scale_pos_weight = (count_0 / count_1) if count_1 > 0 else 1.0

        logger.info(
            f"Class Imbalance Weighting: Label 0 = {count_0}, Label 1 = {count_1} "
            f"(scale_pos_weight = {scale_pos_weight:.2f})"
        )

        base_lgbm = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=5,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )

        if self.cv_mode == "prefit":
            # Prefit calibration: Split Train (70%), Val (15%), Test (15%)
            train_end = int(n * 0.70)
            val_start = min(train_end + self.purge_buffer, n)
            val_end = min(val_start + int(n * 0.15), n)

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            logger.info(f"Prefit CV Mode: Fitting Base Model on Train ({len(train_idx)}) and Calibrating on Val ({len(val_idx)})...")
            try:
                from sklearn.frozen import FrozenEstimator
                base_lgbm.fit(X[train_idx], y[train_idx])
                calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_lgbm), method="sigmoid"
                )
                calibrated_model.fit(X[val_idx], y[val_idx])
            except Exception:
                calibrated_model = CalibratedClassifierCV(
                    estimator=base_lgbm, method="sigmoid", cv=[(train_idx, val_idx)]
                )
                calibrated_model.fit(X[:val_end], y[:val_end])
        else:
            # Purged Time-Series Split Calibration
            cv_splitter = PurgedTimeSeriesSplit(
                n_splits=self.cv_splits, purge_buffer=self.purge_buffer
            )
            logger.info(
                f"Purged TS CV Mode: Fitting and Calibrating across {self.cv_splits} temporal folds (purge buffer = {self.purge_buffer})..."
            )
            calibrated_model = CalibratedClassifierCV(
                estimator=base_lgbm, method="sigmoid", cv=cv_splitter
            )
            calibrated_model.fit(X, y)

        # Base LightGBM probabilities
        base_lgbm.fit(X, y)
        base_probs = base_lgbm.predict_proba(X)[:, 1]
        base_auc = float(roc_auc_score(y, base_probs)) if len(np.unique(y)) > 1 else 0.50

        # Calibrated model probabilities
        probs = calibrated_model.predict_proba(X)[:, 1]
        cal_auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.50

        if cal_auc < 0.50 or cal_auc < (base_auc - 0.10):
            logger.warning(
                f"[Calibration Safeguard] CalibratedClassifierCV degraded ROC-AUC ({cal_auc:.4f} vs base {base_auc:.4f}). "
                f"Using native calibrated LightGBM classifier to ensure monotonically increasing win rate with threshold."
            )
            final_model = base_lgbm
            probs = base_probs
            auc = base_auc
            brier = float(brier_score_loss(y, probs))
        else:
            final_model = calibrated_model
            final_model.cv = None
            auc = cal_auc
            brier = float(brier_score_loss(y, probs))

        logger.info(
            f"Final Model Selected -> Brier Score Loss: {brier:.4f}, ROC-AUC: {auc:.4f} "
            f"(Probability Range: [{probs.min():.4f}, {probs.max():.4f}], Mean: {probs.mean():.4f})"
        )

        # Threshold sweep evaluation
        threshold_evals: List[Dict[str, Any]] = []
        for th in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55]:
            app = probs >= th
            n_app = int(app.sum())
            if n_app > 0:
                th_win_rate = float(y[app].mean() * 100.0)
                prec = float(precision_score(y, app, zero_division=0))
                rec = float(recall_score(y, app, zero_division=0))
                f1 = float(f1_score(y, app, zero_division=0))
            else:
                th_win_rate = 0.0
                prec = 0.0
                rec = 0.0
                f1 = 0.0

            threshold_evals.append({
                "threshold": th,
                "approved_trades": n_app,
                "approval_pct": round(n_app / n * 100.0, 2),
                "win_rate_pct": round(th_win_rate, 2),
                "baseline_win_rate": round(count_1 / n * 100.0, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })

        # Save artifact cleanly without splitter object dependencies
        if hasattr(final_model, "cv"):
            final_model.cv = None
        joblib.dump(final_model, self.output_model_path)
        logger.info(f"Saved Calibrated Model Artifact to: {self.output_model_path}")

        meta_path = self.output_model_path.with_name(
            self.output_model_path.name.replace(".joblib", "_metadata.json")
        )
        metadata = {
            "coin": "+".join(self.coins) if self.is_multi_asset else self.coin,
            "coins": self.coins,
            "is_multi_asset": self.is_multi_asset,
            "timeframe": self.timeframe,
            "total_candidates": n,
            "ground_truth_wins": count_1,
            "ground_truth_losses": count_0,
            "baseline_win_rate_pct": round(count_1 / n * 100.0, 2),
            "brier_score_loss": round(brier, 4),
            "roc_auc": round(auc, 4),
            "prob_min": round(float(probs.min()), 4),
            "prob_max": round(float(probs.max()), 4),
            "prob_mean": round(float(probs.mean()), 4),
            "feature_columns": self.feature_columns,
            "threshold_evaluations": threshold_evals,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved Companion Metadata to: {meta_path}")

        return metadata


def print_training_report(metadata: Dict[str, Any]) -> None:
    """Prints structured summary table of threshold calibration."""
    coin = metadata["coin"].upper()
    tf = metadata["timeframe"]
    print("\n" + "=" * 95)
    print(f"  LIGHTGBM META-LABELER CALIBRATION REPORT ({coin} [{tf}] - 1H CORONG 3 PILAR)  ".center(95))
    print("=" * 95)
    print(f"  Total Candidates Extracted : {metadata['total_candidates']:,} bars")
    print(f"  Ground Truth Win Rate      : {metadata['baseline_win_rate_pct']:.2f}% (Wins: {metadata['ground_truth_wins']}, Losses: {metadata['ground_truth_losses']})")
    print(f"  Brier Score Loss           : {metadata['brier_score_loss']:.4f}")
    print(f"  ROC-AUC Score              : {metadata['roc_auc']:.4f}")
    print(f"  Calibrated Prob Range      : [{metadata['prob_min']:.4f} - {metadata['prob_max']:.4f}] (Mean: {metadata['prob_mean']:.4f})")
    print("-" * 95)
    print(f"{'Threshold':<11} | {'Approved Trades':<16} | {'Approval (%)':<14} | {'Win Rate (%)':<14} | {'Precision':<10} | {'Recall':<8} | {'F1':<8}")
    print("-" * 95)
    for te in metadata["threshold_evaluations"]:
        print(
            f"{te['threshold']:<11.2f} | {te['approved_trades']:<16} | "
            f"{te['approval_pct']:>12.2f}% | {te['win_rate_pct']:>12.2f}% | "
            f"{te['precision']:<10.4f} | {te['recall']:<8.4f} | {te['f1']:<8.4f}"
        )
    print("=" * 95 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & Calibrate 1H Funnel Meta-Labeler")
    parser.add_argument("--coin", type=str, default="btc", help="Single trading asset (default: btc)")
    parser.add_argument("--coins", type=str, nargs="+", default=None, help="Multi-asset coins list (e.g. btc eth sol)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Path to processed datasets")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--use-kalman", action="store_true", default=True, help="Use Kalman filter for primary signals")
    parser.add_argument("--no-kalman", action="store_false", dest="use_kalman", help="Disable Kalman filter (use EMA)")
    parser.add_argument("--single-shot", action="store_true", default=False, help="Limit to single trade candidate per episode")
    parser.add_argument("--fee-rate", type=float, default=0.0002, help="Fee rate per side (default: 0.0002)")
    parser.add_argument("--purge-buffer", type=int, default=5, help="Purge buffer bars between folds (default: 5)")
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of cross-validation splits (default: 4)")
    parser.add_argument("--cv-mode", type=str, default="purged_ts", choices=["purged_ts", "prefit"], help="Calibration mode")
    parser.add_argument("--output-model", type=str, default=None, help="Target model artifact output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = FunnelMetaLabelingTrainer(
        coin=args.coin,
        coins=args.coins,
        timeframe=args.timeframe,
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        use_kalman=args.use_kalman,
        single_shot=args.single_shot,
        fee_rate=args.fee_rate,
        purge_buffer=args.purge_buffer,
        cv_splits=args.cv_splits,
        cv_mode=args.cv_mode,
        output_model_path=Path(args.output_model) if args.output_model else None,
    )
    if trainer.is_multi_asset:
        cand_df = trainer.load_and_prepare_all_candidates()
    else:
        df_feat, strat, cfg = trainer.prepare_dataset_for_coin(trainer.coin)
        trainer.strategy = strat
        trainer.config = cfg
        cand_df = trainer.extract_candidates_and_labels(df_feat)

    meta = trainer.train_and_calibrate(cand_df)
    print_training_report(meta)


if __name__ == "__main__":
    main()
