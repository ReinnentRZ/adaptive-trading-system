#!/usr/bin/env python3
"""
Unsupervised Market Regime Discovery Engine for Spot Trading.

Trains a 4-State Gaussian Hidden Markov Model (HMM) or Gaussian Mixture Model (GMM)
on pure market microstructure features:
- Log Returns: ln(C_t / C_{t-1})
- Signed Volume: volume_intensity * np.sign(log_return) (discriminates buying vs selling volume pressure)
- Normalized ATR Volatility: ATR_14 / Close

Performs chronological 70% Train split (zero lookahead into Val/Test), identifies
characteristic market regimes (Bullish Momentum, Low-Vol Sideways, High-Vol Breakdown, Consolidation),
calculates the Transition Probability Matrix and Expected Duration per State,
and serializes the model and metadata for downstream execution.

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
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

# Try importing TA-Lib
try:
    import talib
except ImportError as err:
    raise ImportError(
        "TA-Lib is required for indicator calculation. "
        "Please ensure TA-Lib C-library and Python bindings are installed."
    ) from err

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RegimeDiscovery")

REGIME_FEATURE_COLUMNS: List[str] = [
    "log_return",
    "signed_volume",
    "volatility",
]


@dataclass(frozen=True)
class StateProfile:
    """Statistical profile of an identified market regime cluster."""
    state_id: int
    mean_return_pct: float
    mean_signed_volume: float
    mean_volume_zscore: float
    mean_volatility_pct: float
    candle_count: int
    distribution_pct: float
    expected_duration_bars: float
    self_transition_prob: float
    interpretation: str


@dataclass(frozen=True)
class RegimeTrainingResult:
    """Summary of trained HMM/GMM regime model and evaluation profiles."""
    coin: str
    timeframe: str
    model_type: str
    train_samples: int
    total_samples: int
    n_components: int
    bullish_state_id: int
    bearish_state_id: int
    sideways_state_id: int
    state_profiles: List[StateProfile]
    transition_matrix: Optional[List[List[float]]]
    model_path: Path
    metadata_path: Path
    elapsed_seconds: float


class MicrostructureFeatureExtractor:
    """
    Extracts pure price action and microstructure features without
    conventional lagging oscillators (No RSI, No WaveTrend, No Lorentzian).
    """

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes log_return, signed_volume, normalized ATR volatility, and rolling volume intensity.
        """
        df = df.copy()
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        # 1. Log Returns: ln(C_t / C_{t-1})
        with np.errstate(divide="ignore", invalid="ignore"):
            log_return = np.full(len(close), np.nan, dtype=np.float64)
            log_return[1:] = np.log(close[1:] / close[:-1])

        # 2. Normalized ATR Volatility: ATR_14 / Close
        atr = talib.ATR(high, low, close, timeperiod=14)
        with np.errstate(divide="ignore", invalid="ignore"):
            volatility = np.where(close > 0, atr / close, np.nan)

        # 3. Volume Intensity: Z-score relative to 50-bar rolling mean & std
        vol_series = pd.Series(volume)
        vol_roll_mean = vol_series.rolling(window=50).mean().to_numpy()
        vol_roll_std = vol_series.rolling(window=50).std().to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            volume_intensity = np.where(
                (vol_roll_std > 0) & (~np.isnan(vol_roll_std)),
                (volume - vol_roll_mean) / (vol_roll_std + 1e-8),
                0.0,
            )

        # 4. Signed Volume: volume_intensity * np.sign(log_return)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_sign = np.sign(log_return)
            ret_sign = np.nan_to_num(ret_sign, nan=0.0)
            signed_volume = volume_intensity * ret_sign

        df["log_return"] = log_return
        df["atr"] = atr
        df["volatility"] = volatility
        df["volume_intensity"] = volume_intensity
        df["signed_volume"] = signed_volume

        return df


def compute_causal_hmm_states(
    model: GaussianHMM,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes causal online forward-filtered state probabilities and MAP states (Zero Lookahead Bias).
    P(S_t = j | X_1, ..., X_t) using log-space forward belief updates.
    """
    framelogprob = model._compute_log_likelihood(X)
    log_trans = np.log(np.maximum(model.transmat_, 1e-300))
    log_start = np.log(np.maximum(model.startprob_, 1e-300))
    n_samples = len(X)
    n_states = model.n_components

    causal_states = np.empty(n_samples, dtype=np.int64)
    causal_probs = np.empty((n_samples, n_states), dtype=np.float64)

    # t = 0
    u_0 = log_start + framelogprob[0]
    u_0 -= logsumexp(u_0)
    causal_states[0] = int(np.argmax(u_0))
    causal_probs[0] = np.exp(u_0)

    u_prev = u_0
    for t in range(1, n_samples):
        log_prior = np.array([logsumexp(u_prev + log_trans[:, j]) for j in range(n_states)])
        u_t = log_prior + framelogprob[t]
        u_t -= logsumexp(u_t)
        causal_states[t] = int(np.argmax(u_t))
        causal_probs[t] = np.exp(u_t)
        u_prev = u_t

    return causal_states, causal_probs


def auto_resample_continuous_ohlcv(
    processed_dir: Path,
    coin: str,
    target_tf: str = "1h",
    source_tf: str = "15m",
) -> Path:
    """
    Automatically resamples continuous OHLCV dataset from a lower timeframe (e.g., 15m)
    to a higher target timeframe (e.g., 1h) if the target dataset does not exist.
    Aggregation rules: Open=first, High=max, Low=min, Close=last, Volume=sum.
    """
    coin_clean = coin.lower()
    target_clean = target_tf.lower()
    source_clean = source_tf.lower()

    target_dir = Path(processed_dir) / coin_clean / target_clean
    target_file = target_dir / f"{coin_clean}_{target_clean}_continuous.parquet"

    if target_file.exists():
        return target_file

    source_path = Path(processed_dir) / coin_clean / source_clean / f"{coin_clean}_{source_clean}_continuous.parquet"
    if not source_path.exists():
        source_path = Path(processed_dir) / f"{coin_clean}_{source_clean}_continuous.parquet"
    if not source_path.exists():
        for fallback_tf in ["30m", "5m"]:
            alt_path = Path(processed_dir) / coin_clean / fallback_tf / f"{coin_clean}_{fallback_tf}_continuous.parquet"
            if alt_path.exists():
                source_path = alt_path
                source_clean = fallback_tf
                break

    if not source_path.exists():
        raise FileNotFoundError(
            f"Cannot resample to {target_clean}: source continuous dataset not found at {source_path}"
        )

    logger.info(f"[{coin_clean.upper()}] Automatically resampling {source_clean} continuous data to {target_clean}...")
    df_src = pd.read_parquet(source_path)
    if "datetime" not in df_src.columns:
        raise ValueError(f"Source dataset {source_path} missing 'datetime' column.")

    df_src["datetime"] = pd.to_datetime(df_src["datetime"], utc=True)
    df_src.sort_values(by="datetime", ascending=True, inplace=True)
    df_src.set_index("datetime", inplace=True)

    agg_rules: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    optional_aggs = {
        "open_time": "first",
        "close_time": "last",
        "quote_volume": "sum",
        "trades_count": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
    }
    for col, rule in optional_aggs.items():
        if col in df_src.columns:
            agg_rules[col] = rule

    offset = target_clean
    if offset == "1h":
        offset = "1h"
    elif offset == "4h":
        offset = "4h"
    elif offset == "1d":
        offset = "1D"

    df_resampled = (
        df_src.resample(offset, closed="left", label="left")
        .agg(agg_rules)
        .dropna(subset=["open"])
        .reset_index()
    )

    preferred_order = [
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
    ordered_cols = [c for c in preferred_order if c in df_resampled.columns] + [
        c for c in df_resampled.columns if c not in preferred_order
    ]
    df_resampled = df_resampled[ordered_cols]

    if "open_time" in df_resampled.columns:
        df_resampled["open_time"] = df_resampled["open_time"].astype("int64")
    if "close_time" in df_resampled.columns:
        df_resampled["close_time"] = df_resampled["close_time"].astype("int64")
    if "trades_count" in df_resampled.columns:
        df_resampled["trades_count"] = df_resampled["trades_count"].astype("int64")

    target_dir.mkdir(parents=True, exist_ok=True)
    df_resampled.to_parquet(target_file, index=False)
    csv_file = target_dir / f"{coin_clean}_{target_clean}_continuous.csv"
    df_resampled.to_csv(csv_file, index=False)

    logger.info(
        f"[{coin_clean.upper()}] Successfully created {target_clean} continuous dataset: "
        f"{len(df_resampled):,} candles ({df_resampled['datetime'].iloc[0]} to {df_resampled['datetime'].iloc[-1]}) at {target_file}"
    )
    return target_file


class RegimeModelTrainer:
    """
    Trains Gaussian HMM or Gaussian Mixture Model on pure market microstructure features.
    """

    def __init__(
        self,
        coin: str = "btc",
        timeframe: str = "15m",
        model_type: str = "hmm",
        n_components: int = 4,
        n_iter: int = 1000,
        train_ratio: float = 0.70,
        processed_dir: Path = Path("dataset/processed"),
        models_dir: Path = Path("models"),
        random_state: int = 42,
    ) -> None:
        self.coin = coin.lower()
        self.timeframe = timeframe.lower()
        self.model_type = model_type.lower()
        self.n_components = n_components
        self.n_iter = n_iter
        self.train_ratio = train_ratio
        self.processed_dir = Path(processed_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.random_state = random_state

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_dataset(self) -> pd.DataFrame:
        """Loads continuous parquet dataset and extracts microstructure features."""
        data_path = (
            self.processed_dir
            / self.coin
            / self.timeframe
            / f"{self.coin}_{self.timeframe}_continuous.parquet"
        )
        if not data_path.exists():
            data_path = self.processed_dir / f"{self.coin}_{self.timeframe}_continuous.parquet"
        if not data_path.exists():
            data_path = auto_resample_continuous_ohlcv(self.processed_dir, self.coin, self.timeframe)
        if not data_path.exists():
            raise FileNotFoundError(f"Continuous dataset not found at: {data_path}")

        df = pd.read_parquet(data_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df.sort_values(by="datetime", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

        logger.info(f"[{self.coin.upper()}] [{self.timeframe}] Ingesting {len(df):,} candles...")
        df = MicrostructureFeatureExtractor.extract_features(df)

        # Drop initial warmup NaNs
        df.dropna(subset=REGIME_FEATURE_COLUMNS + ["atr", "volume_intensity"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def train_and_profile(self) -> RegimeTrainingResult:
        """
        Executes chronological splitting, RobustScaler transformation,
        Gaussian HMM / GMM training, state profiling, and serialization.
        """
        t0 = time.perf_counter()
        logger.info(
            f"=== Training {self.model_type.upper()} Regime Model: {self.coin.upper()} [{self.timeframe}] (n_components={self.n_components}) ==="
        )

        df = self.load_and_prepare_dataset()
        total_samples = len(df)
        train_end = int(total_samples * self.train_ratio)

        df_train = df.iloc[:train_end].copy()
        logger.info(
            f"Chronological Train Partition: {len(df_train):,} bars "
            f"({df_train['datetime'].iloc[0]} to {df_train['datetime'].iloc[-1]})"
        )

        # Feature Scaling (Fit strictly on Train set to prevent lookahead leakage)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(df_train[REGIME_FEATURE_COLUMNS].to_numpy(dtype=np.float64))

        if self.model_type == "hmm":
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X_train)

            # Causal Forward Filtering across full dataset
            X_all = scaler.transform(df[REGIME_FEATURE_COLUMNS].to_numpy(dtype=np.float64))
            all_states, _ = compute_causal_hmm_states(model, X_all)
            df["regime_state"] = all_states
            df_train["regime_state"] = all_states[:train_end]

            transmat = model.transmat_.tolist()
            diag_transmat = np.diag(model.transmat_)
        else:
            # GMM Fallback
            model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                random_state=self.random_state,
                max_iter=300,
                n_init=5,
            )
            model.fit(X_train)

            X_all = scaler.transform(df[REGIME_FEATURE_COLUMNS].to_numpy(dtype=np.float64))
            all_states = model.predict(X_all)
            df["regime_state"] = all_states
            df_train["regime_state"] = all_states[:train_end]

            transmat = None
            diag_transmat = np.zeros(self.n_components)

        # Profile States from Train Set
        train_state_groups = df_train.groupby("regime_state")
        mean_returns = train_state_groups["log_return"].mean().to_dict()
        mean_signed_vols = train_state_groups["signed_volume"].mean().to_dict()
        mean_volume_z = train_state_groups["volume_intensity"].mean().to_dict()
        mean_vols = train_state_groups["volatility"].mean().to_dict()

        # Identify State Roles based on statistical characteristics
        # 1. Bullish State: Highest positive mean return
        bullish_state_id = int(max(mean_returns.keys(), key=lambda k: mean_returns[k]))
        # 2. Bearish / Breakdown State: Lowest mean return
        bearish_state_id = int(min(mean_returns.keys(), key=lambda k: mean_returns[k]))
        # 3. Sideways State: State with lowest volatility among remaining
        remaining_states = [s for s in range(self.n_components) if s not in [bullish_state_id, bearish_state_id]]
        sideways_state_id = int(min(remaining_states, key=lambda s: mean_vols.get(s, 1.0)))

        state_interpretations: Dict[int, str] = {}
        state_interpretations[bullish_state_id] = "Bullish Momentum (Upward Expansion)"
        state_interpretations[bearish_state_id] = "High-Volatility Breakdown (Bearish Dump)"
        state_interpretations[sideways_state_id] = "Low-Volatility Sideways (Consolidation)"
        for s in remaining_states:
            if s != sideways_state_id:
                state_interpretations[s] = "Moderate Volatility / Neutral Transition"

        # Build Profiles across full dataset
        state_profiles: List[StateProfile] = []
        for s in range(self.n_components):
            sub = df[df["regime_state"] == s]
            cnt = len(sub)
            dist_pct = (cnt / total_samples) * 100.0 if total_samples > 0 else 0.0
            a_ii = float(diag_transmat[s]) if transmat is not None else 0.0
            exp_dur = (1.0 / (1.0 - a_ii)) if (a_ii < 1.0 and a_ii > 0.0) else 1.0

            profile = StateProfile(
                state_id=s,
                mean_return_pct=float(sub["log_return"].mean() * 100.0) if cnt > 0 else 0.0,
                mean_signed_volume=float(sub["signed_volume"].mean()) if cnt > 0 else 0.0,
                mean_volume_zscore=float(sub["volume_intensity"].mean()) if cnt > 0 else 0.0,
                mean_volatility_pct=float(sub["volatility"].mean() * 100.0) if cnt > 0 else 0.0,
                candle_count=cnt,
                distribution_pct=dist_pct,
                expected_duration_bars=exp_dur,
                self_transition_prob=a_ii,
                interpretation=state_interpretations.get(s, f"State {s}"),
            )
            state_profiles.append(profile)

        # Sort profiles: Bullish -> Sideways -> Transition -> Bearish
        sorted_profiles = sorted(
            state_profiles,
            key=lambda p: (
                0 if p.state_id == bullish_state_id
                else (1 if p.state_id == sideways_state_id
                      else (2 if p.state_id not in [bullish_state_id, bearish_state_id, sideways_state_id]
                            else 3))
            ),
        )

        # Serialization
        suffix = "hmm" if self.model_type == "hmm" else "model"
        model_filename = f"{self.coin}_{self.timeframe}_regime_{suffix}.joblib"
        metadata_filename = f"{self.coin}_{self.timeframe}_regime_{suffix}_metadata.json"
        model_path = self.models_dir / model_filename
        metadata_path = self.models_dir / metadata_filename

        model_payload = {
            "model": model,
            "scaler": scaler,
            "model_type": self.model_type,
            "coin": self.coin,
            "timeframe": self.timeframe,
            "feature_columns": REGIME_FEATURE_COLUMNS,
            "bullish_state_id": bullish_state_id,
            "bearish_state_id": bearish_state_id,
            "sideways_state_id": sideways_state_id,
            "transition_matrix": transmat,
        }
        joblib.dump(model_payload, model_path)

        # If HMM, also save default filename for generic compatibility
        if self.model_type == "hmm":
            default_path = self.models_dir / f"{self.coin}_{self.timeframe}_regime_model.joblib"
            joblib.dump(model_payload, default_path)

        metadata_dict = {
            "coin": self.coin,
            "timeframe": self.timeframe,
            "model_type": self.model_type,
            "n_components": self.n_components,
            "feature_columns": REGIME_FEATURE_COLUMNS,
            "train_samples": len(df_train),
            "total_samples": total_samples,
            "bullish_state_id": bullish_state_id,
            "bearish_state_id": bearish_state_id,
            "sideways_state_id": sideways_state_id,
            "transition_matrix": transmat,
            "state_profiles": [asdict(p) for p in sorted_profiles],
            "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4)

        elapsed = time.perf_counter() - t0
        logger.info(f"Saved {self.model_type.upper()} model to {model_path.name} in {elapsed:.2f}s")

        return RegimeTrainingResult(
            coin=self.coin,
            timeframe=self.timeframe,
            model_type=self.model_type,
            train_samples=len(df_train),
            total_samples=total_samples,
            n_components=self.n_components,
            bullish_state_id=bullish_state_id,
            bearish_state_id=bearish_state_id,
            sideways_state_id=sideways_state_id,
            state_profiles=sorted_profiles,
            transition_matrix=transmat,
            model_path=model_path,
            metadata_path=metadata_path,
            elapsed_seconds=elapsed,
        )


def display_regime_profiles(result: RegimeTrainingResult) -> None:
    """Prints structured markdown table of discovered market regimes and Transition Matrix."""
    model_name = "GAUSSIAN HIDDEN MARKOV MODEL (HMM)" if result.model_type == "hmm" else "GAUSSIAN MIXTURE MODEL (GMM)"
    title = f"PROFIL REZIM PASAR ({model_name} - {result.n_components} STATES) - {result.coin.upper()} [{result.timeframe}]"
    print("\n" + "=" * 135)
    print(f"{title:^135}")
    print("=" * 135)
    print(
        f"{'State':<7} | {'Rata-rata Return':<18} | {'Signed Volume':<16} | "
        f"{'Volatilitas':<13} | {'Ekspektasi Durasi':<18} | {'Distribusi (%)':<16} | {'Interpretasi Rezim'}"
    )
    print("-" * 135)
    for p in result.state_profiles:
        role_marker = " [ENTRY TARGET]" if p.state_id == result.bullish_state_id else ""
        dur_str = f"{p.expected_duration_bars:.1f} bars" if result.model_type == "hmm" else "N/A (IID)"
        print(
            f"State {p.state_id:<1} | {p.mean_return_pct:>+15.4f}% | {p.mean_signed_volume:>+14.2f} | "
            f"{p.mean_volatility_pct:>10.4f}% | {dur_str:>16} | {p.distribution_pct:>13.2f}% ({p.candle_count:,}) | "
            f"{p.interpretation}{role_marker}"
        )
    print("=" * 135)

    if result.transition_matrix is not None:
        print("\n" + "=" * 75)
        print(f"{'TRANSITION PROBABILITY MATRIX (A)':^75}")
        print("=" * 75)
        k = len(result.transition_matrix)
        header = "From \\ To | " + " | ".join([f"State {j}" for j in range(k)])
        print(header)
        print("-" * 75)
        for i in range(k):
            row_vals = " | ".join([f"{result.transition_matrix[i][j]:.4f} " for j in range(k)])
            print(f"State {i:<3} | {row_vals}")
        print("=" * 75 + "\n")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unsupervised Market Regime Discovery (Gaussian HMM / GMM) Trainer"
    )
    parser.add_argument("--coin", type=str, default="btc", help="Target coin (default: btc)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Target timeframe (default: 15m)")
    parser.add_argument("--model-type", type=str, default="hmm", choices=["hmm", "gmm"], help="Regime model type (default: hmm)")
    parser.add_argument("--n-components", "--components", dest="n_components", type=int, default=4, help="Number of regime states (default: 4)")
    parser.add_argument("--n-iter", type=int, default=1000, help="Maximum EM iterations (default: 1000)")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological training split ratio (default: 0.70)")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    return parser.parse_args()


def main() -> None:
    """Main CLI execution."""
    args = parse_args()
    trainer = RegimeModelTrainer(
        coin=args.coin,
        timeframe=args.timeframe,
        model_type=args.model_type,
        n_components=args.n_components,
        n_iter=args.n_iter,
        train_ratio=args.train_ratio,
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
    )
    result = trainer.train_and_profile()
    display_regime_profiles(result)


if __name__ == "__main__":
    main()
