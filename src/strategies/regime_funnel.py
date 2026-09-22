"""
Market Regime Corong (3-Pilar) Strategy Module.

Single Source of Truth (SSOT) implementation for:
  - Pilar 1: Causal Gaussian HMM Regime Gate (Bullish State, Single-Shot, state_age <= 4)
             + Macro Structural Trend Filter (Close > EMA 200).
  - Pilar 2: Micro Pullback Timing Trigger (Low <= EMA(9) OR RSI(14) <= 52.0).
  - Pilar 3: Dynamic Risk Engine (Scaling Out 50:50, ATR TP1/TP2, Hard-Floored BE Lock).

All feature calculations and inferences are strictly causal with Zero Lookahead Bias.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import talib

from src.config import REGIME_FUNNEL, RegimeFunnelConfig

logger = logging.getLogger("regime_funnel_strategy")


class RegimeFunnelStrategy:
    """
    Centralized quantitative strategy engine implementing the 3-Pilar Regime Corong.
    Integrates HMM causal regime state tracking, trend filtering, micro-pullback triggers,
    and 50:50 scaling-out risk boundaries.
    """

    def __init__(self, config: Optional[RegimeFunnelConfig] = None) -> None:
        self.config: RegimeFunnelConfig = config or REGIME_FUNNEL
        self.model: Any = None
        self.scaler: Any = None
        self.feature_columns: List[str] = ["log_return", "signed_volume", "volatility"]
        self.bullish_state_id: int = self.config.hmm_bullish_state_id
        self.bearish_state_id: int = 3
        self.sideways_state_id: int = 2

        self._load_regime_model()

    def _resolve_model_path(self, raw_path: str) -> Path:
        """Resolves model path across relative execution roots."""
        path_obj = Path(raw_path)
        if path_obj.exists():
            return path_obj

        # Check relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / raw_path
        if candidate.exists():
            return candidate

        # Fallback to models/btc_1h_regime_hmm.joblib
        fallback = project_root / "models" / "btc_1h_regime_hmm.joblib"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(f"Regime HMM model file not found at '{raw_path}' or '{candidate}'.")

    def _load_regime_model(self) -> None:
        """Loads pre-trained Gaussian HMM model and its RobustScaler."""
        resolved_path = self._resolve_model_path(self.config.hmm_model_path)
        payload = joblib.load(resolved_path)

        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_columns = payload.get("feature_columns", ["log_return", "signed_volume", "volatility"])
        self.bullish_state_id = payload.get("bullish_state_id", self.config.hmm_bullish_state_id)
        self.bearish_state_id = payload.get("bearish_state_id", 3)
        self.sideways_state_id = payload.get("sideways_state_id", 2)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all strategy indicators and causal microstructure features:
          - ema_200: Macro trend filter (span=200)
          - ema_9: Micro pullback dynamic boundary (period=9)
          - rsi_14: Momentum oscillator (period=14)
          - atr_14: Volatility range (period=14)
          - Microstructure features: log_return, normalized_atr/volatility, volume_intensity, signed_volume
        Strictly causal: Indicator values at index t rely solely on data from index <= t.
        """
        df = df.copy()
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        # 1. Log Returns: ln(Close_t / Close_{t-1})
        with np.errstate(divide="ignore", invalid="ignore"):
            log_return = np.full(len(close), np.nan, dtype=np.float64)
            log_return[1:] = np.log(close[1:] / close[:-1])

        # 2. Normalized ATR Volatility: ATR_14 / Close
        atr_14 = talib.ATR(high, low, close, timeperiod=14)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_atr = np.where(close > 0, atr_14 / close, np.nan)

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

        # 5. Technical Indicators
        ema_9 = talib.EMA(close, timeperiod=self.config.pullback_ema_period)
        rsi_14 = talib.RSI(close, timeperiod=self.config.pullback_rsi_period)
        ema_200 = df["close"].ewm(span=self.config.ema_trend_period, adjust=False).mean().to_numpy(dtype=np.float64)

        # Populate DataFrame
        df["log_return"] = log_return
        df["atr_14"] = atr_14
        df["atr"] = atr_14  # Alias for backward compatibility
        df["normalized_atr"] = normalized_atr
        df["volatility"] = normalized_atr  # Model feature alias
        df["volume_intensity"] = volume_intensity
        df["signed_volume"] = signed_volume
        df["ema_9"] = ema_9
        df["rsi_14"] = rsi_14
        df["ema_200"] = ema_200

        return df

    def compute_causal_states(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes causal online forward-filtered state probabilities and MAP states.
        Uses recursive log-space forward belief updates P(S_t = j | X_0...X_t)
        without backward smoothing, ensuring Zero Lookahead Bias.
        """
        X = self.scaler.transform(df[self.feature_columns].to_numpy(dtype=np.float64))
        framelogprob = self.model._compute_log_likelihood(X)
        log_trans = np.log(np.maximum(self.model.transmat_, 1e-300))
        log_start = np.log(np.maximum(self.model.startprob_, 1e-300))
        n_samples = len(X)
        n_states = self.model.n_components

        causal_states = np.empty(n_samples, dtype=np.int64)
        causal_probs = np.empty((n_samples, n_states), dtype=np.float64)

        # t = 0 initialization
        u_prev = log_start + framelogprob[0]
        u_prev -= logsumexp(u_prev)
        causal_states[0] = int(np.argmax(u_prev))
        causal_probs[0] = np.exp(u_prev)

        # Online step-by-step causal belief update
        for t in range(1, n_samples):
            log_prior = np.array([logsumexp(u_prev + log_trans[:, j]) for j in range(n_states)])
            u_t = log_prior + framelogprob[t]
            u_t -= logsumexp(u_t)
            causal_states[t] = int(np.argmax(u_t))
            causal_probs[t] = np.exp(u_t)
            u_prev = u_t

        return causal_states, causal_probs

    def evaluate_gates(
        self,
        df: pd.DataFrame,
        idx: int,
        current_state: int,
        state_age: int,
        traded_in_episode: bool,
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Evaluates Cascading Corong gates:
          - Pilar 1 (Macro Gate):
              HMM State == Bullish (0) AND
              Close > EMA(200) AND
              state_age <= state_age_max (4) AND
              not traded_in_episode (Single-Shot)
          - Pilar 2 (Micro Pullback):
              Low <= EMA(9) OR RSI(14) <= pullback_rsi_threshold (52.0)
        Returns:
          (signal_passed: bool, gate_details: dict)
        """
        bar = df.iloc[idx]
        close_p = float(bar["close"])
        low_p = float(bar["low"])
        ema_200 = float(bar["ema_200"])
        ema_9 = float(bar["ema_9"])
        rsi_14 = float(bar["rsi_14"])

        # Pilar 1: Macro Gate
        single_shot_ok = (not traded_in_episode) if self.config.single_shot_per_episode else True
        gate_1 = (
            (current_state == self.config.hmm_bullish_state_id)
            and (close_p > ema_200)
            and (state_age <= self.config.state_age_max)
            and single_shot_ok
        )

        # Pilar 2: Micro Pullback Trigger
        gate_2 = (low_p <= ema_9) or (rsi_14 <= self.config.pullback_rsi_threshold)

        signal_passed = bool(gate_1 and gate_2)
        return signal_passed, {"pilar_1": gate_1, "pilar_2": gate_2}

    def calculate_risk_brackets(self, entry_price: float, atr: float) -> Dict[str, float]:
        """
        Calculates Pilar 3 execution boundaries:
          - sl_price: Initial Stop Loss (Entry - 0.70 * ATR)
          - tp1_price: Target 1 (Entry + 0.80 * ATR) -> Scales out 50%
          - tp2_price: Target 2 (Entry + 1.20 * ATR) -> Scales out remaining 50%
          - be_price: Hard-floored Break-Even lock (Entry * 1.0025)
        """
        sl_price = entry_price - (self.config.risk_sl_mult * atr)
        tp1_price = entry_price + (self.config.risk_tp1_mult * atr)
        tp2_price = entry_price + (self.config.risk_tp2_mult * atr)
        be_price = entry_price * self.config.risk_be_buffer

        return {
            "sl_price": float(sl_price),
            "tp1_price": float(tp1_price),
            "tp2_price": float(tp2_price),
            "be_price": float(be_price),
        }
