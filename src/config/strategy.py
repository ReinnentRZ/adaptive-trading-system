"""
Strategy Configuration Module.

Defines RegimeFunnelConfig for the 3-Pilar Market Regime Architecture:
  - Pilar 1: Gaussian HMM 1H + Macro EMA 200 Trend Filter
  - Pilar 2: Micro Timing Pullback (EMA 9 / RSI 52)
  - Pilar 3: Dynamic Risk & Scaling Out 50:50 (TP1 0.80x, BE Lock 1.0025, TP2 1.20x, SL 0.70x)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _get_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


@dataclass
class RegimeFunnelConfig:
    """
    Configuration for the 3-Pilar Market Regime Funnel Strategy.
    All attributes support environment variable overrides with robust defaults.
    """
    # Pilar 1: Macro Gate
    hmm_model_path: str = _get_env_str("HMM_MODEL_PATH", "models/btc_1h_regime_hmm.joblib")
    hmm_bullish_state_id: int = _get_env_int("HMM_BULLISH_STATE_ID", 0)
    state_age_max: int = _get_env_int("STATE_AGE_MAX", 4)
    ema_trend_period: int = _get_env_int("EMA_TREND_PERIOD", 200)
    single_shot_per_episode: bool = _get_env_bool("SINGLE_SHOT_PER_EPISODE", True)

    # Pilar 2: Micro Timing Pullback
    pullback_ema_period: int = _get_env_int("PULLBACK_EMA_PERIOD", 9)
    pullback_rsi_period: int = _get_env_int("PULLBACK_RSI_PERIOD", 14)
    pullback_rsi_threshold: float = _get_env_float("PULLBACK_RSI_THRESHOLD", 52.0)

    # Pilar 3: Dynamic Risk & Scaling Out
    risk_sl_mult: float = _get_env_float("RISK_SL_MULT", 0.70)       # Initial SL = Entry - (0.70 * ATR)
    risk_tp1_mult: float = _get_env_float("RISK_TP1_MULT", 0.80)      # TP1 = Entry + (0.80 * ATR) -> Jual 50%
    risk_tp2_mult: float = _get_env_float("RISK_TP2_MULT", 1.20)      # TP2 = Entry + (1.20 * ATR) -> Jual 50%
    risk_be_buffer: float = _get_env_float("RISK_BE_BUFFER", 1.0025)   # Hard-floored BE lock (+0.25% net)
    max_hold_bars: int = _get_env_int("MAX_HOLD_BARS", 12)

    # Execution & Fees (Maker Tier)
    capital_total: float = _get_env_float("CAPITAL_TOTAL", 100.0)
    trade_allocation: float = _get_env_float("TRADE_ALLOCATION", 30.0)
    maker_fee: float = _get_env_float("MAKER_FEE", 0.0002)        # 0.02% per sisi (0.04% round-trip)
    slippage: float = _get_env_float("SLIPPAGE", 0.0000)         # 0.0% limit order


# Lazy resolution for backward compatibility with existing legacy Lorentzian strategy callers
def __getattr__(name: str) -> Any:
    if name in ("STRATEGY", "StrategySettings", "AI_THRESHOLD", "REGIME_FUNNEL"):
        import src.config as cfg
        if name == "STRATEGY":
            return cfg.config.strategy
        elif name == "StrategySettings":
            return cfg.StrategyConfig
        elif name == "AI_THRESHOLD":
            return cfg.config.strategy.ai_threshold
        elif name == "REGIME_FUNNEL":
            return cfg.config.regime_funnel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
