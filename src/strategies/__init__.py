"""
Strategies Module.

Exports:
  - RegimeFunnelStrategy: Central SSOT engine for the 3-Pilar Market Regime Corong.
"""

from __future__ import annotations

from src.strategies.meta_labeler import (
    MetaLabelResult,
    MetaLabelingGate,
)
from src.strategies.quant_models import (
    GarchForecastResult,
    KalmanTrendResult,
    compute_rolling_garch_volatility,
    forecast_garch_volatility,
    kalman_trend_filter,
)
from src.strategies.regime_funnel import (
    RegimeFunnelStrategy,
    calculate_adaptive_brackets,
)

__all__ = [
    "RegimeFunnelStrategy",
    "calculate_adaptive_brackets",
    "kalman_trend_filter",
    "forecast_garch_volatility",
    "compute_rolling_garch_volatility",
    "KalmanTrendResult",
    "GarchForecastResult",
    "MetaLabelingGate",
    "MetaLabelResult",
]
