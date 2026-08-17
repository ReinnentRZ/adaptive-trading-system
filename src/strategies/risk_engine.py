from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Union
import numpy as np
import pandas as pd
import talib

from src.core.enums import OrderType


@dataclass(frozen=True)
class AdaptiveRiskParams:
    calculated_atr: float
    applied_period: int
    market_regime: str
    sl_multiplier: float
    tp_multiplier: float
    target_rr_ratio: float
    sl_price: float
    tp_price: float


class AdaptiveRiskCalculator:
    @staticmethod
    def _get_tick_size(symbol: str) -> Decimal:
        tick_sizes = {
            "BTCUSDT": Decimal("0.01"),
            "ETHUSDT": Decimal("0.01"),
            "BNBUSDT": Decimal("0.1"),
            "SOLUSDT": Decimal("0.01"),
            "ADAUSDT": Decimal("0.0001"),
            "XRPUSDT": Decimal("0.0001"),
            "DOGEUSDT": Decimal("0.00001"),
        }
        return tick_sizes.get(symbol.upper(), Decimal("0.01"))

    @classmethod
    def calculate_risk_params(
        cls,
        data: pd.DataFrame,
        entry_price: Union[float, Decimal],
        direction: OrderType,
        symbol: str = "SOLUSDT"
    ) -> AdaptiveRiskParams:
        """
        Calculate adaptive risk management parameters based on market regime.
        All calculations use closed candles only (via shift(1)).
        """
        if isinstance(entry_price, float):
            entry_dec = Decimal(str(entry_price))
        else:
            entry_dec = entry_price

        # Standard default fallback parameters
        fallback_params = AdaptiveRiskParams(
            calculated_atr=0.0,
            applied_period=14,
            market_regime="RANGING",
            sl_multiplier=1.2,
            tp_multiplier=2.4,
            target_rr_ratio=2.0,
            sl_price=float(entry_dec),
            tp_price=float(entry_dec)
        )

        if len(data) < 55:  # Ensure we have enough data for ATR(14) + SMA(ATR, 50) + shift
            return fallback_params

        # Convert to numpy arrays
        high_prices = data["high"].to_numpy().astype(np.float64)
        low_prices = data["low"].to_numpy().astype(np.float64)
        close_prices = data["close"].to_numpy().astype(np.float64)

        # 1. Calculate indicators
        adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        atr_sma_values = talib.SMA(atr_values, timeperiod=50)

        # 2. Shift to prevent lookahead bias (closed candles only)
        adx_shifted = pd.Series(adx_values).shift(1)
        atr_shifted = pd.Series(atr_values).shift(1)
        atr_sma_shifted = pd.Series(atr_sma_values).shift(1)

        # Extract last closed values
        adx_val = adx_shifted.iloc[-1]
        atr_val = atr_shifted.iloc[-1]
        atr_sma_val = atr_sma_shifted.iloc[-1]

        if np.isnan(adx_val) or np.isnan(atr_val) or np.isnan(atr_sma_val) or atr_sma_val == 0:
            return fallback_params

        # 3. Regime Detection Logic
        volatility_ratio = atr_val / atr_sma_val

        if volatility_ratio > 1.3:
            regime = "HIGH_VOLATILITY"
            applied_period = 14
            sl_multiplier = 2.0
            tp_multiplier = 4.0
        elif adx_val > 25:
            regime = "TRENDING"
            applied_period = 10
            sl_multiplier = 1.5
            tp_multiplier = 4.5
        else:
            regime = "RANGING"
            applied_period = 14
            sl_multiplier = 1.2
            tp_multiplier = 2.4

        # 4. Calculate dynamic ATR using applied period
        dynamic_atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=applied_period)
        dynamic_atr_shifted = pd.Series(dynamic_atr_values).shift(1)
        calculated_atr = dynamic_atr_shifted.iloc[-1]

        if np.isnan(calculated_atr):
            return fallback_params

        # 5. Clamping Risk-to-Reward (R:R) Ratio (between 1.5 and 4.0)
        target_rr_ratio = tp_multiplier / sl_multiplier
        target_rr_ratio = max(1.5, min(4.0, target_rr_ratio))
        tp_multiplier = sl_multiplier * target_rr_ratio

        # 6. SL and TP price calculations
        atr_dec = Decimal(str(calculated_atr))
        sl_mult_dec = Decimal(str(sl_multiplier))
        tp_mult_dec = Decimal(str(tp_multiplier))
        tick_size = cls._get_tick_size(symbol)

        if direction == OrderType.LONG:
            sl_raw = entry_dec - (sl_mult_dec * atr_dec)
            tp_raw = entry_dec + (tp_mult_dec * atr_dec)
        else:  # SHORT
            sl_raw = entry_dec + (sl_mult_dec * atr_dec)
            tp_raw = entry_dec - (tp_mult_dec * atr_dec)

        # Clamping price to avoid negative SL/TP
        sl_raw = max(Decimal("0.0001"), sl_raw)
        tp_raw = max(Decimal("0.0001"), tp_raw)

        sl_price = sl_raw.quantize(tick_size)
        tp_price = tp_raw.quantize(tick_size)

        return AdaptiveRiskParams(
            calculated_atr=float(calculated_atr),
            applied_period=int(applied_period),
            market_regime=regime,
            sl_multiplier=float(sl_multiplier),
            tp_multiplier=float(tp_multiplier),
            target_rr_ratio=float(target_rr_ratio),
            sl_price=float(sl_price),
            tp_price=float(tp_price)
        )
