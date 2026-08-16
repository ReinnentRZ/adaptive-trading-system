import numpy as np
import pandas as pd
import talib
from typing import Optional, Union, Sequence

from src.config.indicators import INDICATORS
from src.indicators.base import BaseIndicator, RSIResult


class RSIIndicator(BaseIndicator):
    def __init__(self, rsi_period: Optional[int] = None, ma_period: Optional[int] = None):
        self.rsi_period = rsi_period if rsi_period is not None else INDICATORS.rsi_period
        self.ma_period = ma_period if ma_period is not None else INDICATORS.rsi_ma_period

    def calculate(self, data: pd.DataFrame) -> RSIResult:
        """
        Calculate the Relative Strength Index (RSI) indicator.

        Args:
            data: pd.DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  of type float64.

        Returns:
            RSIResult containing current RSI, smoothed RSI, and the full series.
        """
        df = self._prepare_data(data)
        min_length = self.rsi_period

        if len(df) < min_length:
            return RSIResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        close_prices = df["close"].to_numpy()

        rsi_values = talib.RSI(close_prices, timeperiod=self.rsi_period)
        rsi_smoothing_values = talib.SMA(rsi_values, timeperiod=self.ma_period)

        current_rsi = rsi_values[-1]
        current_smoothing = rsi_smoothing_values[-1]

        if np.isnan(current_rsi) or np.isnan(current_smoothing):
            return RSIResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        return RSIResult(
            current=float(round(current_rsi, 2)),
            smoothing=float(round(current_smoothing, 2)),
            series=rsi_values
        )

    def calculate_rsi(self, candles: Union[pd.DataFrame, Sequence[dict]]) -> RSIResult:
        """
        Legacy method for backward compatibility.
        """
        if not isinstance(candles, pd.DataFrame):
            candles = pd.DataFrame(candles)
        return self.calculate(candles)