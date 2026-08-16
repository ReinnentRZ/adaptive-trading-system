import numpy as np
import pandas as pd
import talib
from typing import Optional, Union, Sequence

from src.config.indicators import INDICATORS
from src.indicators.base import BaseIndicator, ADXResult


class ADXIndicator(BaseIndicator):
    def __init__(self, adx_period: Optional[int] = None, smoothing_period: Optional[int] = None):
        self.adx_period = adx_period if adx_period is not None else INDICATORS.adx_period
        self.smoothing_period = smoothing_period if smoothing_period is not None else INDICATORS.adx_smoothing_period

    def calculate(self, data: pd.DataFrame) -> ADXResult:
        """
        Calculate the Average Directional Index (ADX) indicator.

        Args:
            data: pd.DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  of type float64.

        Returns:
            ADXResult containing current ADX, smoothed ADX, and the full series.
        """
        df = self._prepare_data(data)
        min_length = (self.adx_period * 2) + self.smoothing_period

        if len(df) < min_length:
            return ADXResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        high_prices = df["high"].to_numpy()
        low_prices = df["low"].to_numpy()
        close_prices = df["close"].to_numpy()

        adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
        adx_smoothing_values = talib.SMA(adx_values, timeperiod=self.smoothing_period)

        current_adx = adx_values[-1]
        current_smoothing = adx_smoothing_values[-1]

        if np.isnan(current_adx) or np.isnan(current_smoothing):
            return ADXResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        return ADXResult(
            current=float(round(current_adx, 2)),
            smoothing=float(round(current_smoothing, 2)),
            series=adx_values
        )

    def calculate_adx(self, candles: Union[pd.DataFrame, Sequence[dict]]) -> ADXResult:
        """
        Legacy method for backward compatibility.
        """
        if not isinstance(candles, pd.DataFrame):
            candles = pd.DataFrame(candles)
        return self.calculate(candles)
