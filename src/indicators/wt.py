import numpy as np
import pandas as pd
import talib
from typing import Optional, Union, Sequence

from src.config.indicators import INDICATORS
from src.indicators.base import BaseIndicator, WTResult


class WTIndicator(BaseIndicator):
    def __init__(
        self,
        channel_length: Optional[int] = None,
        average_length: Optional[int] = None,
        wt_sma_length: Optional[int] = None,
    ):
        self.channel_length = channel_length if channel_length is not None else INDICATORS.wt_channel_length
        self.average_length = average_length if average_length is not None else INDICATORS.wt_average_length
        self.wt_sma_length = wt_sma_length if wt_sma_length is not None else INDICATORS.wt_sma_length

    def calculate(self, data: pd.DataFrame) -> WTResult:
        """
        Calculate the WaveTrend (WT) indicator.

        Args:
            data: pd.DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  of type float64.

        Returns:
            WTResult containing current wt1, wt2, and the full series (wt1).
        """
        df = self._prepare_data(data)
        min_length = self.channel_length + self.average_length + self.wt_sma_length

        if len(df) < min_length:
            return WTResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        high_prices = df["high"].to_numpy()
        low_prices = df["low"].to_numpy()
        close_prices = df["close"].to_numpy()

        hlc3 = (high_prices + low_prices + close_prices) / 3.0

        esa = talib.EMA(hlc3, timeperiod=self.channel_length)
        absolute_deviation = np.abs(hlc3 - esa)
        d = talib.EMA(absolute_deviation, timeperiod=self.channel_length)

        # Handle division by zero or very small values safely
        divisor = 0.015 * d
        # Using np.errstate to avoid runtime warnings for division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            commodity_index = np.where(divisor != 0, (hlc3 - esa) / divisor, 0.0)

        wt1_values = talib.EMA(commodity_index, timeperiod=self.average_length)
        wt2_values = talib.SMA(wt1_values, timeperiod=self.wt_sma_length)

        current_wt1 = wt1_values[-1]
        current_wt2 = wt2_values[-1]

        if np.isnan(current_wt1) or np.isnan(current_wt2):
            return WTResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        return WTResult(
            current=float(round(current_wt1, 2)),
            smoothing=float(round(current_wt2, 2)),
            series=wt1_values
        )

    def calculate_wt(self, candles: Union[pd.DataFrame, Sequence[dict]]) -> WTResult:
        """
        Legacy method for backward compatibility.
        """
        if not isinstance(candles, pd.DataFrame):
            candles = pd.DataFrame(candles)
        return self.calculate(candles)