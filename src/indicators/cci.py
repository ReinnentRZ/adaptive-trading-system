import numpy as np
import pandas as pd
import talib
from typing import Optional, Union, Sequence

from src.config.indicators import INDICATORS
from src.indicators.base import BaseIndicator, CCIResult


class CCIIndicator(BaseIndicator):
    def __init__(self, cci_period: Optional[int] = None, smoothing_period: Optional[int] = None):
        self.cci_period = cci_period if cci_period is not None else INDICATORS.cci_period
        self.smoothing_period = smoothing_period if smoothing_period is not None else INDICATORS.cci_smoothing_period

    def calculate(self, data: pd.DataFrame) -> CCIResult:
        """
        Calculate the Commodity Channel Index (CCI) indicator.

        Args:
            data: pd.DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  of type float64.

        Returns:
            CCIResult containing current CCI, smoothed CCI, and the full series.
        """
        df = self._prepare_data(data)
        min_length = self.cci_period + self.smoothing_period

        if len(df) < min_length:
            return CCIResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        high_prices = df["high"].to_numpy()
        low_prices = df["low"].to_numpy()
        close_prices = df["close"].to_numpy()

        cci_values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=self.cci_period)
        cci_smoothing_values = talib.SMA(cci_values, timeperiod=self.smoothing_period)

        current_cci = cci_values[-1]
        current_smoothing = cci_smoothing_values[-1]

        if np.isnan(current_cci) or np.isnan(current_smoothing):
            return CCIResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        return CCIResult(
            current=float(round(current_cci, 2)),
            smoothing=float(round(current_smoothing, 2)),
            series=cci_values
        )

    def calculate_cci(self, candles: Union[pd.DataFrame, Sequence[dict]]) -> CCIResult:
        """
        Legacy method for backward compatibility.
        """
        if not isinstance(candles, pd.DataFrame):
            candles = pd.DataFrame(candles)
        return self.calculate(candles)