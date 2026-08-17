import numpy as np
import pandas as pd
import talib
from typing import Optional, Union, Sequence

from src.config.trading import TRADING
from src.indicators.base import BaseIndicator, ATRResult


class ATRIndicator(BaseIndicator):
    def __init__(self, atr_period: Optional[int] = None):
        self.atr_period = atr_period if atr_period is not None else getattr(TRADING, "atr_period", 14)

    def _prepare_data(self, data: Union[pd.DataFrame, Sequence[dict]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, (list, tuple)):
            df = pd.DataFrame(data)
        else:
            raise TypeError("Input data must be a pandas DataFrame or a sequence of dicts.")

        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}' in input data.")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        return df

    def calculate(self, data: pd.DataFrame) -> ATRResult:
        """
        Calculate the Average True Range (ATR) indicator.

        Args:
            data: pd.DataFrame with columns ['high', 'low', 'close']
                  of type float64.

        Returns:
            ATRResult containing current ATR and the full series.
        """
        df = self._prepare_data(data)
        # TA-Lib ATR requires at least timeperiod + 1 points to get a non-nan result
        min_length = self.atr_period + 1

        if len(df) < min_length:
            return ATRResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        high_prices = df["high"].to_numpy()
        low_prices = df["low"].to_numpy()
        close_prices = df["close"].to_numpy()

        atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.atr_period)
        current_atr = atr_values[-1]

        if np.isnan(current_atr):
            return ATRResult(
                current=None,
                smoothing=None,
                series=np.full(len(df), np.nan)
            )

        return ATRResult(
            current=float(round(current_atr, 4)),
            smoothing=None,
            series=atr_values
        )

    def calculate_atr(self, candles: Union[pd.DataFrame, Sequence[dict]]) -> ATRResult:
        """
        Legacy/convenience method for backward compatibility.
        """
        if not isinstance(candles, pd.DataFrame):
            candles = pd.DataFrame(candles)
        return self.calculate(candles)
