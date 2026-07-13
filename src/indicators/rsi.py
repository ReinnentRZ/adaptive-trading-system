import numpy as np
import talib

from src.config.indicators import INDICATORS

class RSIIndicator:
    def __init__(self, rsi_period: int = None, ma_period: int = None):
        self.rsi_period = rsi_period if rsi_period is not None else INDICATORS.rsi_period
        self.ma_period = ma_period if ma_period is not None else INDICATORS.rsi_ma_period

    def calculate_rsi(self, candles):

        if len(candles) < self.rsi_period:
            return {"rsi": None, "rsi_smoothing": None, "series": [float('nan')] * len(candles)}

        close_prices = [float(c["close"]) for c in candles]

        np_closes = np.array(close_prices, dtype=np.float64)

        rsi_values = talib.RSI(np_closes, timeperiod=self.rsi_period)

        rsi_smoothing_values = talib.SMA(rsi_values, timeperiod=self.ma_period)

        current_rsi = rsi_values[-1]
        current_smoothing = rsi_smoothing_values[-1]

        if np.isnan(current_rsi) or np.isnan(current_smoothing):
            return {
                "rsi": None, 
                "rsi_smoothing": None,
                "series": np.full(len(candles), np.nan)
            }

        return {
            "rsi": round(current_rsi, 2),
            "rsi_smoothing": round(current_smoothing, 2),
            "series": rsi_values
        }