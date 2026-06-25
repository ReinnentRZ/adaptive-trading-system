import numpy as np
import talib

class RSIIndicator:
    def __init__(self, rsi_period, ma_period):
        self.rsi_period = rsi_period
        self.ma_period = ma_period 

    def calculate_rsi(self, candles):

        if len(candles) < self.rsi_period:
            return {"rsi": None, "rsi_smoothing": None}

        close_prices = [float(c["close"]) for c in candles]

        np_closes = np.array(close_prices, dtype=np.float64)

        rsi_values = talib.RSI(np_closes, timeperiod=self.rsi_period)

        rsi_smoothing_values = talib.SMA(rsi_values, timeperiod=self.ma_period)

        current_rsi = rsi_values[-1]
        current_smoothing = rsi_smoothing_values[-1]

        if np.isnan(current_rsi) or np.isnan(current_smoothing):
            return {"rsi": None, "rsi_smoothing": None}

        return {
            "rsi": round(current_rsi, 2),
            "rsi_smoothing": round(current_smoothing, 2)
        }