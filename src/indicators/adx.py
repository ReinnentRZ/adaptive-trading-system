import numpy as np
import talib

from src.config.indicators import INDICATORS

class ADXIndicator:
    def __init__(self, adx_period: int = None, smoothing_period: int = None):
        self.adx_period = adx_period if adx_period is not None else INDICATORS.adx_period
        self.smoothing_period = smoothing_period if smoothing_period is not None else INDICATORS.adx_smoothing_period

    def calculate_adx(self, candles):              
        if len(candles) < (self.adx_period * 2) + self.smoothing_period:
            return {"adx": None, "adx_smoothing": None, "series": [float('nan')] * len(candles)}
        
        high_prices = np.array([float(h["high"]) for h in candles], dtype=np.float64)
        low_prices = np.array([float(l["low"]) for l in candles], dtype=np.float64)
        close_prices = np.array([float(c["close"]) for c in candles], dtype=np.float64)

        adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.adx_period)

        adx_smoothing_values = talib.SMA(adx_values, timeperiod=self.smoothing_period)

        current_adx = adx_values[-1]
        current_smoothing = adx_smoothing_values[-1]

        if np.isnan(current_adx) or np.isnan(current_smoothing):
            return {
                "adx": None, 
                "adx_smoothing": None,
                "series": np.full(len(candles), np.nan) 
            }
        

        return {
            "adx": round(current_adx, 2),
            "adx_smoothing": round(current_smoothing, 2),
            "series": adx_values
        }
