import numpy as np
import talib
class CCIIndicator:
    def __init__(self, cci_period=20, smoothing_period=14):
        self.cci_period = cci_period
        self.smoothing_period = smoothing_period

    def calculate_cci(self, candles):
        if len(candles) < (self.cci_period + self.smoothing_period):
            return {"cci": None, "cci_smoothing": None,"series": [float('nan')] * len(candles)}
        
        high_prices = np.array([float(h["high"]) for h in candles], dtype=np.float64)
        low_prices = np.array([float(l["low"]) for l in candles], dtype=np.float64)
        close_prices = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    
        cci_values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=self.cci_period)

        cci_smoothing_values = talib.SMA(cci_values, timeperiod=self.smoothing_period)
        
        current_cci = cci_values[-1]
        current_smoothing = cci_smoothing_values[-1]

        if np.isnan(current_cci) or np.isnan(current_smoothing):
            return {
                "cci": None, 
                "cci_smoothing": None,
                "series": np.full(len(candles), np.nan)
            }
        
        return {
            "cci": round(current_cci, 2),
            "cci_smoothing": round(current_smoothing, 2),
            "series": cci_values
        }