import numpy as np
import talib

class WTIndicator:
    # Tambahin wt_sma_length=4 untuk garis sinyal silang (WT2)
    def __init__(self, channel_length=10, average_length=21, wt_sma_length=4):
        self.channel_length = channel_length
        self.average_length = average_length
        self.wt_sma_length = wt_sma_length

    def calculate_wt(self, candles):
        if len(candles) < (self.channel_length + self.average_length + self.wt_sma_length):
            return {"wt1": None, "wt2": None,"series": [float('nan')] * len(candles)}
        
        high_prices = np.array([float(h["high"]) for h in candles], dtype=np.float64)
        low_prices = np.array([float(l["low"]) for l in candles], dtype=np.float64)
        close_prices = np.array([float(c["close"]) for c in candles], dtype=np.float64)

        hlc3 = (high_prices + low_prices + close_prices) / 3.0

        esa = talib.EMA(hlc3, timeperiod=self.channel_length)

        absolute_deviation = np.abs(hlc3 - esa)

        d = talib.EMA(absolute_deviation, timeperiod=self.channel_length)

        commodity_index = (hlc3 - esa) / (0.015 * d)

        wt1_values = talib.EMA(commodity_index, timeperiod=self.average_length)

        wt2_values = talib.SMA(wt1_values, timeperiod=self.wt_sma_length)

        current_wt1 = wt1_values[-1]
        current_wt2 = wt2_values[-1]

        if np.isnan(current_wt1) or np.isnan(current_wt2):
            return {
                "wt1": None, 
                "wt2": None,
                "series": np.full(len(candles), np.nan) 
            }

        return {
            "wt1": round(current_wt1, 2),
            "wt2": round(current_wt2, 2),
            "series": wt1_values

        }