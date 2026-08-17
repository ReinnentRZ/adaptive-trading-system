import unittest
import numpy as np
import pandas as pd
import talib

from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.rsi import RSIIndicator
from src.indicators.wt import WTIndicator
from src.indicators.atr import ATRIndicator
from src.indicators.base import IndicatorResult, RSIResult, ADXResult, CCIResult, WTResult, ATRResult


class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Generate 100 bars of synthetic klines
        np.random.seed(42)
        self.length = 100
        
        # Simulating random walk for close prices
        close = 100.0 + np.cumsum(np.random.normal(0, 1, self.length))
        high = close + np.abs(np.random.normal(1, 0.5, self.length))
        low = close - np.abs(np.random.normal(1, 0.5, self.length))
        open_ = close + np.random.normal(0, 0.5, self.length)
        volume = np.random.uniform(100, 1000, self.length)
        
        self.df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
        self.candles_list = self.df.to_dict("records")

    def test_rsi_indicator(self):
        indicator = RSIIndicator(rsi_period=14, ma_period=14)
        
        # 1. Test calculation with DataFrame
        res = indicator.calculate(self.df)
        self.assertIsInstance(res, RSIResult)
        self.assertIsInstance(res.series, np.ndarray)
        self.assertEqual(len(res.series), self.length)
        
        # 2. Test legacy method with list of dicts
        res_legacy = indicator.calculate_rsi(self.candles_list)
        self.assertIsInstance(res_legacy, RSIResult)
        self.assertEqual(res.current, res_legacy.current)
        self.assertEqual(res.smoothing, res_legacy.smoothing)
        np.testing.assert_array_equal(res.series, res_legacy.series)

        # 3. Test backward compatibility dictionary access
        self.assertEqual(res["rsi"], res.current)
        self.assertEqual(res["rsi_smoothing"], res.smoothing)
        np.testing.assert_array_equal(res["series"], res.series)
        self.assertEqual(res.get("rsi"), res.current)
        self.assertIsNone(res.get("invalid_key"))

        # 4. Test properties
        self.assertEqual(res.rsi, res.current)
        self.assertEqual(res.rsi_smoothing, res.smoothing)

        # 5. Verify against raw TA-Lib values
        expected_rsi = talib.RSI(self.df["close"].to_numpy(), timeperiod=14)
        np.testing.assert_array_equal(res.series, expected_rsi)

    def test_adx_indicator(self):
        indicator = ADXIndicator(adx_period=14, smoothing_period=14)
        
        # 1. Test calculation with DataFrame
        res = indicator.calculate(self.df)
        self.assertIsInstance(res, ADXResult)
        self.assertIsInstance(res.series, np.ndarray)
        self.assertEqual(len(res.series), self.length)
        
        # 2. Test legacy method
        res_legacy = indicator.calculate_adx(self.candles_list)
        self.assertEqual(res.current, res_legacy.current)
        self.assertEqual(res.smoothing, res_legacy.smoothing)
        np.testing.assert_array_equal(res.series, res_legacy.series)

        # 3. Test dict-like mapping
        self.assertEqual(res["adx"], res.current)
        self.assertEqual(res["adx_smoothing"], res.smoothing)
        self.assertEqual(res.adx, res.current)
        self.assertEqual(res.adx_smoothing, res.smoothing)

    def test_cci_indicator(self):
        indicator = CCIIndicator(cci_period=20, smoothing_period=14)
        
        # 1. Test calculation with DataFrame
        res = indicator.calculate(self.df)
        self.assertIsInstance(res, CCIResult)
        self.assertIsInstance(res.series, np.ndarray)
        self.assertEqual(len(res.series), self.length)
        
        # 2. Test legacy method
        res_legacy = indicator.calculate_cci(self.candles_list)
        self.assertEqual(res.current, res_legacy.current)
        self.assertEqual(res.smoothing, res_legacy.smoothing)

        # 3. Test dict-like mapping
        self.assertEqual(res["cci"], res.current)
        self.assertEqual(res["cci_smoothing"], res.smoothing)
        self.assertEqual(res.cci, res.current)
        self.assertEqual(res.cci_smoothing, res.smoothing)

    def test_wt_indicator(self):
        indicator = WTIndicator(channel_length=10, average_length=21, wt_sma_length=4)
        
        # 1. Test calculation with DataFrame
        res = indicator.calculate(self.df)
        self.assertIsInstance(res, WTResult)
        self.assertIsInstance(res.series, np.ndarray)
        
        # 2. Test legacy method
        res_legacy = indicator.calculate_wt(self.candles_list)
        self.assertEqual(res.current, res_legacy.current)
        self.assertEqual(res.smoothing, res_legacy.smoothing)

        # 3. Test dict-like mapping
        self.assertEqual(res["wt1"], res.current)
        self.assertEqual(res["wt2"], res.smoothing)
        self.assertEqual(res.wt1, res.current)
        self.assertEqual(res.wt2, res.smoothing)

    def test_atr_indicator(self):
        indicator = ATRIndicator(atr_period=14)
        
        # 1. Test calculation with DataFrame
        res = indicator.calculate(self.df)
        self.assertIsInstance(res, ATRResult)
        self.assertIsInstance(res.series, np.ndarray)
        self.assertEqual(len(res.series), self.length)
        
        # 2. Test legacy/convenience method
        res_legacy = indicator.calculate_atr(self.candles_list)
        self.assertEqual(res.current_atr, res_legacy.current_atr)
        np.testing.assert_array_equal(res.series, res_legacy.series)

        # 3. Test dict-like mapping
        self.assertEqual(res["atr"], res.current)
        self.assertEqual(res["current_atr"], res.current_atr)
        self.assertEqual(res.current_atr, res.current)

        # 4. Verify against raw TA-Lib values
        expected_atr = talib.ATR(
            self.df["high"].to_numpy(),
            self.df["low"].to_numpy(),
            self.df["close"].to_numpy(),
            timeperiod=14
        )
        np.testing.assert_array_equal(res.series, expected_atr)

    def test_insufficient_data_handling(self):
        short_df = self.df.iloc[:5]  # Only 5 rows
        
        rsi_ind = RSIIndicator(rsi_period=14)
        res = rsi_ind.calculate(short_df)
        self.assertIsNone(res.current)
        self.assertIsNone(res.smoothing)
        self.assertTrue(np.isnan(res.series).all())
        self.assertEqual(len(res.series), 5)

        atr_ind = ATRIndicator(atr_period=14)
        res_atr = atr_ind.calculate(short_df)
        self.assertIsNone(res_atr.current_atr)
        self.assertTrue(np.isnan(res_atr.series).all())
        self.assertEqual(len(res_atr.series), 5)


if __name__ == "__main__":
    unittest.main()
