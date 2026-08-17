import unittest
from dataclasses import FrozenInstanceError

from src.config import config, MarketConfig, AppConfig


class TestConfiguration(unittest.TestCase):
    def test_config_immutability(self):
        # Verify that AppConfig is frozen (immutable)
        with self.assertRaises(FrozenInstanceError):
            config.market.symbol = "BTCUSDT"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            config.trading.stop_loss_pct = 2.0  # type: ignore

    def test_market_computed_properties(self):
        # Default is SOLUSDT, 1m
        market_cfg = config.market
        self.assertEqual(market_cfg.symbol_lower, "solusdt")
        self.assertEqual(market_cfg.base_asset, "SOL")
        self.assertEqual(market_cfg.quote_asset, "USDT")
        self.assertEqual(market_cfg.ws_stream_name, "solusdt@kline_1m")

    def test_computed_properties_custom(self):
        # Custom instantiation to verify dynamic computations
        btc_market = MarketConfig(symbol="BTCUSDT", interval="5m")
        self.assertEqual(btc_market.symbol_lower, "btcusdt")
        self.assertEqual(btc_market.base_asset, "BTC")
        self.assertEqual(btc_market.quote_asset, "USDT")
        self.assertEqual(btc_market.ws_stream_name, "btcusdt@kline_5m")

        eth_busd = MarketConfig(symbol="ETHBUSD", interval="1h")
        self.assertEqual(eth_busd.symbol_lower, "ethbusd")
        self.assertEqual(eth_busd.base_asset, "ETH")
        self.assertEqual(eth_busd.quote_asset, "BUSD")
        self.assertEqual(eth_busd.ws_stream_name, "ethbusd@kline_1h")

    def test_validation_checks(self):
        # 1. Invalid symbol (lowercase)
        with self.assertRaises(ValueError) as ctx:
            MarketConfig(symbol="solusdt")
        self.assertIn("symbol must be an uppercase string", str(ctx.exception))

        # 2. Invalid symbol (spaces)
        with self.assertRaises(ValueError) as ctx:
            MarketConfig(symbol="SOL USDT")
        self.assertIn("symbol must be an uppercase string", str(ctx.exception))

        # 3. Invalid trade quantity (zero or negative)
        from src.config import TradingConfig
        with self.assertRaises(ValueError) as ctx:
            TradingConfig(trade_quantity_usdt=0.0)
        self.assertIn("trade_quantity_usdt must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            TradingConfig(trade_quantity_usdt=-10.0)
        self.assertIn("trade_quantity_usdt must be positive", str(ctx.exception))

        # 4. Invalid confidence threshold
        from src.config import StrategyConfig
        with self.assertRaises(ValueError) as ctx:
            StrategyConfig(confidence_threshold=105.0)
        self.assertIn("confidence_threshold must be between 0.0 and 100.0", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            StrategyConfig(confidence_threshold=-5.0)
        self.assertIn("confidence_threshold must be between 0.0 and 100.0", str(ctx.exception))

    def test_api_key_masking(self):
        from src.config import APIConfig
        api_cfg = APIConfig(api_key="1234567890abcdef", api_secret="my_super_secret_key")
        repr_str = repr(api_cfg)
        
        # Verify secret is redacted completely
        self.assertNotIn("my_super_secret_key", repr_str)
        self.assertIn("***REDACTED***", repr_str)
        
        # Verify key is masked
        self.assertIn("1234***cdef", repr_str)


if __name__ == "__main__":
    unittest.main()
