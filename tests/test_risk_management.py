import unittest
from decimal import Decimal
from unittest.mock import patch

from src.core.enums import OrderType, OrderStatus, PositionStatus
from src.core.signal import Signal
from src.config.trading import TRADING
from src.execution.mock_broker import MockBroker
from src.execution.order_manager import OrderManager


class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        self.broker = MockBroker()
        self.order_manager = OrderManager(self.broker)
        self.symbol = "SOLUSDT"

    def test_atr_sl_tp_calculation_long(self):
        # Setup: entry price = 100.00, ATR = 2.00
        # Formula for LONG:
        # SL = Entry Price - (sl_multiplier * ATR) = 100.00 - (1.5 * 2.00) = 97.00
        # TP = Entry Price + (tp_multiplier * ATR) = 100.00 + (4.5 * 2.00) = 109.00
        
        signal = Signal(
            type=OrderType.LONG,
            candle_close=Decimal("100.00"),
            confidence=Decimal("80.00"),
            raw_vote=6
        )
        current_atr = Decimal("2.00")
        
        order = self.order_manager.process_signal(
            signal=signal,
            symbol=self.symbol,
            timestamp=1629000000.0,
            current_atr=current_atr
        )
        
        # Verify order attributes
        self.assertIsNotNone(order)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.atr_at_entry, current_atr)
        self.assertEqual(order.sl_price, Decimal("97.00"))
        self.assertEqual(order.tp_price, Decimal("109.00"))

        # Verify active position attributes in the broker
        position = self.broker.get_active_position(self.symbol)
        self.assertIsNotNone(position)
        self.assertEqual(position.status, PositionStatus.OPEN)
        self.assertEqual(position.atr_at_entry, current_atr)
        self.assertEqual(position.sl_price, Decimal("97.00"))
        self.assertEqual(position.tp_price, Decimal("109.00"))

    def test_atr_sl_tp_calculation_short(self):
        # Setup: entry price = 100.00, ATR = 2.00
        # Formula for SHORT:
        # SL = Entry Price + (sl_multiplier * ATR) = 100.00 + (1.5 * 2.00) = 103.00
        # TP = Entry Price - (tp_multiplier * ATR) = 100.00 - (4.5 * 2.00) = 91.00
        
        signal = Signal(
            type=OrderType.SHORT,
            candle_close=Decimal("100.00"),
            confidence=Decimal("80.00"),
            raw_vote=-6
        )
        current_atr = Decimal("2.00")
        
        order = self.order_manager.process_signal(
            signal=signal,
            symbol=self.symbol,
            timestamp=1629000000.0,
            current_atr=current_atr
        )
        
        # Verify order attributes
        self.assertIsNotNone(order)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.atr_at_entry, current_atr)
        self.assertEqual(order.sl_price, Decimal("103.00"))
        self.assertEqual(order.tp_price, Decimal("91.00"))

        # Verify active position attributes in the broker
        position = self.broker.get_active_position(self.symbol)
        self.assertIsNotNone(position)
        self.assertEqual(position.status, PositionStatus.OPEN)
        self.assertEqual(position.atr_at_entry, current_atr)
        self.assertEqual(position.sl_price, Decimal("103.00"))
        self.assertEqual(position.tp_price, Decimal("91.00"))

    def test_atr_sl_trigger_long(self):
        signal = Signal(
            type=OrderType.LONG,
            candle_close=Decimal("100.00"),
            confidence=Decimal("80.00"),
            raw_vote=6
        )
        current_atr = Decimal("2.00")
        
        self.order_manager.process_signal(
            signal=signal,
            symbol=self.symbol,
            timestamp=1629000000.0,
            current_atr=current_atr
        )
        
        # Price goes down below SL (97.00)
        self.order_manager.update_market_price(self.symbol, current_price=96.99, timestamp=1629000010.0)
        
        # Position should be closed
        position = self.broker.get_active_position(self.symbol)
        self.assertIsNone(position)
        self.assertEqual(len(self.broker.closed_positions), 1)
        self.assertEqual(self.broker.closed_positions[0].status, PositionStatus.CLOSED)

    def test_atr_tp_trigger_long(self):
        signal = Signal(
            type=OrderType.LONG,
            candle_close=Decimal("100.00"),
            confidence=Decimal("80.00"),
            raw_vote=6
        )
        current_atr = Decimal("2.00")
        
        self.order_manager.process_signal(
            signal=signal,
            symbol=self.symbol,
            timestamp=1629000000.0,
            current_atr=current_atr
        )
        
        # Price goes up above TP (109.00)
        self.order_manager.update_market_price(self.symbol, current_price=109.01, timestamp=1629000010.0)
        
        # Position should be closed
        position = self.broker.get_active_position(self.symbol)
        self.assertIsNone(position)
        self.assertEqual(len(self.broker.closed_positions), 1)
        self.assertEqual(self.broker.closed_positions[0].status, PositionStatus.CLOSED)

    def test_adaptive_risk_calculator(self):
        import pandas as pd
        import numpy as np
        from src.strategies.risk_engine import AdaptiveRiskCalculator, AdaptiveRiskParams

        # Generate synthetic data with 100 bars (must be >= 65 to populate SMA of ATR)
        length = 100
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.normal(0, 1, length))
        high = close + np.abs(np.random.normal(1, 0.5, length))
        low = close - np.abs(np.random.normal(1, 0.5, length))
        open_ = close + np.random.normal(0, 0.5, length)
        volume = np.random.uniform(100, 1000, length)
        
        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })

        # Test LONG calculation
        params = AdaptiveRiskCalculator.calculate_risk_params(
            data=df,
            entry_price=Decimal("100.00"),
            direction=OrderType.LONG,
            symbol="SOLUSDT"
        )

        self.assertIsInstance(params, AdaptiveRiskParams)
        self.assertIn(params.market_regime, ["TRENDING", "RANGING", "HIGH_VOLATILITY"])
        self.assertTrue(params.calculated_atr > 0)
        self.assertTrue(params.sl_price < 100.0)
        self.assertTrue(params.tp_price > 100.0)
        self.assertTrue(1.5 <= params.target_rr_ratio <= 4.0)

        # Test OrderManager integration
        signal = Signal(
            type=OrderType.LONG,
            candle_close=Decimal("100.00"),
            confidence=Decimal("80.00"),
            raw_vote=6
        )
        
        order = self.order_manager.process_signal(
            signal=signal,
            symbol=self.symbol,
            timestamp=1629000000.0,
            adaptive_risk_params=params
        )
        
        self.assertEqual(order.sl_price, Decimal(str(params.sl_price)))
        self.assertEqual(order.tp_price, Decimal(str(params.tp_price)))
        self.assertEqual(order.atr_at_entry, Decimal(str(params.calculated_atr)))


if __name__ == "__main__":
    unittest.main()
