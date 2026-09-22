"""
Unit tests for the Unified Live Execution Bot and Orchestrator.
"""

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.config import config, AppConfig, MarketConfig, TradingConfig, StrategyConfig
from src.core.enums import OrderType, OrderStatus, PositionStatus
from src.core.signal import Signal
from src.execution.live_bot import LiveTradingBot
from src.execution.mock_broker import MockBroker
from src.execution.order_manager import OrderManager
from main import save_bot_state


class TestMainOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.state_file = self.test_dir / "bot_state.json"
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_bot_state_empty_positions(self) -> None:
        broker = MockBroker()
        save_bot_state(
            state_file=self.state_file,
            symbol="BTCUSDT",
            timeframe="5m",
            bot_mode="paper",
            broker=broker,
            initial_capital=100.0,
            model_loaded=True,
            threshold=0.44,
        )

        self.assertTrue(self.state_file.exists())
        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["symbol"], "BTCUSDT")
        self.assertEqual(data["timeframe"], "5m")
        self.assertEqual(data["bot_mode"], "paper")
        self.assertEqual(data["initial_capital"], 100.0)
        self.assertEqual(data["meta_threshold"], 0.44)
        self.assertTrue(data["model_loaded"])
        self.assertIsNone(data["active_position"])
        self.assertEqual(data["status"], "STOPPED")

    def test_save_bot_state_with_active_position(self) -> None:
        broker = MockBroker()
        order_mgr = OrderManager(broker)

        signal = Signal(
            type=OrderType.LONG,
            candle_close=Decimal("50000.00"),
            confidence=Decimal("85.00"),
            raw_vote=6
        )
        order_mgr.process_signal(
            signal=signal,
            symbol="BTCUSDT",
            timestamp=1700000000.0,
            current_atr=Decimal("500.00")
        )

        save_bot_state(
            state_file=self.state_file,
            symbol="BTCUSDT",
            timeframe="5m",
            bot_mode="paper",
            broker=broker,
            initial_capital=100.0,
            model_loaded=True,
            threshold=0.44,
        )

        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsNotNone(data["active_position"])
        self.assertEqual(data["active_position"]["symbol"], "BTCUSDT")
        self.assertEqual(data["active_position"]["direction"], "LONG")
        self.assertEqual(data["active_position"]["entry_price"], 50000.0)

    def test_live_bot_regime_dry_run_and_state_persistence(self) -> None:
        mock_exchange = MagicMock()
        # Mock 100 OHLCV candles
        now_ts = 1700000000000
        candles = []
        for i in range(250):
            candles.append([now_ts + i * 3600000, 50000.0 + i, 50100.0 + i, 49900.0 + i, 50050.0 + i, 100.0])
        mock_exchange.fetch_ohlcv.return_value = candles

        bot = LiveTradingBot(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            timeframe="1h",
            state_file=self.state_file,
            is_testnet=True,
        )

        res = bot.run_once(dry_run=True)
        self.assertIn("close", res)
        self.assertIn("ema_200", res)
        self.assertIn("risk_brackets", res)
        self.assertTrue(self.state_file.exists())

        # Verify state reloaded
        bot2 = LiveTradingBot(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            timeframe="1h",
            state_file=self.state_file,
            is_testnet=True,
        )
        self.assertEqual(bot2.last_state_id, res["current_state"])

    def test_live_bot_pilar3_scaling_out_and_stop_loss(self) -> None:
        from src.execution.live_bot import ActivePosition
        mock_exchange = MagicMock()
        bot = LiveTradingBot(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            timeframe="1h",
            state_file=self.state_file,
            is_testnet=True,
        )

        # Setup active position: Entry 100,000, TP1 100,800, TP2 101,200, SL 99,300, BE 100,250
        bot.active_position = ActivePosition(
            position_id="POS-001",
            symbol="BTC/USDT",
            entry_price=100000.0,
            initial_quantity=0.001,
            remaining_quantity=0.001,
            tp1_price=100800.0,
            tp2_price=101200.0,
            sl_price=99300.0,
            be_price=100250.0,
            tp1_hit=False,
        )

        # 1. Price hits TP1 (100,850)
        bot.monitor_active_position(current_price=100850.0, current_regime_state=0)
        self.assertTrue(bot.active_position.tp1_hit)
        self.assertEqual(bot.active_position.sl_price, 100250.0)  # Locked to BE
        self.assertAlmostEqual(bot.active_position.remaining_quantity, 0.0005, places=5)
        self.assertTrue(mock_exchange.create_market_sell_order.called)

        # 2. Price hits TP2 (101,250) -> Sells remaining 50%
        bot.monitor_active_position(current_price=101250.0, current_regime_state=0)
        self.assertIsNone(bot.active_position)  # Closed
        self.assertEqual(len(bot.completed_trades), 1)
        self.assertEqual(bot.completed_trades[0]["exit_reason"], "TAKE_PROFIT_2")

    def test_live_bot_time_barrier_expiry(self) -> None:
        broker = MockBroker()
        order_mgr = OrderManager(broker)

        # Place initial LONG position
        signal = Signal(
            type=OrderType.LONG,
            candle_close=Decimal("50000.00"),
            confidence=Decimal("85.00"),
            raw_vote=6,
        )
        order_mgr.process_signal(
            signal=signal,
            symbol="BTCUSDT",
            timestamp=1700000000.0,
            current_atr=Decimal("500.00"),
        )
        self.assertIsNotNone(broker.get_active_position("BTCUSDT"))

        # Simulate 12 candle close events
        for i in range(1, 13):
            order_mgr.on_candle_close(
                symbol="BTCUSDT",
                close_price=50100.0,
                timestamp=1700000000.0 + i * 300,
            )

        # After 12 candles, position should be closed due to time barrier
        self.assertIsNone(broker.get_active_position("BTCUSDT"))
        self.assertEqual(len(broker.closed_positions), 1)
        self.assertEqual(broker.closed_positions[0].status, PositionStatus.CLOSED)


if __name__ == "__main__":
    unittest.main()
