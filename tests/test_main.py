import json
from pathlib import Path
import tempfile
import unittest
from decimal import Decimal

from src.core.enums import OrderType, OrderStatus
from src.core.signal import Signal
from src.execution.mock_broker import MockBroker
from src.execution.order_manager import OrderManager
from main import save_bot_state


class TestMainOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.state_file = self.test_dir / "bot_state.json"

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


if __name__ == "__main__":
    unittest.main()
