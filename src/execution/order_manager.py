import uuid
from typing import Optional
from src.core.enums import OrderType, OrderStatus
from src.core.signal import Signal
from src.config.trading import TRADING
from src.config.strategy import STRATEGY
from src.models.order import Order
from src.models.position import Position
from src.execution.broker import Broker

class OrderManager:
    """
    Orchestrator responsible for managing signals, validating risk rules,
    creating orders, and monitoring active positions for Stop Loss and Take Profit breaches.
    Uses Dependency Injection to remain decoupled from specific Broker implementations.
    """
    def __init__(self, broker: Broker):
        self.broker: Broker = broker

    def process_signal(self, signal: Signal, symbol: str, timestamp: float) -> Optional[Order]:
        """
        Processes an incoming trading Signal.
        Decides whether to execute position entries, closures, or reversals.
        """
        # Validate confidence threshold
        if signal.confidence < STRATEGY.confidence_threshold:
            print(f"[OrderManager] Signal ignored: Confidence ({signal.confidence}%) is below threshold ({STRATEGY.confidence_threshold}%).")
            return None

        active_position = self.broker.get_active_position(symbol)

        if active_position is not None:
            # If we have an active position in the opposite direction of the signal, we close it (position reversal)
            if active_position.direction != signal.type:
                print(f"[OrderManager] Signal direction ({signal.type.name}) opposite to active position ({active_position.direction.name}). Closing position.")
                self.broker.close_active_position(symbol, exit_price=signal.candle_close, timestamp=timestamp)
                
                # After closing the old position, open a new one in the direction of the signal
                return self._create_and_execute_order(signal, symbol, timestamp)
            else:
                # Same direction signal - ignore to avoid over-exposure (or scaling in can be enabled if desired)
                print(f"[OrderManager] Hold existing {active_position.direction.name} position. New signal in same direction ignored.")
                return None
        else:
            # No active position exists, create a new entry order
            return self._create_and_execute_order(signal, symbol, timestamp)

    def update_market_price(self, symbol: str, current_price: float, timestamp: float) -> None:
        """
        Monitors active positions against Stop Loss and Take Profit thresholds.
        Closes position if risk bounds are breached.
        """
        position = self.broker.get_active_position(symbol)
        if position is None:
            return

        # Calculate PnL percentage based on average entry price
        if position.direction == OrderType.LONG:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100.0
        elif position.direction == OrderType.SHORT:
            pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100.0
        else:
            return

        # Check Stop Loss (SL)
        if pnl_pct <= -TRADING.stop_loss_pct:
            print(f"[OrderManager] Stop Loss triggered for {symbol} | Price: {current_price:.2f} | PnL: {pnl_pct:.2f}%")
            self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)
        
        # Check Take Profit (TP)
        elif pnl_pct >= TRADING.take_profit_pct:
            print(f"[OrderManager] Take Profit triggered for {symbol} | Price: {current_price:.2f} | PnL: {pnl_pct:.2f}%")
            self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)

    def _create_and_execute_order(self, signal: Signal, symbol: str, timestamp: float) -> Order:
        """
        Helper method to create and send an order to the broker.
        """
        # Calculate order quantity based on allocated risk capital in USDT
        quantity = TRADING.trade_quantity_usdt / signal.candle_close

        order_id = f"ord-{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            type=signal.type,
            quantity=quantity,
            price=signal.candle_close,
            status=OrderStatus.PENDING,
            timestamp=timestamp,
            signal=signal
        )

        print(f"[OrderManager] Placing {order.type.name} entry order for {symbol} | Qty: {order.quantity:.6f} | Price: {order.price:.2f}")
        self.broker.execute_order(order)
        return order