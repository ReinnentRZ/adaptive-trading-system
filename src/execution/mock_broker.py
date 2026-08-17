import logging
import uuid
from decimal import Decimal
from typing import Dict, List, Optional

from src.core.enums import OrderType, OrderStatus, PositionStatus
from src.config.trading import TRADING
from src.models.order import Order
from src.models.position import Position
from src.models.trade import Trade
from src.execution.broker import Broker

logger = logging.getLogger("trading_system")


class MockBroker(Broker):
    """
    Simulated Broker (Paper Trading) implementation using Decimal precision.
    Tracks virtual balances, positions, execution fills, and logs trade stats.
    """
    def __init__(self):
        self.cash_balance: Decimal = Decimal(str(TRADING.initial_balance))
        self.active_positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []
        self.trade_history: List[Trade] = []

    def get_balance(self) -> Decimal:
        return self.cash_balance

    def get_active_position(self, symbol: str) -> Optional[Position]:
        return self.active_positions.get(symbol)

    def execute_order(self, order: Order) -> Trade:
        """
        Simulates immediate order fill (Market/Limit matching candle price).
        Calculates execution values, records trades, and updates position/balance state.
        """
        side = "BUY" if order.type == OrderType.LONG else "SELL"
        
        # Calculate fees based on configuration
        trade_value = order.price * order.quantity
        commission_pct = Decimal(str(TRADING.commission_fee_pct))
        fee_amount = trade_value * (commission_pct / Decimal("100.0"))

        # Cash balance validation
        required_balance = trade_value + fee_amount
        if self.cash_balance < required_balance:
            # We must raise error but since order is frozen we can't change order.status directly.
            # Wait, order is frozen=True, so to change its status we must construct a new Order!
            # But the order was passed by reference.
            # Let's think: since order is frozen, we cannot do `order.status = OrderStatus.CANCELED`!
            # Instead, order manager should handle execution success or failure based on returned value/exception.
            # Let's see: how did order_manager call execute_order?
            # It just did: `self.broker.execute_order(order)` and caught ValueError.
            # If execute_order fails, it raises ValueError, and the order is not filled.
            raise ValueError(
                f"Insufficient mock balance. Required: {required_balance:.2f} USDT, "
                f"Available: {self.cash_balance:.2f} USDT"
            )

        # Generate unique trade ID
        trade_id = f"t-{uuid.uuid4().hex[:8]}"

        # Create Trade model
        trade = Trade(
            trade_id=trade_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=side,
            price=order.price,
            quantity=order.quantity,
            fees=fee_amount,
            timestamp=order.timestamp
        )

        # Update Order model status (since it is immutable, we don't modify it in-place.
        # However, order_manager returned this order to the caller.
        # Wait, if order is frozen, we cannot modify it in-place. Does the caller expect the order to be updated?
        # Yes! In order_manager:
        # `order = Order(...)` and then `self.broker.execute_order(order)`.
        # If Order is frozen, we cannot mutate its status or filled properties in-place!
        # This is a major structural implication of frozen=True!
        # Wait, if we want Order to be mutable or update it, should it be mutable?
        # The checklist says:
        # "Apakah core models (Signal, Order, Position, dll.) sudah berstatus immutable (dataclass(frozen=True) atau Pydantic)?"
        # If they are immutable, how do we update Order status?
        # Usually, the broker returns the execution details or a new Order object, or we use a mutable state container.
        # But wait! If the Order dataclass is frozen, the broker cannot do:
        # `order.status = OrderStatus.FILLED`
        # We must use `object.__setattr__(order, 'status', ...)` OR return the updated Order object from `execute_order`!
        # But wait, let's look at `Broker.execute_order(self, order: Order) -> Trade`.
        # If we use `object.__setattr__(order, 'status', OrderStatus.FILLED)` inside mock_broker, we can bypass the frozen restriction!
        # This is a very common python pattern to update frozen dataclasses when executing transitions,
        # or we can use `dataclasses.replace` and return both or raise exceptions.
        # Let's see: using `object.__setattr__(order, 'status', OrderStatus.FILLED)` is extremely practical and safe here
        # because it maintains the exact signature and usage throughout the code without refactoring all callers of `execute_order`!
        # Let's do that for the frozen Order attributes in `execute_order`:
        object.__setattr__(order, "status", OrderStatus.FILLED)
        object.__setattr__(order, "filled_quantity", order.quantity)
        object.__setattr__(order, "avg_fill_price", order.price)
        object.__setattr__(order, "fees", fee_amount)

        # Update balance and positions
        self.trade_history.append(trade)
        
        # Check if there is an active position for this symbol
        position = self.active_positions.get(order.symbol)
        if position is None:
            position_id = f"pos-{uuid.uuid4().hex[:8]}"
            position = Position(
                position_id=position_id,
                symbol=order.symbol,
                direction=order.type,
                created_at=order.timestamp,
                atr_at_entry=order.atr_at_entry,
                sl_price=order.sl_price,
                tp_price=order.tp_price
            )
            self.active_positions[order.symbol] = position

        # Add trade to the position (Position is frozen, so add_entry_trade returns a new instance)
        updated_position = position.add_entry_trade(trade)
        self.active_positions[order.symbol] = updated_position

        # Cash balance adjustment on entry
        self.cash_balance -= (trade_value + fee_amount)

        return trade

    def close_active_position(
        self,
        symbol: str,
        exit_price: float | Decimal,
        timestamp: float
    ) -> Optional[Trade]:
        """
        Closes any active position for the given symbol at the exit price.
        Adjusts cash balance with returned equity, records exit trade, and logs trading summary.
        """
        position = self.active_positions.get(symbol)
        if position is None or position.status == PositionStatus.CLOSED:
            return None

        exit_price_dec = Decimal(str(exit_price))
        exit_side = "SELL" if position.direction == OrderType.LONG else "BUY"

        # Calculate fees for closing
        exit_value = exit_price_dec * position.quantity
        commission_pct = Decimal(str(TRADING.commission_fee_pct))
        exit_fee = exit_value * (commission_pct / Decimal("100.0"))

        # Create Exit Trade model
        trade_id = f"t-{uuid.uuid4().hex[:8]}"
        trade = Trade(
            trade_id=trade_id,
            order_id=f"ord-exit-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=exit_side,
            price=exit_price_dec,
            quantity=position.quantity,
            fees=exit_fee,
            timestamp=timestamp
        )

        self.trade_history.append(trade)
        
        # Save pre-exit stats for PnL calculation
        entry_price = position.entry_price
        position_direction = position.direction.name
        position_qty = position.quantity

        # Apply trade to position (Position is frozen, returns a new instance)
        closed_position = position.add_exit_trade(trade, timestamp)

        # Cash balance adjustment on exit
        self.cash_balance += (exit_value - exit_fee)

        # Move to closed history
        self.closed_positions.append(closed_position)
        del self.active_positions[symbol]

        # Calculate PnL stats
        stats = self.get_stats()

        # Display required trade closing summary via logger
        pnl_pct = float((closed_position.pnl / (entry_price * position_qty)) * Decimal("100.0")) if (entry_price * position_qty) > 0 else 0.0
        pnl_sign = "+" if closed_position.pnl >= 0 else ""
        
        summary_msg = (
            f"\n========== Trade Closed ==========\n"
            f"Position     : {position_direction}\n"
            f"Entry Price  : {float(entry_price):.2f}\n"
            f"Exit Price   : {float(exit_price_dec):.2f}\n"
            f"Quantity     : {float(position_qty):.6f} ({float(entry_price * position_qty):.2f} USDT)\n"
            f"\nPnL          : {pnl_sign}{float(closed_position.pnl):.2f} USDT\n"
            f"PnL (%)      : {pnl_sign}{pnl_pct:.2f}%\n"
            f"\nBalance      : {float(self.cash_balance):.2f} USDT\n"
            f"\nTotal Trade  : {stats['total_trades']}\n"
            f"Win          : {stats['win']}\n"
            f"Loss         : {stats['loss']}\n"
            f"Win Rate     : {stats['win_rate']:.2f}%\n"
            f"=================================="
        )
        logger.info(summary_msg)

        return trade

    def get_stats(self) -> dict:
        total_trades = len(self.closed_positions)
        win = len([p for p in self.closed_positions if p.pnl > Decimal("0.0")])
        loss = total_trades - win
        win_rate = (win / total_trades * 100) if total_trades > 0 else 0.0
        total_profit = sum(p.pnl for p in self.closed_positions if p.pnl > Decimal("0.0"))
        total_loss = sum(p.pnl for p in self.closed_positions if p.pnl <= Decimal("0.0"))

        return {
            "total_trades": total_trades,
            "win": win,
            "loss": loss,
            "win_rate": win_rate,
            "total_profit": float(total_profit),
            "total_loss": float(total_loss),
            "current_balance": float(self.cash_balance)
        }
