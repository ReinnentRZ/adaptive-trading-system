import uuid
from typing import Dict, List, Optional
from src.core.enums import OrderType, OrderStatus, PositionStatus
from src.config.trading import TRADING
from src.models.order import Order
from src.models.position import Position
from src.models.trade import Trade
from src.execution.broker import Broker

class MockBroker(Broker):
    """
    Simulated Broker (Paper Trading) implementation.
    Tracks virtual balances, positions, execution fills, and logs trade stats.
    """
    def __init__(self):
        self.cash_balance: float = TRADING.initial_balance
        self.active_positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []
        self.trade_history: List[Trade] = []

    def get_balance(self) -> float:
        return self.cash_balance

    def get_active_position(self, symbol: str) -> Optional[Position]:
        return self.active_positions.get(symbol)

    def execute_order(self, order: Order) -> Trade:
        """
        Simulates immediate order fill (Market/Limit matching candle price).
        Calculates execution values, records trades, and updates position/balance state.
        """
        # Calculate mechanical trade side
        # For opening: BUY if LONG, SELL if SHORT
        side = "BUY" if order.type == OrderType.LONG else "SELL"
        
        # Calculate fees based on configuration
        trade_value = order.price * order.quantity
        fee_amount = trade_value * (TRADING.commission_fee_pct / 100.0)

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

        # Update Order model status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = order.price
        order.fees = fee_amount

        # Update balance and positions
        self.trade_history.append(trade)
        
        # Check if there is an active position for this symbol
        position = self.active_positions.get(order.symbol)
        if position is None:
            # Open new position
            position_id = f"pos-{uuid.uuid4().hex[:8]}"
            position = Position(
                position_id=position_id,
                symbol=order.symbol,
                direction=order.type,
                created_at=order.timestamp
            )
            self.active_positions[order.symbol] = position

        # Add trade to the position
        position.add_entry_trade(trade)

        # Cash balance adjustment on entry
        # Balance decreases by the capital cost and the entry fee
        self.cash_balance -= (trade_value + fee_amount)

        return trade

    def close_active_position(self, symbol: str, exit_price: float, timestamp: float) -> Optional[Trade]:
        """
        Closes any active position for the given symbol at the exit price.
        Adjusts cash balance with returned equity, records exit trade, and prints trading summary.
        """
        position = self.active_positions.get(symbol)
        if position is None or position.status == PositionStatus.CLOSED:
            return None

        # Exit order type direction is opposite of current position direction
        exit_side = "SELL" if position.direction == OrderType.LONG else "BUY"

        # Calculate fees for closing
        exit_value = exit_price * position.quantity
        exit_fee = exit_value * (TRADING.commission_fee_pct / 100.0)

        # Create Exit Trade model
        trade_id = f"t-{uuid.uuid4().hex[:8]}"
        trade = Trade(
            trade_id=trade_id,
            order_id=f"ord-exit-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=exit_side,
            price=exit_price,
            quantity=position.quantity,
            fees=exit_fee,
            timestamp=timestamp
        )

        self.trade_history.append(trade)
        
        # Save pre-exit stats for PnL calculation
        entry_price = position.entry_price
        position_direction = position.direction.name
        position_qty = position.quantity

        # Apply trade to position (this sets status to CLOSED)
        position.add_exit_trade(trade, timestamp)

        # Cash balance adjustment on exit
        # Balance increases by position value and decreases by exit fee
        self.cash_balance += (exit_value - exit_fee)

        # Move to closed history
        self.closed_positions.append(position)
        del self.active_positions[symbol]

        # Calculate PnL stats
        stats = self.get_stats()

        # Display required trade closing summary
        pnl_pct = (position.pnl / (entry_price * position_qty)) * 100.0 if (entry_price * position_qty) > 0 else 0.0
        pnl_sign = "+" if position.pnl >= 0 else ""
        
        print("\n" + "=" * 10 + " Trade Closed " + "=" * 10)
        print(f"Position     : {position_direction}")
        print(f"Entry Price  : {entry_price:.2f}")
        print(f"Exit Price   : {exit_price:.2f}")
        print(f"Quantity     : {position_qty:.6f} ({entry_price * position_qty:.2f} USDT)")
        print(f"\nPnL          : {pnl_sign}{position.pnl:.2f} USDT")
        print(f"PnL (%)      : {pnl_sign}{pnl_pct:.2f}%")
        print(f"\nBalance      : {self.cash_balance:.2f} USDT")
        print(f"\nTotal Trade  : {stats['total_trades']}")
        print(f"Win          : {stats['win']}")
        print(f"Loss         : {stats['loss']}")
        print(f"Win Rate     : {stats['win_rate']:.2f}%")
        print("=" * 34 + "\n")

        return trade

    def get_stats(self) -> dict:
        total_trades = len(self.closed_positions)
        win = len([p for p in self.closed_positions if p.pnl > 0])
        loss = total_trades - win
        win_rate = (win / total_trades * 100) if total_trades > 0 else 0.0
        total_profit = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
        total_loss = sum(p.pnl for p in self.closed_positions if p.pnl <= 0)

        return {
            "total_trades": total_trades,
            "win": win,
            "loss": loss,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "current_balance": self.cash_balance
        }
