from dataclasses import dataclass, field
from typing import List, Optional
from src.core.enums import OrderType, PositionStatus
from src.models.trade import Trade

@dataclass
class Position:
    """
    Domain model representing an active market exposure (holding) for a specific symbol.
    A position tracks entries and exits (trades), average costs, realized PnL, and fees.
    """
    position_id: str
    symbol: str
    direction: OrderType  # LONG or SHORT
    status: PositionStatus = PositionStatus.OPEN
    quantity: float = 0.0  # Current active position quantity (size)
    entry_price: float = 0.0  # Average entry price (cost basis)
    exit_price: Optional[float] = None  # Average exit price (once position is closed or scaled out)
    pnl: float = 0.0  # Realized Profit/Loss
    fees: float = 0.0  # Total accumulated fees for all entry and exit trades
    created_at: float = 0.0  # Timestamp when the position was opened
    closed_at: Optional[float] = None  # Timestamp when the position was fully closed
    entry_trades: List[Trade] = field(default_factory=list)  # List of trades that opened/increased this position
    exit_trades: List[Trade] = field(default_factory=list)  # List of trades that closed/reduced this position

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculates unrealized PnL based on the current market price.
        Unrealized PnL is positive if the price moves in the position's favor,
        and negative if it moves against it.
        """
        if self.status == PositionStatus.CLOSED or self.quantity == 0:
            return 0.0

        if self.direction == OrderType.LONG:
            return (current_price - self.entry_price) * self.quantity
        elif self.direction == OrderType.SHORT:
            return (self.entry_price - current_price) * self.quantity
        return 0.0

    def add_entry_trade(self, trade: Trade) -> None:
        """
        Adds an entry trade to open or increase (scale in) the position.
        Recalculates average entry price, total quantity, and accumulated fees.
        """
        if self.status == PositionStatus.CLOSED:
            raise ValueError("Cannot add entry trade to a closed position.")
        
        self.entry_trades.append(trade)
        self.fees += trade.fees

        # Calculate weighted average entry price
        new_quantity = self.quantity + trade.quantity
        if new_quantity > 0:
            self.entry_price = (
                (self.entry_price * self.quantity) + (trade.price * trade.quantity)
            ) / new_quantity
            self.quantity = new_quantity

    def add_exit_trade(self, trade: Trade, timestamp: float) -> None:
        """
        Adds an exit trade to reduce (scale out) or close the position.
        Calculates realized PnL, updates average exit price, reduces quantity, and adds fees.
        If quantity reaches 0, the position is automatically closed.
        """
        if self.status == PositionStatus.CLOSED:
            raise ValueError("Cannot add exit trade to a closed position.")
        if trade.quantity > self.quantity:
            raise ValueError(
                f"Exit trade quantity ({trade.quantity}) cannot exceed current position quantity ({self.quantity})."
            )

        self.exit_trades.append(trade)
        self.fees += trade.fees

        # Calculate realized PnL for this specific exit trade transaction
        trade_pnl = 0.0
        if self.direction == OrderType.LONG:
            trade_pnl = (trade.price - self.entry_price) * trade.quantity
        elif self.direction == OrderType.SHORT:
            trade_pnl = (self.entry_price - trade.price) * trade.quantity

        self.pnl += trade_pnl

        # Calculate weighted average exit price
        total_exit_quantity = sum(t.quantity for t in self.exit_trades)
        if total_exit_quantity > 0:
            total_exit_value = sum(t.price * t.quantity for t in self.exit_trades)
            self.exit_price = total_exit_value / total_exit_quantity

        # Update remaining position size
        self.quantity = max(0.0, self.quantity - trade.quantity)

        # Handle position closing
        if self.quantity == 0.0:
            self.status = PositionStatus.CLOSED
            self.closed_at = timestamp
