from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import List, Optional
from src.core.enums import OrderType, PositionStatus
from src.models.trade import Trade


@dataclass(frozen=True)
class Position:
    """
    Immutable domain model representing active market exposure.
    Mutation methods return a new instance of the Position.
    """
    position_id: str
    symbol: str
    direction: OrderType  # LONG or SHORT
    status: PositionStatus = PositionStatus.OPEN
    quantity: Decimal = Decimal("0.0")  # Current active position quantity (size)
    entry_price: Decimal = Decimal("0.0")  # Average entry price (cost basis)
    exit_price: Optional[Decimal] = None  # Average exit price (once position is closed)
    pnl: Decimal = Decimal("0.0")  # Realized Profit/Loss
    fees: Decimal = Decimal("0.0")  # Total accumulated fees for all entry and exit trades
    created_at: float = 0.0  # Timestamp when the position was opened
    closed_at: Optional[float] = None  # Timestamp when the position was fully closed
    entry_trades: List[Trade] = field(default_factory=list)  # List of trades that opened/increased this position
    exit_trades: List[Trade] = field(default_factory=list)  # List of trades that closed/reduced this position
    atr_at_entry: Optional[Decimal] = None
    sl_price: Optional[Decimal] = None
    tp_price: Optional[Decimal] = None
    bars_held: int = 0
    max_holding_bars: int = 12

    def increment_bars_held(self) -> "Position":
        """Increments the count of completed candles held for this position."""
        return replace(self, bars_held=self.bars_held + 1)

    def check_time_barrier(self) -> bool:
        """Returns True if the position has reached or exceeded max holding bars."""
        return self.bars_held >= self.max_holding_bars

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """
        Calculates unrealized PnL based on the current market price using Decimal precision.
        """
        if self.status == PositionStatus.CLOSED or self.quantity == Decimal("0.0"):
            return Decimal("0.0")

        if self.direction == OrderType.LONG:
            return (current_price - self.entry_price) * self.quantity
        elif self.direction == OrderType.SHORT:
            return (self.entry_price - current_price) * self.quantity
        return Decimal("0.0")

    def add_entry_trade(self, trade: Trade) -> "Position":
        """
        Adds an entry trade to open or increase the position, returning a new updated Position instance.
        """
        if self.status == PositionStatus.CLOSED:
            raise ValueError("Cannot add entry trade to a closed position.")
        
        new_entry_trades = self.entry_trades + [trade]
        new_fees = self.fees + trade.fees
        new_qty = self.quantity + trade.quantity

        new_entry_price = self.entry_price
        if new_qty > Decimal("0.0"):
            new_entry_price = (
                (self.entry_price * self.quantity) + (trade.price * trade.quantity)
            ) / new_qty

        return replace(
            self,
            entry_trades=new_entry_trades,
            fees=new_fees,
            quantity=new_qty,
            entry_price=new_entry_price
        )

    def add_exit_trade(self, trade: Trade, timestamp: float) -> "Position":
        """
        Adds an exit trade to reduce or close the position, returning a new updated Position instance.
        """
        if self.status == PositionStatus.CLOSED:
            raise ValueError("Cannot add exit trade to a closed position.")
        if trade.quantity > self.quantity:
            raise ValueError(
                f"Exit trade quantity ({trade.quantity}) cannot exceed current position quantity ({self.quantity})."
            )

        new_exit_trades = self.exit_trades + [trade]
        new_fees = self.fees + trade.fees

        # Calculate realized PnL for this specific exit trade transaction
        trade_pnl = Decimal("0.0")
        if self.direction == OrderType.LONG:
            trade_pnl = (trade.price - self.entry_price) * trade.quantity
        elif self.direction == OrderType.SHORT:
            trade_pnl = (self.entry_price - trade.price) * trade.quantity

        new_pnl = self.pnl + trade_pnl
        new_qty = max(Decimal("0.0"), self.quantity - trade.quantity)

        # Calculate weighted average exit price
        total_exit_quantity = sum(t.quantity for t in new_exit_trades)
        new_exit_price = self.exit_price
        if total_exit_quantity > Decimal("0.0"):
            total_exit_value = sum(t.price * t.quantity for t in new_exit_trades)
            new_exit_price = total_exit_value / total_exit_quantity

        # Handle position closing
        new_status = self.status
        new_closed_at = self.closed_at
        if new_qty == Decimal("0.0"):
            new_status = PositionStatus.CLOSED
            new_closed_at = timestamp

        return replace(
            self,
            exit_trades=new_exit_trades,
            fees=new_fees,
            pnl=new_pnl,
            quantity=new_qty,
            exit_price=new_exit_price,
            status=new_status,
            closed_at=new_closed_at
        )
