from dataclasses import dataclass
from typing import Optional
from src.core.enums import OrderType, OrderStatus
from src.core.signal import Signal

@dataclass
class Order:
    """
    Domain model representing a transaction request submitted to the broker/exchange.
    Contains reference to the parent Signal that triggered this order for performance tracking.
    """
    order_id: str
    symbol: str
    type: OrderType  # LONG or SHORT
    quantity: float
    price: float  # Intended execution price (limit price)
    status: OrderStatus
    timestamp: float
    signal: Optional[Signal] = None  # Reference to the Signal that triggered this order
    filled_quantity: float = 0.0  # Quantity filled so far (supports partial fills)
    avg_fill_price: float = 0.0  # Average price of all executed fills (trades) for this order
    fees: float = 0.0  # Cumulative fees incurred by this order

    @property
    def is_filled(self) -> bool:
        """Checks if the order has been fully executed."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> float:
        """Calculates the remaining quantity to be filled."""
        return max(0.0, self.quantity - self.filled_quantity)
