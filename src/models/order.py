from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from src.core.enums import OrderType, OrderStatus
from src.core.signal import Signal


@dataclass(frozen=True)
class Order:
    """
    Immutable domain model representing a transaction request submitted to the broker/exchange.
    All financial values use Decimal for numeric precision.
    """
    order_id: str
    symbol: str
    type: OrderType
    quantity: Decimal
    price: Decimal
    status: OrderStatus
    timestamp: float
    signal: Optional[Signal] = None
    filled_quantity: Decimal = Decimal("0.0")
    avg_fill_price: Decimal = Decimal("0.0")
    fees: Decimal = Decimal("0.0")

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> Decimal:
        return max(Decimal("0.0"), self.quantity - self.filled_quantity)
