from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Trade:
    """
    Immutable domain model representing a single executed transaction (fill) on the exchange.
    A trade belongs to a specific Order and is used to update the Position state.
    """
    trade_id: str
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL" representing execution mechanics
    price: Decimal
    quantity: Decimal
    fees: Decimal
    timestamp: float
