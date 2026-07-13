from dataclasses import dataclass

@dataclass
class Trade:
    """
    Domain model representing a single executed transaction (fill) on the exchange.
    A trade belongs to a specific Order and is used to update the Position state.
    """
    trade_id: str
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL" representing execution mechanics
    price: float
    quantity: float
    fees: float
    timestamp: float
