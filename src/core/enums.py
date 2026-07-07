from enum import Enum

class OrderType(Enum):
    LONG = "LONG"    
    SHORT = "SHORT"  
    HOLD = "HOLD"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"