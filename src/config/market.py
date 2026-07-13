from dataclasses import dataclass

# ==============================================================================
# USER-EDITABLE CONFIGURATION
# ==============================================================================
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 2000

# ==============================================================================
# TYPE-SAFE INTERNAL REPRESENTATION
# ==============================================================================
@dataclass(frozen=True)
class MarketSettings:
    symbol: str
    interval: str
    limit: int

MARKET = MarketSettings(
    symbol=SYMBOL,
    interval=INTERVAL,
    limit=LIMIT,
)
