from dataclasses import dataclass

# ==============================================================================
# USER-EDITABLE CONFIGURATION
# ==============================================================================
INITIAL_BALANCE = 100.0  # Starting balance in USDT
TAKE_PROFIT_PCT = 3.0     # Take Profit in percent (e.g. 3.0%)
STOP_LOSS_PCT = 1.0       # Stop Loss in percent (e.g. 1.0%)
COMMISSION_FEE_PCT = 0.1  # Broker fee in percent per transaction (e.g., 0.1% for Binance)
TRADE_QUANTITY_USDT = 5.0  # Allocation amount per trade in USDT

# ==============================================================================
# TYPE-SAFE INTERNAL REPRESENTATION
# ==============================================================================
@dataclass(frozen=True)
class TradingSettings:
    initial_balance: float
    take_profit_pct: float
    stop_loss_pct: float
    commission_fee_pct: float
    trade_quantity_usdt: float

TRADING = TradingSettings(
    initial_balance=INITIAL_BALANCE,
    take_profit_pct=TAKE_PROFIT_PCT,
    stop_loss_pct=STOP_LOSS_PCT,
    commission_fee_pct=COMMISSION_FEE_PCT,
    trade_quantity_usdt=TRADE_QUANTITY_USDT,
)
