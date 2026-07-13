from dataclasses import dataclass

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
BINANCE_WS_BASE_URL = "wss://stream.binance.com:9443/ws"
PING_INTERVAL = 30
PING_TIMEOUT = 10
RECONNECT_DELAY_SECONDS = 5

# ==============================================================================
# INTERNAL
# ==============================================================================
@dataclass(frozen=True)
class WebSocketSettings:
    binance_ws_base_url: str
    ping_interval: int
    ping_timeout: int
    reconnect_delay_seconds: int

WEBSOCKET = WebSocketSettings(
    binance_ws_base_url=BINANCE_WS_BASE_URL,
    ping_interval=PING_INTERVAL,
    ping_timeout=PING_TIMEOUT,
    reconnect_delay_seconds=RECONNECT_DELAY_SECONDS,
)
