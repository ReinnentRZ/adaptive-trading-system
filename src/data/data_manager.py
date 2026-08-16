import uuid
from collections import deque
from typing import List, Sequence
from src.services.binance_client import BinanceService


class DataManager:
    """
    Manages historical candles utilizing a bounded collections.deque
    to prevent memory exhaustion.
    """
    def __init__(self, binance_service: BinanceService):
        self.binance: BinanceService = binance_service
        self.candles_history: deque = deque()
        self.limit: int = 200

    def initialize_bot(self, symbol: str, interval: str, limit: int) -> None:
        """
        Initializes historical candles buffer by fetching recent data from the service.
        """
        self.limit = limit
        self.candles_history = deque(maxlen=self.limit)
        raw_candles: Sequence = self.binance.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        for c in raw_candles:
            self.candles_history.append({
                "id": uuid.uuid4().hex,  
                "symbol": symbol,
                "interval": interval,
                "time_open": int(c[0]),      
                "time_closed": int(c[6]), 
                "first_trade_id": 0,       
                "last_trade_id": 0,        
                "open": float(c[1]),     
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]), 
                "volume": float(c[5]),
                "trades_count": int(c[8]),
                "is_closed": True,
                "quote_volume": float(c[7]),
                "taker_buy_base": float(c[9]),
                "taker_buy_quote": float(c[10]),
                "ignore": str(c[11])
            })

    def get_data(self) -> List[dict]:
        """
        Returns the historical candles buffer as a list of dictionaries.
        """
        return list(self.candles_history)

    def add_new_candle(self, new_candle: dict) -> None:
        """
        Appends a new candle to the bounded history. Automatically handles max length eviction.
        """
        self.candles_history.append(new_candle)