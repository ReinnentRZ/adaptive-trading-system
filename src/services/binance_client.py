import time
from binance.client import Client
from src.config import config

class BinanceService:
    def __init__(self):
        self.client = Client(config.api.api_key, config.api.api_secret)

    def on_start(self, symbol: str = config.market.symbol, interval: str = config.market.interval, limit: int = config.market.kline_limit):
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
    def get_price(self, symbol: str = config.market.symbol):
        return self.client.get_symbol_ticker(symbol=symbol)

    def get_klines(self, symbol: str, interval: str, limit: int = config.market.kline_limit):
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

    def ping(self):
        return self.client.ping()
    
    def get_server_time(self):
        return self.client.get_server_time()
    
    def get_ping_latency(self):
            try:
                start_time = int(time.time() * 1000)
                
                server_response = self.client.get_server_time()
                
                end_time = int(time.time() * 1000)
                
                ping_ms = end_time - start_time
                
                return f"Ping: {ping_ms} ms"
            except Exception as e:
                return f"Gagal cek ping: {e}"
