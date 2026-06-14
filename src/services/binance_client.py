import time
from binance.client import Client
from src.config.api import API_KEY, API_SECRET

class BinanceService:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        
    def get_price(self, symbol="BTCUSDT"):
        return self.client.get_symbol_ticker(symbol=symbol)

    def get_klines(self, symbol, interval, limit=100):
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
    
    def get_uiKlines(self, symbol, interval, limit=100):
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
            