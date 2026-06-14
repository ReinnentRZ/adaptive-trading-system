import sys
from src.services.binance_client import BinanceService
from src.data.market_data import klines_to_chart 
from src.feeds.binance_ws import streamKline

bot = BinanceService()

if __name__ == "__main__":
    streamKline("btcusdt", "1m")
    

