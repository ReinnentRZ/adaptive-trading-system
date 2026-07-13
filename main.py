
from src.services.binance_client import BinanceService
from src.data.data_manager import DataManager
from src.feeds.binance_ws import BinanceWebSocket
from src.config import MARKET

def main():
    binance_api = BinanceService()
    
    data_manager = DataManager(binance_service=binance_api)
    
    data_manager.initialize_bot(
        symbol=MARKET.symbol, 
        interval=MARKET.interval, 
        limit=MARKET.limit
    )
    
    bot_stream = BinanceWebSocket(data_manager=data_manager)

    bot_stream.start_stream(symbol=MARKET.symbol, interval=MARKET.interval)

if __name__ == "__main__":
    main()
    

