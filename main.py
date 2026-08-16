
from src.services.binance_client import BinanceService
from src.data.data_manager import DataManager
from src.feeds.binance_ws import BinanceWebSocket
from src.config import MARKET
from src.execution.mock_broker import MockBroker
from src.execution.order_manager import OrderManager

def main():
    binance_api = BinanceService()
    
    data_manager = DataManager(binance_service=binance_api)
    
    data_manager.initialize_bot(
        symbol=MARKET.symbol, 
        interval=MARKET.interval, 
        limit=MARKET.limit
    )
    
    # Initialize execution components for Paper Trading
    broker = MockBroker()
    order_manager = OrderManager(broker=broker)
    
    bot_stream = BinanceWebSocket(data_manager=data_manager, order_manager=order_manager)

    bot_stream.start_stream(symbol=MARKET.symbol, interval=MARKET.interval)

if __name__ == "__main__":
    main()
    

