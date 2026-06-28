
from src.services.binance_client import BinanceService
from src.data.data_manager import DataManager
from src.feeds.binance_ws import BinanceWebSocket

# ==========================================
# KONFIGURASI BOT 
# ==========================================
SYMBOL = "BTCUSDT"
INTERVAL = "1m"  
LIMIT = 200       


def main():
    binance_api = BinanceService()
    
    data_manager = DataManager(binance_service=binance_api)
    
    data_manager.initialize_bot(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
    
    bot_stream = BinanceWebSocket(data_manager=data_manager)

    bot_stream.start_stream(symbol=SYMBOL, interval=INTERVAL)

if __name__ == "__main__":
    main()
    

