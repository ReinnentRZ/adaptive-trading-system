import uuid
class DataManager:
    def __init__(self, binance_service):
        self.binance = binance_service
        self.candles_history = [] 
        self.limit = 200

    def initialize_bot(self, symbol, interval, limit):

        self.limit = limit

        raw_candles = self.binance.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        self.candles_history = []
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

    def get_data(self):
        return self.candles_history

    def add_new_candle(self, new_candle):

        self.candles_history.append(new_candle)
        if len(self.candles_history) > self.limit:
            self.candles_history.pop(0)
            