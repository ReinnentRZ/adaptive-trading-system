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
                "symbol": symbol,
                "interval": interval,
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]), 
                "time": int(c[0]),
                "is_closed": True
            })

    def get_data_for_rsi(self):
        return self.candles_history

    def add_new_candle(self, new_candle):

        self.candles_history.append(new_candle)
        if len(self.candles_history) > self.limit:
            self.candles_history.pop(0)