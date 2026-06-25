import websocket
import time
from src.feeds.stream import process_kline
from src.indicators.rsi import RSIIndicator
from logs.log import log_close

class BinanceWebSocket:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.rsi_bot = RSIIndicator(rsi_period=14, ma_period=14)

    def on_message(self, ws, message):

        data = process_kline(message)
        
        if data.get("is_closed", True): 
            
            self.data_manager.add_new_candle(data)
            
            candles = self.data_manager.get_data_for_rsi()

            hasil_rsi = self.rsi_bot.calculate(candles)

            if hasil_rsi["rsi"] is not None:
                #print(f"[{data['symbol']}] RSI: {hasil_rsi['rsi']} | Smoothing: {hasil_rsi['rsi_smoothing']}")
                
                log_close(
                data_kline=data, 
                time_close=data["time_closed"], 
                rsi_value=hasil_rsi["rsi"], 
                rsi_smoothing=hasil_rsi["rsi_smoothing"]
            )

    def on_error(self, ws, error):
        print("WebSocket Error:", error)

    def on_close(self, ws, *args):
        print("WebSocket Connection Closed:", args)

    def start_stream(self, symbol, interval):
        symbol_lower = symbol.lower()
        socket = f"wss://stream.binance.com:9443/ws/{symbol_lower}@kline_{interval}"
        
        while True:
            try:
                print(f"Membuka pintu gerbang WebSocket untuk {symbol} [{interval}]...")
                ws = websocket.WebSocketApp(
                    socket,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                
                ws.run_forever(ping_interval=30, ping_timeout=10)
                
                print("Koneksi ditutup secara normal. Mencoba menyambung kembali...")
                
            except Exception as e:
                print(f"Terjadi kesalahan koneksi: {e}")
            
            print("Koneksi terputus. Menunggu 5 detik sebelum menyambung ulang...")
            time.sleep(5)