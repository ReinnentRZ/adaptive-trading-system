import websocket
import time
from src.feeds.stream import process_kline
from src.indicators.rsi import RSIIndicator
from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.wt import WTIndicator
from src.strategies.lorentzian import LorentzianStrategy, Direction
from logs.log import log_close

class BinanceWebSocket:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
        self.rsi_bot = RSIIndicator(rsi_period=14, ma_period=14)
        self.adx_bot = ADXIndicator(adx_period=14)
        self.cci_bot = CCIIndicator(cci_period=20, smoothing_period=14)
        self.wt_bot = WTIndicator(channel_length=10, average_length=21, wt_sma_length=4)
        
        self.strategy = LorentzianStrategy(
            rsi=self.rsi_bot, 
            adx=self.adx_bot, 
            cci=self.cci_bot, 
            wt=self.wt_bot
        )

    def on_message(self, ws, message):
        data = process_kline(message)
        
        if data.get("is_closed", True): 
            self.data_manager.add_new_candle(data)
            
            candles_all = self.data_manager.get_data()

            hasil_prediksi = self.strategy.analyze(candles_all)
            angka_voting = hasil_prediksi.prediction
            array_tetangga = hasil_prediksi.neighbors_labels
            
            hasil_rsi = self.rsi_bot.calculate_rsi(candles_all)
            hasil_adx = self.adx_bot.calculate_adx(candles_all)
            hasil_cci = self.cci_bot.calculate_cci(candles_all)
            hasil_wt = self.wt_bot.calculate_wt(candles_all)

            if hasil_rsi["rsi"] is not None and hasil_wt["wt1"] is not None:
                log_close(

                    data_kline=data, 
                    time_close=data["time_closed"], 
                    rsi_value=hasil_rsi["rsi"], 
                    rsi_smoothing=hasil_rsi["rsi_smoothing"],
                    adx_value=hasil_adx["adx"],
                    adx_smoothing=hasil_adx["adx_smoothing"],
                    cci_value=hasil_cci["cci"],
                    cci_smoothing=hasil_cci["cci_smoothing"],
                    wt1_value=hasil_wt["wt1"],
                    wt2_value=hasil_wt["wt2"],
                    array_tetangga=array_tetangga,
                    raw_prediction=angka_voting,
                    signal_name=hasil_prediksi.signal.name 
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