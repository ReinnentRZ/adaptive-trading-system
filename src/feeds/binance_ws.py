import websocket
import time
from src.feeds.stream import process_kline
from src.data.market_data import klines_to_df
from src.indicators.rsi import RSIIndicator
from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.wt import WTIndicator
from src.strategies.lorentzian import LorentzianStrategy
from src.core.signal import Signal
from logs.log import log_close
from src.config import WEBSOCKET

class BinanceWebSocket:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
        self.rsi_bot = RSIIndicator()
        self.adx_bot = ADXIndicator()
        self.cci_bot = CCIIndicator()
        self.wt_bot = WTIndicator()
        
        self.strategy = LorentzianStrategy(
            rsi=self.rsi_bot, 
            adx=self.adx_bot, 
            cci=self.cci_bot, 
            wt=self.wt_bot
        )

    def on_message(self, ws, message):
        data_live = process_kline(message)
        
        if data_live.get("is_closed", True): 
            self.data_manager.add_new_candle(data_live)
            
            candles_all = self.data_manager.get_data()

            df_clean = klines_to_df(candles_all)

            candles_clean_list = df_clean.to_dict('records')

            candle_terakhir = candles_clean_list[-1]
            harga_close_terakhir = candle_terakhir['close']

            hasil_prediksi = self.strategy.analyze(candles_clean_list)
            angka_voting = hasil_prediksi.prediction
            array_tetangga = hasil_prediksi.neighbors_labels
            
            hasil_rsi = self.rsi_bot.calculate_rsi(candles_clean_list)
            hasil_adx = self.adx_bot.calculate_adx(candles_clean_list)
            hasil_cci = self.cci_bot.calculate_cci(candles_clean_list)
            hasil_wt = self.wt_bot.calculate_wt(candles_clean_list)
            

            current_candle_log = candles_clean_list[-1]

            if hasil_rsi["rsi"] is not None and hasil_wt["wt1"] is not None:
                log_close(
                    data_kline=current_candle_log, 
                    time_close=current_candle_log["time_closed"], 
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
                sinyal_bot = Signal.from_lorentzian(
                    hasil_prediksi, 
                    harga_close_terakhir
                    )

    def on_error(self, ws, error):
        print("WebSocket Error:", error)

    def on_close(self, ws, *args):
        print("WebSocket Connection Closed:", args)

    def start_stream(self, symbol, interval):
        symbol_lower = symbol.lower()
        socket = f"{WEBSOCKET.binance_ws_base_url}/{symbol_lower}@kline_{interval}"
        
        while True:
            try:
                print(f"Membuka pintu gerbang WebSocket untuk {symbol} [{interval}]...")
                ws = websocket.WebSocketApp(
                    socket,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                
                ws.run_forever(
                    ping_interval=WEBSOCKET.ping_interval, 
                    ping_timeout=WEBSOCKET.ping_timeout
                )
                
                print("Koneksi ditutup secara normal. Mencoba menyambung kembali...")
                
            except Exception as e:
                print(f"Terjadi kesalahan koneksi: {e}")
            
            print(f"Koneksi terputus. Menunggu {WEBSOCKET.reconnect_delay_seconds} detik sebelum menyambung ulang...")
            time.sleep(WEBSOCKET.reconnect_delay_seconds)