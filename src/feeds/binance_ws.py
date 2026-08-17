from decimal import Decimal
import websocket
import time
import ssl
import certifi
import logging
from src.feeds.stream import process_kline

logger = logging.getLogger("trading_system")
from src.data.market_data import klines_to_df
from src.indicators.rsi import RSIIndicator
from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.wt import WTIndicator
from src.indicators.atr import ATRIndicator
from src.strategies.lorentzian import LorentzianStrategy
from src.strategies.risk_engine import AdaptiveRiskCalculator
from src.core.enums import OrderType
from src.core.signal import Signal
from logs.log import log_close
from src.config import config

class BinanceWebSocket:
    def __init__(self, data_manager, order_manager):
        self.data_manager = data_manager
        self.order_manager = order_manager
        
        self.rsi_bot = RSIIndicator()
        self.adx_bot = ADXIndicator()
        self.cci_bot = CCIIndicator()
        self.wt_bot = WTIndicator()
        self.atr_bot = ATRIndicator()
        
        self.strategy = LorentzianStrategy(
            rsi=self.rsi_bot, 
            adx=self.adx_bot, 
            cci=self.cci_bot, 
            wt=self.wt_bot
        )

    def on_message(self, ws, message):
        data_live = process_kline(message)
        
        # 1. Real-time Stop Loss & Take Profit check on every single WebSocket tick update (multi-tick support)
        harga_close_live = float(data_live["close"])
        waktu_live = data_live["time_closed"]
        self.order_manager.update_market_price(
            symbol=config.market.symbol,
            current_price=harga_close_live,
            timestamp=waktu_live
        )
        
        # 2. Strategy and signal processing ONLY executed when a candle officially closes
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
            hasil_atr = self.atr_bot.calculate(df_clean)
            
            current_candle_log = candle_terakhir

            # Calculate dynamic risk parameters for telemetry logging (defaulting to LONG if HOLD)
            direction_for_risk = OrderType.LONG
            if hasil_prediksi.signal.name == "SHORT":
                direction_for_risk = OrderType.SHORT

            risk_params = AdaptiveRiskCalculator.calculate_risk_params(
                data=df_clean,
                entry_price=harga_close_terakhir,
                direction=direction_for_risk,
                symbol=config.market.symbol
            )

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
                    signal_name=hasil_prediksi.signal.name,
                    atr_value=risk_params.calculated_atr,
                    applied_atr_period=risk_params.applied_period,
                    market_regime=risk_params.market_regime,
                    target_rr_ratio=risk_params.target_rr_ratio
                )
                sinyal_bot = Signal.from_lorentzian(
                    hasil_prediksi, 
                    harga_close_terakhir
                )

                # Eksekusi sinyal jika terbentuk sinyal buy/sell (bukan HOLD/None)
                if sinyal_bot is not None:
                    # Calculate direction-specific dynamic risk parameters for order placement
                    signal_risk_params = AdaptiveRiskCalculator.calculate_risk_params(
                        data=df_clean,
                        entry_price=harga_close_terakhir,
                        direction=sinyal_bot.type,
                        symbol=config.market.symbol
                    )
                    self.order_manager.process_signal(
                        signal=sinyal_bot,
                        symbol=config.market.symbol,
                        timestamp=current_candle_log["time_closed"],
                        current_atr=Decimal(str(signal_risk_params.calculated_atr)),
                        adaptive_risk_params=signal_risk_params
                    )

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, *args):
        logger.warning(f"WebSocket Connection Closed: {args}")

    def start_stream(self, symbol, interval):
        symbol_lower = symbol.lower()
        socket = f"{config.websocket.binance_ws_base_url}/{symbol_lower}@kline_{interval}"
        
        while True:
            try:
                logger.info(f"Opening WebSocket connection for {symbol} [{interval}]...")
                ws = websocket.WebSocketApp(
                    socket,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                
                ws.run_forever(
                    ping_interval=config.websocket.ping_interval, 
                    ping_timeout=config.websocket.ping_timeout,
                    sslopt={"ca_certs": certifi.where()}
                )
                
                logger.warning("Connection closed normally. Attempting to reconnect...")
                
            except Exception as e:
                logger.error(f"Connection exception occurred: {e}")
            
            logger.warning(f"Connection lost. Waiting {config.websocket.reconnect_delay_seconds} seconds before reconnecting...")
            time.sleep(config.websocket.reconnect_delay_seconds)