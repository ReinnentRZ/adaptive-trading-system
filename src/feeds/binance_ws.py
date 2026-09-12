import certifi
from decimal import Decimal
import logging
import ssl
import time
from typing import Any, Optional
import numpy as np
import talib
import websocket

from src.config import config
from src.core.enums import OrderType
from src.core.signal import Signal
from src.data.market_data import klines_to_df
from src.feeds.stream import process_kline
from src.indicators.adx import ADXIndicator
from src.indicators.atr import ATRIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.rsi import RSIIndicator
from src.indicators.wt import WTIndicator
from src.strategies.lorentzian import LorentzianStrategy
from src.strategies.risk_engine import AdaptiveRiskCalculator, AdaptiveRiskParams
from logs.log import log_close

logger = logging.getLogger("trading_system")


class BinanceWebSocket:
    """
    Real-time Binance WebSocket Kline stream listener with on-the-fly feature
    extraction, Meta-Labeling ML validation, and dynamic ATR risk management.
    """
    def __init__(
        self,
        data_manager,
        order_manager,
        meta_model: Optional[Any] = None,
        meta_threshold: float = 0.50,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
    ):
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.meta_model = meta_model
        self.meta_threshold = meta_threshold
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        
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
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False

    def stop(self) -> None:
        """Gracefully terminates WebSocket connection."""
        self._running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

    def on_message(self, ws, message):
        data_live = process_kline(message)
        
        # 1. Real-time Stop Loss & Take Profit check on every single WebSocket tick update
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

            if len(candles_clean_list) < 30:
                logger.warning(f"Insufficient candle history ({len(candles_clean_list)} bars) for indicator warmup.")
                return

            candle_terakhir = candles_clean_list[-1]
            harga_close_terakhir = candle_terakhir['close']

            # Compute technical indicators via TALib for high precision
            close_arr = df_clean['close'].to_numpy(dtype=np.float64)
            high_arr = df_clean['high'].to_numpy(dtype=np.float64)
            low_arr = df_clean['low'].to_numpy(dtype=np.float64)

            rsi_arr = talib.RSI(close_arr, timeperiod=14)
            cci_arr = talib.CCI(high_arr, low_arr, close_arr, timeperiod=20)
            adx_arr = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
            atr_arr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
            atr_ema_arr = talib.EMA(atr_arr, timeperiod=20)

            rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0
            cci_val = float(cci_arr[-1]) if not np.isnan(cci_arr[-1]) else 0.0
            adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 20.0
            atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
            ema_atr_val = float(atr_ema_arr[-1]) if not np.isnan(atr_ema_arr[-1]) else atr_val
            vol_ratio = (atr_val / ema_atr_val) if ema_atr_val > 0 else 1.0
            norm_atr = (atr_val / harga_close_terakhir) if harga_close_terakhir > 0 else 0.0

            # WaveTrend calculation
            hasil_wt = self.wt_bot.calculate_wt(candles_clean_list)
            wt1_val = hasil_wt.get("wt1")
            wt2_val = hasil_wt.get("wt2")
            wt_diff = float(wt1_val - wt2_val) if (wt1_val is not None and wt2_val is not None) else 0.0

            # Lorentzian strategy analyze
            hasil_prediksi = self.strategy.analyze(candles_clean_list)
            angka_voting = hasil_prediksi.prediction
            array_tetangga = hasil_prediksi.neighbors_labels
            lorentzian_signal = 1.0 if hasil_prediksi.signal.name == "LONG" else (-1.0 if hasil_prediksi.signal.name == "SHORT" else 0.0)

            # Telemetry logging
            hasil_rsi = self.rsi_bot.calculate_rsi(candles_clean_list)
            hasil_adx = self.adx_bot.calculate_adx(candles_clean_list)
            hasil_cci = self.cci_bot.calculate_cci(candles_clean_list)

            risk_params = AdaptiveRiskCalculator.calculate_risk_params(
                data=df_clean,
                entry_price=harga_close_terakhir,
                direction=OrderType.LONG,
                symbol=config.market.symbol
            )

            log_close(
                data_kline=candle_terakhir, 
                time_close=candle_terakhir["time_closed"], 
                rsi_value=rsi_val, 
                rsi_smoothing=hasil_rsi.get("rsi_smoothing"),
                adx_value=adx_val,
                adx_smoothing=hasil_adx.get("adx_smoothing"),
                cci_value=cci_val,
                cci_smoothing=hasil_cci.get("cci_smoothing"),
                wt1_value=wt1_val,
                wt2_value=wt2_val,
                array_tetangga=array_tetangga,
                raw_prediction=angka_voting,
                signal_name=hasil_prediksi.signal.name,
                atr_value=atr_val,
                applied_atr_period=14,
                market_regime=risk_params.market_regime,
                target_rr_ratio=self.tp_multiplier / self.sl_multiplier
            )

            # Primary Buy Event Check (Spot Long-Only)
            is_primary_buy = (lorentzian_signal == 1.0) or (rsi_val < 40.0 and wt_diff > 0.0)

            if is_primary_buy:
                logger.info(
                    f"[Layer-1] Primary Buy Condition Met | "
                    f"Lorentzian={lorentzian_signal} | RSI={rsi_val:.2f} | WT_Diff={wt_diff:.2f}"
                )

                should_execute = False
                confidence_score = 75.0

                if self.meta_model is not None:
                    feature_vector = np.array([[
                        rsi_val,
                        cci_val,
                        adx_val,
                        wt_diff,
                        atr_val,
                        vol_ratio,
                        norm_atr,
                        lorentzian_signal
                    ]], dtype=np.float64)

                    proba = float(self.meta_model.predict_proba(feature_vector)[0, 1])
                    confidence_score = round(proba * 100, 2)

                    logger.info(
                        f"[Meta-Model] Inference Probability: {proba:.4f} | "
                        f"Calibrated Threshold: {self.meta_threshold:.2f}"
                    )

                    if proba >= self.meta_threshold:
                        should_execute = True
                        logger.info(f"[Meta-Model] SIGNAL APPROVED (P={proba:.4f} >= {self.meta_threshold:.2f})")
                    else:
                        logger.info(f"[Meta-Model] SIGNAL REJECTED (P={proba:.4f} < {self.meta_threshold:.2f})")
                else:
                    # Fallback without ML meta-model
                    should_execute = (hasil_prediksi.signal.name == "LONG")
                    logger.warning("[Meta-Model] No model loaded. Using Layer-1 direct signal.")

                if should_execute and atr_val > 0:
                    entry_dec = Decimal(str(harga_close_terakhir))
                    atr_dec = Decimal(str(atr_val))
                    tick_size = AdaptiveRiskCalculator._get_tick_size(config.market.symbol)

                    sl_price = (entry_dec - (Decimal(str(self.sl_multiplier)) * atr_dec)).quantize(tick_size)
                    tp_price = (entry_dec + (Decimal(str(self.tp_multiplier)) * atr_dec)).quantize(tick_size)

                    custom_risk_params = AdaptiveRiskParams(
                        calculated_atr=atr_val,
                        applied_period=14,
                        market_regime=risk_params.market_regime,
                        sl_multiplier=self.sl_multiplier,
                        tp_multiplier=self.tp_multiplier,
                        target_rr_ratio=self.tp_multiplier / self.sl_multiplier,
                        sl_price=float(sl_price),
                        tp_price=float(tp_price)
                    )

                    sinyal_bot = Signal(
                        type=OrderType.LONG,
                        candle_close=entry_dec,
                        confidence=Decimal(str(confidence_score)),
                        raw_vote=int(angka_voting)
                    )

                    self.order_manager.process_signal(
                        signal=sinyal_bot,
                        symbol=config.market.symbol,
                        timestamp=candle_terakhir["time_closed"],
                        current_atr=atr_dec,
                        adaptive_risk_params=custom_risk_params
                    )

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, *args):
        logger.warning(f"WebSocket Connection Closed: {args}")

    def start_stream(self, symbol: str, interval: str) -> None:
        symbol_lower = symbol.lower()
        socket_url = f"{config.websocket.binance_ws_base_url}/{symbol_lower}@kline_{interval}"
        self._running = True
        
        while self._running:
            try:
                logger.info(f"Opening WebSocket connection for {symbol} [{interval}] -> {socket_url}...")
                self.ws = websocket.WebSocketApp(
                    socket_url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                
                self.ws.run_forever(
                    ping_interval=config.websocket.ping_interval, 
                    ping_timeout=config.websocket.ping_timeout,
                    sslopt={"ca_certs": certifi.where()}
                )
                
                if not self._running:
                    logger.info("WebSocket stopped by user/system request.")
                    break

                logger.warning("Connection closed normally. Attempting to reconnect...")
                
            except Exception as e:
                logger.error(f"Connection exception occurred: {e}")
            
            if self._running:
                logger.warning(f"Connection lost. Waiting {config.websocket.reconnect_delay_seconds}s before reconnecting...")
                time.sleep(config.websocket.reconnect_delay_seconds)