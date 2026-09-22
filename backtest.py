"""
Historical Backtesting Script for Adaptive Trading System.
Evaluates the Unified Two-Stage Machine Learning Pipeline (Lorentzian + LightGBM Gatekeeper)
using historical candle data.
"""

import sys
import os
import uuid
from pathlib import Path
from decimal import Decimal
import joblib
import numpy as np
import pandas as pd
import talib

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import config
from src.services.binance_client import BinanceService
from src.data.data_manager import DataManager
from src.data.market_data import klines_to_df
from src.execution.mock_broker import MockBroker
from src.execution.order_manager import OrderManager
from src.indicators.rsi import RSIIndicator
from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.wt import WTIndicator
from src.indicators.atr import ATRIndicator
from src.strategies.lorentzian import LorentzianStrategy
from src.strategies.risk_engine import AdaptiveRiskCalculator, AdaptiveRiskParams
from src.core.signal import Signal
from src.core.enums import OrderType


def format_candle(c, symbol, interval):
    return {
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
        "ignore": str(c[11]),
    }


def run_backtest():
    symbol = config.market.symbol
    interval = config.market.interval
    kline_limit = config.market.kline_limit
    initial_balance = config.trading.initial_balance
    ai_threshold = config.strategy.ai_threshold
    tp_mult = config.trading.atr_tp_multiplier
    sl_mult = config.trading.atr_sl_multiplier

    print(f"==================================================")
    print(f"   ADAPTIVE TRADING SYSTEM - BACKTEST SIMULATION  ")
    print(f"==================================================")
    print(f"Symbol           : {symbol}")
    print(f"Interval         : {interval}")
    print(f"Kline Limit      : {kline_limit}")
    print(f"Initial Balance  : {initial_balance:.2f} USDT")
    print(f"AI Threshold     : {ai_threshold:.4f}")
    print(f"Risk Management  : TP={tp_mult}x ATR | SL={sl_mult}x ATR | Barrier={config.trading.time_barrier_bars} bars")
    print(f"==================================================")

    # 1. Load Meta-Model if available
    coin_code = symbol.lower().replace("usdt", "").replace("busd", "")
    model_path = Path("models") / f"{coin_code}_{interval.lower()}_dedication_lgbm.joblib"
    meta_model = None
    if model_path.exists():
        try:
            meta_model = joblib.load(model_path)
            print(f"[Model] Successfully loaded LightGBM dedication model: {model_path.name}")
        except Exception as e:
            print(f"[Model] Warning: Failed to load model {model_path}: {e}")
    else:
        print(f"[Model] Warning: Model not found at {model_path}. Running with Stage-1 signals only.")

    # 2. Fetch historical klines
    print("\nFetching historical data from Binance...")
    binance_service = BinanceService()
    try:
        raw_klines = binance_service.get_klines(
            symbol=symbol,
            interval=interval,
            limit=kline_limit,
        )
    except Exception as e:
        print(f"Error fetching klines: {e}")
        print("Please check network connectivity or use offline dataset.")
        sys.exit(1)

    print(f"Successfully fetched {len(raw_klines)} candles.")

    # 3. Setup components
    data_manager = DataManager(binance_service=binance_service)
    warmup_count = min(200, len(raw_klines) // 2)
    if len(raw_klines) <= warmup_count:
        print(f"Error: Need more than {warmup_count} candles.")
        sys.exit(1)

    warmup_klines = raw_klines[:warmup_count]
    test_klines = raw_klines[warmup_count:]

    from collections import deque
    formatted_warmup = [format_candle(k, symbol, interval) for k in warmup_klines]
    data_manager.limit = kline_limit
    data_manager.candles_history = deque(formatted_warmup, maxlen=data_manager.limit)

    broker = MockBroker()
    order_manager = OrderManager(broker=broker)

    rsi_bot = RSIIndicator()
    adx_bot = ADXIndicator()
    cci_bot = CCIIndicator()
    wt_bot = WTIndicator()
    atr_bot = ATRIndicator()

    strategy = LorentzianStrategy(
        rsi=rsi_bot,
        adx=adx_bot,
        cci=cci_bot,
        wt=wt_bot,
    )

    print(f"\nWarming up indicator history with {warmup_count} candles...")
    print(f"Simulating live execution across {len(test_klines)} test candles...\n")

    # 4. Step-by-step simulation
    for idx, k in enumerate(test_klines):
        candle_live = format_candle(k, symbol, interval)
        time_closed = candle_live["time_closed"]
        close_price = candle_live["close"]

        # A. Update high/low ticks for intraday SL/TP
        order_manager.update_market_price(symbol=symbol, current_price=candle_live["low"], timestamp=time_closed)
        order_manager.update_market_price(symbol=symbol, current_price=candle_live["high"], timestamp=time_closed)

        # B. Candle closed event: add candle & enforce Time Barrier
        data_manager.add_new_candle(candle_live)
        order_manager.on_candle_close(symbol=symbol, close_price=close_price, timestamp=time_closed)

        # C. Feature calculation
        candles_all = data_manager.get_data()
        df_clean = klines_to_df(candles_all)
        candles_clean_list = df_clean.to_dict("records")

        if len(candles_clean_list) < 30:
            continue

        close_arr = df_clean["close"].to_numpy(dtype=np.float64)
        high_arr = df_clean["high"].to_numpy(dtype=np.float64)
        low_arr = df_clean["low"].to_numpy(dtype=np.float64)

        rsi_arr = talib.RSI(close_arr, timeperiod=14)
        cci_arr = talib.CCI(high_arr, low_arr, close_arr, timeperiod=20)
        adx_arr = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
        atr_arr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
        atr_sma50_arr = talib.SMA(atr_arr, timeperiod=min(50, len(atr_arr)))

        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0
        cci_val = float(cci_arr[-1]) if not np.isnan(cci_arr[-1]) else 0.0
        adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 20.0
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        sma_atr = float(atr_sma50_arr[-1]) if (not np.isnan(atr_sma50_arr[-1]) and atr_sma50_arr[-1] > 0) else atr_val
        vol_ratio = (atr_val / sma_atr) if sma_atr > 0 else 1.0
        norm_atr = (atr_val / close_price) if close_price > 0 else 0.0

        hasil_wt = wt_bot.calculate_wt(candles_clean_list)
        wt1_val = hasil_wt.get("wt1")
        wt2_val = hasil_wt.get("wt2")
        wt_diff = float(wt1_val - wt2_val) if (wt1_val is not None and wt2_val is not None) else 0.0

        hasil_prediksi = strategy.analyze(candles_clean_list)
        lorentzian_signal = 1.0 if hasil_prediksi.signal.name == "LONG" else (-1.0 if hasil_prediksi.signal.name == "SHORT" else 0.0)

        # Stage 1: Primary Signal Trigger
        primary_signal = (lorentzian_signal == 1.0) or (rsi_val < 40.0 and wt_diff > 0.0)

        if not primary_signal or broker.get_active_position(symbol) is not None:
            continue

        # Stage 2: LightGBM AI Gatekeeper
        should_execute = False
        confidence_score = 75.0

        if meta_model is not None:
            feat_vec = np.array([[rsi_val, cci_val, adx_val, wt_diff, atr_val, vol_ratio, norm_atr, lorentzian_signal]], dtype=np.float64)
            raw_p = meta_model.predict_proba(feat_vec)
            prob = float(raw_p[0, 1] if raw_p.ndim == 2 and raw_p.shape[1] > 1 else raw_p.ravel()[0])
            confidence_score = round(prob * 100.0, 2)
            should_execute = (prob >= ai_threshold)
        else:
            should_execute = (lorentzian_signal == 1.0)

        if should_execute and atr_val > 0:
            entry_dec = Decimal(str(close_price))
            atr_dec = Decimal(str(atr_val))
            tick_size = AdaptiveRiskCalculator._get_tick_size(symbol)

            sl_price = (entry_dec - (Decimal(str(sl_mult)) * atr_dec)).quantize(tick_size)
            tp_price = (entry_dec + (Decimal(str(tp_mult)) * atr_dec)).quantize(tick_size)

            risk_p = AdaptiveRiskParams(
                calculated_atr=atr_val,
                applied_period=14,
                market_regime="Trending",
                sl_multiplier=sl_mult,
                tp_multiplier=tp_mult,
                target_rr_ratio=tp_mult / sl_mult,
                sl_price=float(sl_price),
                tp_price=float(tp_price),
            )

            sig = Signal(
                type=OrderType.LONG,
                candle_close=entry_dec,
                confidence=Decimal(str(confidence_score)),
                raw_vote=int(hasil_prediksi.prediction),
            )

            order_manager.process_signal(
                signal=sig,
                symbol=symbol,
                timestamp=time_closed,
                current_atr=atr_dec,
                adaptive_risk_params=risk_p,
            )

    # 5. Summary statistics
    stats = broker.get_stats()
    print("\n==================================================")
    print("           BACKTEST SIMULATION RESULTS            ")
    print("==================================================")
    print(f"Total Trades     : {stats['total_trades']}")
    print(f"Winning Trades   : {stats['win']}")
    print(f"Losing Trades    : {stats['loss']}")
    print(f"Win Rate         : {stats['win_rate']:.2f}%")
    print(f"Total Profit     : {stats['total_profit']:.2f} USDT")
    print(f"Total Loss       : {stats['total_loss']:.2f} USDT")
    print(f"Final Balance    : {stats['current_balance']:.2f} USDT")
    net_return = ((stats['current_balance'] - initial_balance) / initial_balance) * 100.0
    print(f"Net Return       : {net_return:+.2f}%")
    print("==================================================")


if __name__ == "__main__":
    run_backtest()
