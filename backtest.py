import sys
import os
import uuid
import pandas as pd
from decimal import Decimal

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
from src.strategies.risk_engine import AdaptiveRiskCalculator
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
        "ignore": str(c[11])
    }


def run_backtest():
    print(f"=== Starting Backtest Simulation ===")
    print(f"Symbol: {config.market.symbol}")
    print(f"Interval: {config.market.interval}")
    print(f"Kline Limit: {config.market.kline_limit}")
    print(f"Initial Balance: {config.trading.initial_balance} USDT")
    print(f"Dry Run Mode: {config.trading.dry_run}")
    
    # 1. Fetch historical klines
    print("\nFetching historical data from Binance...")
    binance_service = BinanceService()
    try:
        raw_klines = binance_service.get_klines(
            symbol=config.market.symbol,
            interval=config.market.interval,
            limit=config.market.kline_limit
        )
    except Exception as e:
        print(f"Error fetching klines: {e}")
        print("Please check your network connection and API rate limits.")
        sys.exit(1)
        
    print(f"Successfully fetched {len(raw_klines)} candles.")

    # 2. Setup components
    data_manager = DataManager(binance_service=binance_service)
    
    # We initialize the DataManager with the first 200 candles to warm up indicators,
    # then feed the rest one-by-one to simulate live trading.
    warmup_count = 200
    if len(raw_klines) <= warmup_count:
        print(f"Error: Not enough candles. Need more than {warmup_count} candles.")
        sys.exit(1)
        
    warmup_klines = raw_klines[:warmup_count]
    test_klines = raw_klines[warmup_count:]
    
    formatted_warmup = [
        format_candle(k, config.market.symbol, config.market.interval) 
        for k in warmup_klines
    ]
        
    data_manager.candles_history = pd.Series(formatted_warmup).tolist()
    # Let's populate the deque inside data_manager properly
    from collections import deque
    data_manager.limit = config.market.kline_limit
    data_manager.candles_history = deque(formatted_warmup, maxlen=data_manager.limit)
    
    # Initialize execution components
    broker = MockBroker()
    order_manager = OrderManager(broker=broker)
    
    # Indicators
    rsi_bot = RSIIndicator()
    adx_bot = ADXIndicator()
    cci_bot = CCIIndicator()
    wt_bot = WTIndicator()
    atr_bot = ATRIndicator()
    
    strategy = LorentzianStrategy(
        rsi=rsi_bot,
        adx=adx_bot,
        cci=cci_bot,
        wt=wt_bot
    )
    
    print(f"\nWarming up indicator history with {warmup_count} candles...")
    print(f"Running simulation on the remaining {len(test_klines)} candles...")
    
    # 3. Simulate candle feed one-by-one
    for idx, k in enumerate(test_klines):
        candle_live = format_candle(k, config.market.symbol, config.market.interval)
        
        # A. Update OrderManager with current live tick (high/low price check for SL/TP)
        order_manager.update_market_price(
            symbol=config.market.symbol,
            current_price=candle_live["low"],
            timestamp=candle_live["time_closed"]
        )
        order_manager.update_market_price(
            symbol=config.market.symbol,
            current_price=candle_live["high"],
            timestamp=candle_live["time_closed"]
        )
        
        # B. Add candle to DataManager
        data_manager.add_new_candle(candle_live)
        
        # C. Run Strategy & Indicator Analysis
        candles_all = data_manager.get_data()
        df_clean = klines_to_df(candles_all)
        candles_clean_list = df_clean.to_dict('records')
        
        harga_close_terakhir = candles_clean_list[-1]['close']
        
        hasil_prediksi = strategy.analyze(candles_clean_list)
        
        # D. Calculate dynamic risk parameters
        direction_for_risk = OrderType.LONG
        if hasil_prediksi.signal.name == "SHORT":
            direction_for_risk = OrderType.SHORT
            
        risk_params = AdaptiveRiskCalculator.calculate_risk_params(
            data=df_clean,
            entry_price=harga_close_terakhir,
            direction=direction_for_risk,
            symbol=config.market.symbol
        )
        
        # E. Sinyal check
        sinyal_bot = Signal.from_lorentzian(
            hasil_prediksi,
            harga_close_terakhir
        )
        
        if sinyal_bot is not None:
            signal_risk_params = AdaptiveRiskCalculator.calculate_risk_params(
                data=df_clean,
                entry_price=harga_close_terakhir,
                direction=sinyal_bot.type,
                symbol=config.market.symbol
            )
            order_manager.process_signal(
                signal=sinyal_bot,
                symbol=config.market.symbol,
                timestamp=candle_live["time_closed"],
                current_atr=Decimal(str(signal_risk_params.calculated_atr)),
                adaptive_risk_params=signal_risk_params
            )
            
    # 4. Print results
    stats = broker.get_stats()
    print("\n=== Backtest Simulation Results ===")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Total Profit: {stats['total_profit']:.2f} USDT")
    print(f"Total Loss: {stats['total_loss']:.2f} USDT")
    print(f"Final Balance: {stats['current_balance']:.2f} USDT")
    print(f"Net Return: {((stats['current_balance'] - config.trading.initial_balance) / config.trading.initial_balance) * 100:.2f}%")
    print("===================================")


if __name__ == "__main__":
    run_backtest()
