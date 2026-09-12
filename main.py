import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone
import joblib

from src.config import config
from src.data.data_manager import DataManager
from src.execution.mock_broker import MockBroker
from src.execution.order_manager import OrderManager
from src.feeds.binance_ws import BinanceWebSocket
from src.services.binance_client import BinanceService

logger = logging.getLogger("trading_system")


def save_bot_state(
    state_file: Path,
    symbol: str,
    timeframe: str,
    bot_mode: str,
    broker: MockBroker,
    initial_capital: float,
    model_loaded: bool,
    threshold: float,
) -> None:
    """
    Saves current portfolio, open position, and operational metrics
    to logs/bot_state.json upon shutdown or state transition.
    """
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        active_pos = broker.get_active_position(symbol)
        active_pos_dict = None
        if active_pos is not None:
            active_pos_dict = {
                "position_id": active_pos.position_id,
                "symbol": active_pos.symbol,
                "direction": active_pos.direction.name,
                "status": active_pos.status.name,
                "entry_price": float(active_pos.entry_price),
                "quantity": float(active_pos.quantity),
                "sl_price": float(active_pos.sl_price) if active_pos.sl_price is not None else None,
                "tp_price": float(active_pos.tp_price) if active_pos.tp_price is not None else None,
                "created_at": active_pos.created_at,
            }

        stats = broker.get_stats()
        state_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot_mode": bot_mode,
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_capital": initial_capital,
            "current_balance": float(broker.get_balance()),
            "model_loaded": model_loaded,
            "meta_threshold": threshold,
            "active_position": active_pos_dict,
            "stats": stats,
            "status": "STOPPED",
        }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_payload, f, indent=2)

        logger.info(f"[Shutdown] Saved bot portfolio state to {state_file}")
    except Exception as e:
        logger.error(f"[Shutdown] Failed to save bot state: {e}")


def main() -> None:
    # 1. Parse active environment variables & configurations
    symbol = os.getenv("ACTIVE_PAIR", os.getenv("TRADING_SYMBOL", "BTCUSDT")).upper()
    timeframe = os.getenv("ACTIVE_TIMEFRAME", os.getenv("TRADING_INTERVAL", "5m")).lower()
    initial_capital = float(os.getenv("INITIAL_CAPITAL", os.getenv("RISK_INITIAL_BALANCE", "100.0")))
    trade_amount = float(os.getenv("TRADE_AMOUNT", os.getenv("TRADE_QUANTITY_USDT", "5.0")))
    bot_mode = os.getenv("BOT_MODE", "paper").lower()
    dry_run = (bot_mode != "live")
    tp_multiplier = float(os.getenv("RISK_ATR_TP_MULTIPLIER", "2.0"))
    sl_multiplier = float(os.getenv("RISK_ATR_SL_MULTIPLIER", "1.0"))

    logger.info("=" * 60)
    logger.info("   ADAPTIVE TRADING SYSTEM - SPOT META-LABELING BOT   ")
    logger.info("=" * 60)
    logger.info(f"Bot Mode         : {bot_mode.upper()} (Dry Run: {dry_run})")
    logger.info(f"Active Pair      : {symbol}")
    logger.info(f"Active Timeframe : {timeframe}")
    logger.info(f"Initial Capital  : {initial_capital:.2f} USDT")
    logger.info(f"Trade Allocation : {trade_amount:.2f} USDT")
    logger.info(f"Risk Management  : TP={tp_multiplier}x ATR | SL={sl_multiplier}x ATR")
    logger.info("=" * 60)

    # 2. Ingest Meta-Labeling Model Artifacts
    coin_code = symbol.lower().replace("usdt", "").replace("busd", "")
    model_dir = Path("models").resolve()
    model_path = model_dir / f"{coin_code}_{timeframe}_dedication_lgbm.joblib"
    meta_path = model_dir / f"{coin_code}_{timeframe}_metadata.json"

    meta_model = None
    meta_threshold = 0.50
    model_loaded = False

    if model_path.exists():
        try:
            meta_model = joblib.load(model_path)
            model_loaded = True
            logger.info(f"[Model] Successfully loaded LightGBM dedication model from {model_path.name}")
            
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    meta_threshold = float(meta.get("recommended_threshold", 0.50))
                    logger.info(
                        f"[Model] Metadata loaded: Calibrated Threshold={meta_threshold:.2f} | "
                        f"OOS Win Rate={meta.get('test_win_rate', 0.0)*100:.1f}% | "
                        f"ROC-AUC={meta.get('test_roc_auc', 0.0):.4f}"
                    )
        except Exception as e:
            logger.error(f"[Model] Error loading model {model_path}: {e}. Running in fallback mode.")
    else:
        logger.warning(
            f"[Model] Model file not found at {model_path}. Bot will run using Layer-1 signals only."
        )

    # 3. Initialize Exchange Ingestion & Broker Services
    binance_api = BinanceService()
    data_manager = DataManager(binance_service=binance_api)

    warmup_limit = max(100, config.market.kline_limit)
    logger.info(f"[Data] Initializing candle buffer ({warmup_limit} candles) for {symbol} [{timeframe}]...")
    try:
        data_manager.initialize_bot(
            symbol=symbol,
            interval=timeframe,
            limit=warmup_limit
        )
        logger.info(f"[Data] Candle buffer ready with {len(data_manager.get_data())} historical bars.")
    except Exception as e:
        logger.error(f"[Data] Failed to pre-populate candle buffer from Binance API: {e}")

    # 4. Initialize Execution Layer
    broker = MockBroker()
    order_manager = OrderManager(broker=broker)

    # 5. Initialize Real-Time Feed Engine
    bot_stream = BinanceWebSocket(
        data_manager=data_manager,
        order_manager=order_manager,
        meta_model=meta_model,
        meta_threshold=meta_threshold,
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
    )

    # 6. Graceful Shutdown Handlers
    state_file = Path("logs/bot_state.json").resolve()

    def handle_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(f"[Shutdown] Received shutdown signal: {sig_name}. Stopping bot...")
        bot_stream.stop()
        save_bot_state(
            state_file=state_file,
            symbol=symbol,
            timeframe=timeframe,
            bot_mode=bot_mode,
            broker=broker,
            initial_capital=initial_capital,
            model_loaded=model_loaded,
            threshold=meta_threshold,
        )
        logger.info("[Shutdown] Graceful shutdown completed cleanly. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # 7. Start Real-Time WebSocket Loop
    logger.info(f"[Runtime] Starting live market data WebSocket listener for {symbol} [{timeframe}]...")
    try:
        bot_stream.start_stream(symbol=symbol, interval=timeframe)
    except (KeyboardInterrupt, SystemExit):
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"[Runtime] Fatal runtime error in WebSocket stream: {e}", exc_info=True)
        handle_shutdown(signal.SIGTERM, None)


if __name__ == "__main__":
    main()
