"""
Main Entry Point for the Adaptive Trading System (Spot Meta-Labeling Bot).

Executes the Unified Two-Stage Machine Learning Pipeline (Lorentzian + LightGBM Gatekeeper)
with real-time WebSocket feeds and dynamic ATR risk management.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Optional

from src.config import config
from src.execution.broker import Broker
from src.execution.live_bot import LiveTradingBot

logger = logging.getLogger("trading_system")


def save_bot_state(
    state_file: Path,
    symbol: str,
    timeframe: str,
    bot_mode: str,
    broker: Broker,
    initial_capital: float,
    model_loaded: bool,
    threshold: float,
) -> None:
    """
    Saves current portfolio, open position, and operational metrics
    to logs/bot_state.json upon shutdown or state transition.
    """
    try:
        target_path = Path(state_file).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(state_payload, f, indent=2)

        logger.info(f"[Shutdown] Saved bot portfolio state to {target_path}")
    except Exception as e:
        logger.error(f"[Shutdown] Failed to save bot state: {e}")


def main() -> None:
    """
    Initializes and starts the unified live trading bot engine.
    """
    bot = LiveTradingBot(app_config=config)

    def handle_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(f"[Shutdown] Received signal: {sig_name}. Stopping live bot cleanly...")
        bot.stop()
        logger.info("[Shutdown] Clean shutdown completed. Exiting process.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        bot.start()
    except (KeyboardInterrupt, SystemExit):
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"[Runtime] Fatal runtime error: {e}", exc_info=True)
        handle_shutdown(signal.SIGTERM, None)


if __name__ == "__main__":
    main()
