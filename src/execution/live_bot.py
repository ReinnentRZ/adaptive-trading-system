#!/usr/bin/env python3
"""
Unified Live Execution Engine for Spot Trading (Market Regime Corong 3-Pilar).

Consumes RegimeFunnelStrategy and RegimeFunnelConfig as the Single Source of Truth (SSOT):
  - Pilar 1: Causal Gaussian HMM Regime Gate (Bullish State 0, Single-Shot, state_age <= 4)
             + Macro Structural Trend Filter (Close > EMA 200).
  - Pilar 2: Micro Pullback Timing Trigger (Low <= EMA(9) OR RSI(14) <= 52.0).
  - Pilar 3: Dynamic Risk Engine (Maker Limit Entry, Scaling Out 50:50, Hard-Floored BE Lock).

Includes hourly candle ingestion (00:05), state persistence (data/live_bot_state.json),
tick monitoring (15-30s), and dry-run CLI evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Auto-bootstrap into .venv if current interpreter is running outside of it
_project_root = Path(__file__).resolve().parent.parent.parent
_venv_py = _project_root / ".venv" / "bin" / "python"
if _venv_py.exists() and Path(sys.prefix).resolve() != (_project_root / ".venv").resolve():
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

import ccxt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import REGIME_FUNNEL, RegimeFunnelConfig
from src.strategies.regime_funnel import RegimeFunnelStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LiveBot")


@dataclass
class ActivePosition:
    """Represents an active spot position managed by Pilar 3 risk engine."""
    position_id: str
    symbol: str
    entry_price: float
    initial_quantity: float
    remaining_quantity: float
    tp1_price: float
    tp2_price: float
    sl_price: float
    be_price: float
    tp1_hit: bool = False
    entry_time: str = ""
    bars_held: int = 0


@dataclass
class PendingOrder:
    """Represents an open maker limit buy order placed at bar close."""
    order_id: str
    symbol: str
    price: float
    amount: float
    signal_atr: float
    created_at_time: str
    expire_after_bar_time: Optional[str] = None


class LiveTradingBot:
    """
    Live and Paper Trading Bot implementing the 3-Pilar Regime Corong Strategy.
    """

    def __init__(
        self,
        config: Optional[RegimeFunnelConfig] = None,
        exchange: Optional[ccxt.Exchange] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        state_file: Optional[Path] = None,
        is_testnet: Optional[bool] = None,
        app_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.config: RegimeFunnelConfig = config or getattr(app_config, "regime_funnel", REGIME_FUNNEL)
        self.strategy = RegimeFunnelStrategy(config=self.config)

        # Environment configuration
        self.symbol = symbol or os.getenv("SYMBOL", "BTC/USDT")
        self.timeframe = timeframe or os.getenv("TIMEFRAME", "1h")
        self.state_file = Path(state_file or os.getenv("STATE_FILE", "data/live_bot_state.json"))
        
        testnet_env = os.getenv("IS_TESTNET", "true").lower() in ("true", "1", "yes")
        self.is_testnet = is_testnet if is_testnet is not None else testnet_env

        # CCXT Exchange Client initialization
        self.exchange = exchange or self._init_exchange()

        # Operational state
        self.active_position: Optional[ActivePosition] = None
        self.pending_order: Optional[PendingOrder] = None
        self.traded_in_current_episode: bool = False
        self.last_state_id: Optional[int] = None
        self.state_age: int = 0
        self.completed_trades: List[Dict[str, Any]] = []
        self.is_running: bool = False

        # Load persisted state if exists
        self.load_state()

    def _init_exchange(self) -> ccxt.binance:
        """Initializes ccxt.binance exchange client with rate-limiting and credentials."""
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret_key = os.getenv("BINANCE_SECRET_KEY", "")

        exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
            }
        )

        if self.is_testnet:
            try:
                exchange.set_sandbox_mode(True)
                logger.info("[Exchange] Sandbox / Testnet mode enabled.")
            except Exception as e:
                logger.warning(f"[Exchange] Sandbox mode notice: {e}")

        return exchange

    def load_state(self) -> None:
        """Loads bot state from disk to maintain continuity across restarts."""
        if not self.state_file.exists():
            logger.info(f"[State] No previous state found at {self.state_file}. Starting fresh.")
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("active_position"):
                pos_data = data["active_position"]
                self.active_position = ActivePosition(**pos_data)
                logger.info(f"[State] Restored active position: {self.active_position.symbol} @ ${self.active_position.entry_price:,.2f}")

            if data.get("pending_order"):
                ord_data = data["pending_order"]
                self.pending_order = PendingOrder(**ord_data)
                logger.info(f"[State] Restored pending order: #{self.pending_order.order_id} @ ${self.pending_order.price:,.2f}")

            self.traded_in_current_episode = data.get("traded_in_current_episode", False)
            self.last_state_id = data.get("last_state_id", None)
            self.state_age = data.get("state_age", 0)
            self.completed_trades = data.get("completed_trades", [])
            logger.info(f"[State] Loaded persistent state successfully (History: {len(self.completed_trades)} trades).")
        except Exception as e:
            logger.error(f"[State] Failed to load state from {self.state_file}: {e}")

    def save_state(self) -> None:
        """Persists bot state, active positions, and completed trade ledger."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "is_testnet": self.is_testnet,
                "active_position": asdict(self.active_position) if self.active_position else None,
                "pending_order": asdict(self.pending_order) if self.pending_order else None,
                "traded_in_current_episode": self.traded_in_current_episode,
                "last_state_id": self.last_state_id,
                "state_age": self.state_age,
                "completed_trades": self.completed_trades[-100:],  # Store recent 100
            }

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            logger.debug(f"[State] State persisted to {self.state_file}")
        except Exception as e:
            logger.error(f"[State] Failed to save state to {self.state_file}: {e}")

    def fetch_recent_candles(self, limit: int = 250) -> pd.DataFrame:
        """
        Fetches historical OHLCV candles from Binance via CCXT.
        Falls back to public mainnet API if testnet has insufficient data.
        """
        raw_candles = None
        try:
            raw_candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        except Exception as e:
            logger.warning(f"[MarketData] Error fetching from current exchange configuration: {e}")
            if self.is_testnet:
                # Fallback to public mainnet for market data reading
                logger.info("[MarketData] Falling back to public Binance endpoint for OHLCV ingestion...")
                pub_ex = ccxt.binance({"enableRateLimit": True})
                raw_candles = pub_ex.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)

        if not raw_candles or len(raw_candles) < 60:
            raise ValueError(f"Insufficient candle data fetched ({len(raw_candles) if raw_candles else 0} bars). Minimum 60 required.")

        df = pd.DataFrame(
            raw_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.sort_values(by="datetime", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def evaluate_market(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation:
          1. Compute all indicators and causal microstructure features
          2. Forward-filter Gaussian HMM states (Zero Lookahead Bias)
          3. Evaluate Pilar 1 (Macro Gate) and Pilar 2 (Micro Pullback Trigger)
          4. Calculate Pilar 3 risk boundaries
        """
        if df is None:
            df = self.fetch_recent_candles(limit=350)

        # 1. Compute Indicators
        df = self.strategy.compute_indicators(df)
        df.dropna(subset=self.strategy.feature_columns + ["atr", "ema_9", "rsi_14", "ema_200"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 2. Causal Online Inference
        causal_states, _ = self.strategy.compute_causal_states(df)
        df["regime_state"] = causal_states

        latest_idx = len(df) - 1
        last_bar = df.iloc[latest_idx]
        current_state = int(last_bar["regime_state"])

        # Episode and State Age Tracking
        if current_state == self.strategy.bullish_state_id:
            if self.last_state_id == self.strategy.bullish_state_id:
                state_age = self.state_age + 1
            else:
                state_age = 1
                self.traded_in_current_episode = False
        else:
            state_age = 0
            self.traded_in_current_episode = False

        # 3. Evaluate Gates
        signal_passed, gate_details = self.strategy.evaluate_gates(
            df=df,
            idx=latest_idx,
            current_state=current_state,
            state_age=state_age,
            traded_in_episode=self.traded_in_current_episode,
        )

        close_p = float(last_bar["close"])
        ema_200 = float(last_bar["ema_200"])
        ema_9 = float(last_bar["ema_9"])
        rsi_14 = float(last_bar["rsi_14"])
        atr_14 = float(last_bar["atr_14"])

        # 4. Projected Pilar 3 Risk Boundaries
        entry_target_price = round(ema_9, 2)
        risk_brackets = self.strategy.calculate_risk_brackets(entry_target_price, atr_14)

        return {
            "datetime": str(last_bar["datetime"]),
            "close": close_p,
            "ema_200": ema_200,
            "ema_9": ema_9,
            "rsi_14": rsi_14,
            "atr_14": atr_14,
            "current_state": current_state,
            "state_age": state_age,
            "traded_in_episode": self.traded_in_current_episode,
            "pilar_1": gate_details["pilar_1"],
            "pilar_2": gate_details["pilar_2"],
            "signal_passed": signal_passed,
            "entry_target_price": entry_target_price,
            "risk_brackets": risk_brackets,
            "df": df,
        }

    def format_precision(self, amount: float, price: float) -> Tuple[float, float]:
        """Formats order quantity and price conforming to exchange specifications."""
        try:
            if hasattr(self.exchange, "amount_to_precision") and callable(self.exchange.amount_to_precision):
                res = self.exchange.amount_to_precision(self.symbol, amount)
                if isinstance(res, (int, float, str)) and not isinstance(res, bool):
                    amount = float(res)
                else:
                    amount = round(amount, 6)
            else:
                amount = round(amount, 6)
        except Exception:
            amount = round(amount, 6)

        try:
            if hasattr(self.exchange, "price_to_precision") and callable(self.exchange.price_to_precision):
                res = self.exchange.price_to_precision(self.symbol, price)
                if isinstance(res, (int, float, str)) and not isinstance(res, bool):
                    price = float(res)
                else:
                    price = round(price, 2)
            else:
                price = round(price, 2)
        except Exception:
            price = round(price, 2)

        return amount, price

    def check_pending_order(self, current_bar_time: str) -> None:
        """
        Monitors open limit order. If unfilled by the next hourly bar, cancel it
        to prevent stale execution out of signal context.
        """
        if self.pending_order is None:
            return

        order_id = self.pending_order.order_id
        logger.info(f"[Order] Checking status of pending Maker Limit Order #{order_id}...")

        try:
            order = self.exchange.fetch_order(order_id, self.symbol)
            status = order.get("status", "open").lower()

            if status == "filled":
                fill_price = float(order.get("average", order.get("price", self.pending_order.price)))
                filled_qty = float(order.get("filled", self.pending_order.amount))
                logger.info(f"[Order] Order #{order_id} FILLED! Price: ${fill_price:,.2f}, Qty: {filled_qty:.6f}")

                # Establish Pilar 3 Active Position
                brackets = self.strategy.calculate_risk_brackets(fill_price, self.pending_order.signal_atr)
                self.active_position = ActivePosition(
                    position_id=f"POS-{int(time.time())}",
                    symbol=self.symbol,
                    entry_price=fill_price,
                    initial_quantity=filled_qty,
                    remaining_quantity=filled_qty,
                    tp1_price=brackets["tp1_price"],
                    tp2_price=brackets["tp2_price"],
                    sl_price=brackets["sl_price"],
                    be_price=brackets["be_price"],
                    tp1_hit=False,
                    entry_time=current_bar_time,
                    bars_held=0,
                )
                self.pending_order = None
                self.save_state()
                return

            elif status in ("canceled", "rejected", "expired"):
                logger.info(f"[Order] Order #{order_id} closed with status '{status}'. Clearing pending.")
                self.pending_order = None
                self.save_state()
                return

            # If still unfilled and bar has rolled over -> Cancel
            if self.pending_order.expire_after_bar_time and current_bar_time != self.pending_order.expire_after_bar_time:
                logger.info(f"[Order] Order #{order_id} unfilled past bar {self.pending_order.expire_after_bar_time}. Canceling...")
                try:
                    self.exchange.cancel_order(order_id, self.symbol)
                except Exception as cancel_err:
                    logger.warning(f"[Order] Cancel order returned: {cancel_err}")
                self.pending_order = None
                self.save_state()

        except Exception as e:
            logger.error(f"[Order] Error querying order #{order_id}: {e}")

    def place_maker_limit_buy(self, price: float, atr: float, current_bar_time: str) -> Optional[PendingOrder]:
        """
        Submits passive Maker Limit Order at queue level (EMA 9).
        """
        allocation = self.config.trade_allocation
        raw_qty = allocation / price
        qty, clean_price = self.format_precision(raw_qty, price)

        logger.info(
            f"[Order] Placing Maker Limit BUY Order: {self.symbol} | "
            f"Qty: {qty:.6f} | Price: ${clean_price:,.2f} | Notional: ~${allocation:.2f} USDT"
        )

        try:
            res = self.exchange.create_limit_buy_order(self.symbol, qty, clean_price)
            order_id = str(res["id"])
            self.pending_order = PendingOrder(
                order_id=order_id,
                symbol=self.symbol,
                price=clean_price,
                amount=qty,
                signal_atr=atr,
                created_at_time=datetime.now(timezone.utc).isoformat(),
                expire_after_bar_time=current_bar_time,
            )
            self.traded_in_current_episode = True
            self.save_state()
            logger.info(f"[Order] Limit BUY order successfully placed. Order ID: #{order_id}")
            return self.pending_order
        except Exception as e:
            logger.error(f"[Order] Failed to create limit buy order: {e}")
            return None

    def monitor_active_position(self, current_price: float, current_regime_state: Optional[int] = None) -> None:
        """
        Pilar 3 Real-time Position Management & Scaling Out:
          - TP1 (50% scale out): current_price >= tp1_price -> Sell 50%, lock BE ratchet
          - TP2 (Remaining 50%): current_price >= tp2_price -> Sell remaining 50%
          - SL (Stop Loss): current_price <= sl_price -> Sell all remaining
          - Emergency Regime Change: Bearish dump (State 3) -> Immediate market exit
        """
        if self.active_position is None:
            return

        pos = self.active_position
        hit_sl = current_price <= pos.sl_price
        hit_tp1 = (not pos.tp1_hit) and (current_price >= pos.tp1_price)
        hit_tp2 = (pos.tp1_hit) and (current_price >= pos.tp2_price)
        hit_emergency = (current_regime_state == self.strategy.bearish_state_id)

        # 1. Stop Loss Check
        if hit_sl:
            reason = "BREAK_EVEN_SL" if pos.tp1_hit else "STOP_LOSS"
            logger.warning(
                f"[Position] {reason} TRIGGERED! Price: ${current_price:,.2f} <= SL: ${pos.sl_price:,.2f}. "
                f"Closing remaining {pos.remaining_quantity:.6f} {pos.symbol}..."
            )
            self._close_position_market(pos.remaining_quantity, current_price, reason)
            return

        # 2. TP1 Scale-Out (50%)
        if hit_tp1:
            sell_qty = pos.initial_quantity * 0.5
            sell_qty, _ = self.format_precision(sell_qty, current_price)
            logger.info(
                f"[Position] TARGET 1 (TP1) HIT! Price: ${current_price:,.2f} >= TP1: ${pos.tp1_price:,.2f}. "
                f"Scaling out 50% ({sell_qty:.6f} {pos.symbol})..."
            )
            try:
                self.exchange.create_market_sell_order(self.symbol, sell_qty)
                pos.remaining_quantity -= sell_qty
                pos.tp1_hit = True
                # Ratchet SL to Hard-Floored Break-Even (+0.25% net buffer)
                pos.sl_price = max(pos.sl_price, pos.be_price)
                logger.info(f"[Position] Stop loss ratcheted to Break-Even: ${pos.sl_price:,.2f}")
                self.save_state()
            except Exception as e:
                logger.error(f"[Position] Failed to execute TP1 sell order: {e}")
            return

        # 3. TP2 Full Take Profit (Remaining 50%)
        if hit_tp2:
            logger.info(
                f"[Position] TARGET 2 (TP2) HIT! Price: ${current_price:,.2f} >= TP2: ${pos.tp2_price:,.2f}. "
                f"Closing remaining {pos.remaining_quantity:.6f} {pos.symbol} (Trade Complete)..."
            )
            self._close_position_market(pos.remaining_quantity, current_price, "TAKE_PROFIT_2")
            return

        # 4. Emergency Regime Exit
        if hit_emergency:
            logger.warning(
                f"[Position] EMERGENCY REGIME CHANGE! State {current_regime_state} (Bearish Dump). "
                f"Closing remaining {pos.remaining_quantity:.6f} {pos.symbol}..."
            )
            self._close_position_market(pos.remaining_quantity, current_price, "EMERGENCY_REGIME_CHANGE")
            return

    def _close_position_market(self, quantity: float, exit_price: float, exit_reason: str) -> None:
        """Executes full position closure at market price."""
        clean_qty, _ = self.format_precision(quantity, exit_price)
        try:
            self.exchange.create_market_sell_order(self.symbol, clean_qty)
            gross_pnl = (exit_price - self.active_position.entry_price) * clean_qty
            trade_record = {
                "position_id": self.active_position.position_id,
                "symbol": self.symbol,
                "entry_price": self.active_position.entry_price,
                "exit_price": exit_price,
                "quantity": clean_qty,
                "gross_pnl": gross_pnl,
                "exit_reason": exit_reason,
                "entry_time": self.active_position.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            }
            self.completed_trades.append(trade_record)
            self.active_position = None
            self.save_state()
            logger.info(f"[Position] Position closed ({exit_reason}). PnL: ${gross_pnl:+,.2f} USDT.")
        except Exception as e:
            logger.error(f"[Position] Error executing market sell order: {e}")

    def run_once(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Executes a single market evaluation and prints a formatted diagnostic report.
        In dry-run mode, no real orders are sent to the exchange.
        """
        eval_res = self.evaluate_market()
        current_price = eval_res["close"]
        current_state = eval_res["current_state"]
        self.last_state_id = current_state
        self.state_age = eval_res["state_age"]
        self.save_state()

        # Terminal Visual Report
        print("\n" + "=" * 80)
        print(f"{'ADAPTIVE TRADING SYSTEM - REGIME CORONG LIVE ENGINE':^80}")
        print("=" * 80)
        mode_str = "DRY-RUN (Simulated)" if dry_run else f"LIVE ({'TESTNET' if self.is_testnet else 'MAINNET'})"
        print(f"  Mode           : {mode_str}")
        print(f"  Symbol / TF    : {self.symbol} [{self.timeframe}]")
        print(f"  Bar Datetime   : {eval_res['datetime']}")
        print(f"  Current Price  : ${current_price:,.2f}")
        print("-" * 80)
        print(f"  EMA(200) Trend : ${eval_res['ema_200']:,.2f} ({'BULLISH (Close > EMA200)' if current_price > eval_res['ema_200'] else 'BEARISH (Close <= EMA200)'})")
        print(f"  EMA(9) Micro   : ${eval_res['ema_9']:,.2f}")
        print(f"  RSI(14)        : {eval_res['rsi_14']:.2f}")
        print(f"  ATR(14)        : ${eval_res['atr_14']:,.2f}")
        print("-" * 80)
        print(f"  HMM State      : State {current_state} ({'BULLISH' if current_state == self.strategy.bullish_state_id else 'BEARISH/SIDEWAYS'})")
        print(f"  State Age      : {self.state_age} bar(s)")
        print(f"  Single-Shot    : {'AVAILABLE' if not self.traded_in_current_episode else 'ALREADY TRADED THIS EPISODE'}")
        print("-" * 80)
        p1_status = "[PASSED]" if eval_res["pilar_1"] else "[REJECTED]"
        p2_status = "[PASSED]" if eval_res["pilar_2"] else "[REJECTED]"
        gate_status = "GATE OPEN (BUY SIGNAL)" if eval_res["signal_passed"] else "GATE CLOSED (WAITING)"
        print(f"  Pilar 1 (Macro Gate)     : {p1_status}")
        print(f"  Pilar 2 (Micro Pullback) : {p2_status}")
        print(f"  Cascading Decision       : {gate_status}")

        brackets = eval_res["risk_brackets"]
        print("-" * 80)
        print("  Pilar 3 Execution Boundaries (if triggered):")
        print(f"    - Target Limit Buy (EMA 9) : ${eval_res['entry_target_price']:,.2f}")
        print(f"    - Initial Stop Loss (0.70x): ${brackets['sl_price']:,.2f}")
        print(f"    - TP 1 Scale Out (0.80x)   : ${brackets['tp1_price']:,.2f} (Close 50%)")
        print(f"    - TP 2 Full Target (1.20x) : ${brackets['tp2_price']:,.2f} (Close 50%)")
        print(f"    - Break-Even Ratchet Lock  : ${brackets['be_price']:,.2f} (+0.25% fee buffer)")
        print("-" * 80)
        pos_str = f"Active ({self.active_position.position_id} @ ${self.active_position.entry_price:,.2f})" if self.active_position else "None"
        ord_str = f"Pending (#{self.pending_order.order_id} @ ${self.pending_order.price:,.2f})" if self.pending_order else "None"
        print(f"  Active Position : {pos_str}")
        print(f"  Pending Order   : {ord_str}")
        print(f"  State Saved To  : {self.state_file}")
        print("=" * 80 + "\n")

        return eval_res

    def start(self, poll_interval: int = 20) -> None:
        """
        Continuous scheduling loop:
          - Hourly bar evaluation at minute 00:05 (5 seconds after candle close)
          - High-frequency tick monitoring every poll_interval seconds (15-30s) for TP/SL execution
        """
        self.is_running = True
        logger.info(f"[LiveBot] Starting engine loop for {self.symbol} [{self.timeframe}] (Tick interval: {poll_interval}s)")

        last_evaluated_bar: Optional[str] = None

        while self.is_running:
            try:
                now_utc = datetime.now(timezone.utc)
                minute = now_utc.minute
                second = now_utc.second

                # Hourly Candle Close Trigger: At minute 00, between 05s and 25s
                is_hourly_cycle = (minute == 0) and (5 <= second <= 35)

                if is_hourly_cycle:
                    df = self.fetch_recent_candles(limit=250)
                    latest_bar_time = str(df["datetime"].iloc[-1])

                    if latest_bar_time != last_evaluated_bar:
                        logger.info(f"[Hourly] New 1H candle closed: {latest_bar_time}. Running evaluation...")
                        last_evaluated_bar = latest_bar_time

                        # 1. Check & cancel expired pending orders
                        self.check_pending_order(latest_bar_time)

                        # 2. Evaluate market gates
                        eval_res = self.evaluate_market(df)
                        current_state = eval_res["current_state"]
                        self.last_state_id = current_state
                        self.state_age = eval_res["state_age"]

                        # 3. Check for new buy signal
                        if eval_res["signal_passed"] and self.active_position is None and self.pending_order is None:
                            logger.info("[Hourly] GATE OPEN! Submitting Maker Limit Buy order...")
                            self.place_maker_limit_buy(
                                price=eval_res["entry_target_price"],
                                atr=eval_res["atr_14"],
                                current_bar_time=latest_bar_time,
                            )
                        else:
                            logger.info(f"[Hourly] Evaluation complete. Status: {p1_status if 'p1_status' in locals() else 'Evaluated'}")

                        self.save_state()

                # High-frequency tick monitor for active positions
                if self.active_position is not None:
                    try:
                        ticker = self.exchange.fetch_ticker(self.symbol)
                        current_price = float(ticker.get("last", ticker.get("close", 0.0)))
                        if current_price > 0:
                            self.monitor_active_position(current_price, self.last_state_id)
                    except Exception as tick_err:
                        logger.warning(f"[Monitor] Error fetching ticker: {tick_err}")

                # Sleep until next tick
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.warning("[LiveBot] Interrupted by user. Stopping...")
                self.stop()
                break
            except Exception as e:
                logger.error(f"[LiveBot] Unexpected exception in main loop: {e}", exc_info=True)
                time.sleep(10)

    def stop(self) -> None:
        """Stops the bot and guarantees state persistence."""
        self.is_running = False
        self.save_state()
        logger.info("[LiveBot] Bot stopped cleanly and state persisted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Trading System - Live Regime Bot Engine")
    parser.add_argument("--dry-run-once", action="store_true", help="Run a single evaluation cycle in dry-run mode and exit")
    parser.add_argument("--symbol", type=str, default=None, help="Trading pair symbol (default: BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Trading timeframe (default: 1h)")
    parser.add_argument("--poll-interval", type=int, default=20, help="Tick polling interval in seconds (default: 20)")
    parser.add_argument("--state-file", type=str, default="data/live_bot_state.json", help="Path to state JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = LiveTradingBot(
        symbol=args.symbol,
        timeframe=args.timeframe,
        state_file=Path(args.state_file),
    )

    if args.dry_run_once:
        logger.info("Executing dry-run market evaluation cycle...")
        bot.run_once(dry_run=True)
        sys.exit(0)

    bot.start(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
