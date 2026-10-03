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
    Now upgraded to Multi-Pair Scanner (BTC/USDT, ETH/USDT, SOL/USDT) with portfolio quota.
    """

    def __init__(
        self,
        config: Optional[RegimeFunnelConfig] = None,
        exchange: Optional[ccxt.Exchange] = None,
        symbol: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        timeframe: Optional[str] = None,
        state_file: Optional[Path] = None,
        is_testnet: Optional[bool] = None,
        app_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.config: RegimeFunnelConfig = config or getattr(app_config, "regime_funnel", REGIME_FUNNEL)

        # Multi-Pair Symbols configuration
        symbols_env = os.getenv("SYMBOLS", "")
        if symbols:
            self.symbols = [s.strip().upper() for s in symbols]
        elif symbols_env:
            self.symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
        elif symbol:
            self.symbols = [symbol.strip().upper()]
        else:
            self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        self.symbol = symbol or self.symbols[0]
        self.timeframe = timeframe or os.getenv("TIMEFRAME", "1h")
        self.state_file = Path(state_file or os.getenv("STATE_FILE", "data/live_bot_state.json"))
        
        testnet_env = os.getenv("IS_TESTNET", "true").lower() in ("true", "1", "yes")
        self.is_testnet = is_testnet if is_testnet is not None else testnet_env

        self.max_concurrent_positions: int = int(getattr(self.config, "max_concurrent_positions", 3))
        self.allocation_per_trade_usd: float = float(getattr(self.config, "trade_allocation", 100.0))

        # CCXT Exchange Client initialization
        self.exchange = exchange or self._init_exchange()

        # Operational state per symbol
        self.active_positions: Dict[str, Optional[ActivePosition]] = {s: None for s in self.symbols}
        self.pending_orders: Dict[str, Optional[PendingOrder]] = {s: None for s in self.symbols}
        self.traded_in_episodes: Dict[str, bool] = {s: False for s in self.symbols}
        self.last_state_ids: Dict[str, Optional[int]] = {s: None for s in self.symbols}
        self.state_ages: Dict[str, int] = {s: 0 for s in self.symbols}
        self.completed_trades: List[Dict[str, Any]] = []
        self.is_running: bool = False

        # Build Strategy instances per symbol (with coin-specific HMM and shared multi-asset meta-labeler)
        project_root = Path(__file__).resolve().parent.parent.parent
        self.strategies: Dict[str, RegimeFunnelStrategy] = {}
        for sym in self.symbols:
            coin_tag = sym.split("/")[0].lower()
            coin_hmm = project_root / "models" / f"{coin_tag}_1h_regime_hmm.joblib"
            hmm_path = str(coin_hmm) if coin_hmm.exists() else self.config.hmm_model_path

            sym_cfg = RegimeFunnelConfig(
                hmm_model_path=hmm_path,
                use_kalman_filter=self.config.use_kalman_filter,
                kalman_q=self.config.kalman_q,
                kalman_r=self.config.kalman_r,
                use_volume_filter=self.config.use_volume_filter,
                volume_filter_mult=self.config.volume_filter_mult,
                single_shot_per_episode=self.config.single_shot_per_episode,
                use_meta_labeler=self.config.use_meta_labeler,
                meta_label_threshold=self.config.meta_label_threshold,
                meta_model_path=self.config.meta_model_path,
                trade_allocation=self.allocation_per_trade_usd,
                max_concurrent_positions=self.max_concurrent_positions,
            )
            self.strategies[sym] = RegimeFunnelStrategy(config=sym_cfg)

        # Load persisted state if exists
        self.load_state()

        # Log Multi-Pair Scanner and AI Meta-Labeling Decider initialization status at startup
        meta_status_logged = False
        for sym, strat in self.strategies.items():
            if getattr(strat.config, "use_meta_labeler", False) and not meta_status_logged:
                if strat.meta_gate is not None and strat.meta_gate.model is not None:
                    m_name = (
                        strat.meta_gate.resolved_model_path.name
                        if strat.meta_gate.resolved_model_path
                        else "LightGBM"
                    )
                    logger.info(
                        f"[Startup] Layer-2 AI Meta-Labeling Gate INITIALIZED! "
                        f"Model='{m_name}', Threshold={strat.config.meta_label_threshold:.2f}"
                    )
                    meta_status_logged = True

        logger.info(
            f"[Startup] Multi-Pair Scanner INITIALIZED for {self.symbols} "
            f"(Max Concurrent Positions: {self.max_concurrent_positions}, Allocation: ${self.allocation_per_trade_usd:,.2f}/trade)"
        )

    # Backward-compatible property delegates for single-symbol test cases
    @property
    def strategy(self) -> RegimeFunnelStrategy:
        return self.strategies.get(self.symbol, next(iter(self.strategies.values())))

    @strategy.setter
    def strategy(self, strat: RegimeFunnelStrategy) -> None:
        self.strategies[self.symbol] = strat

    @property
    def active_position(self) -> Optional[ActivePosition]:
        return self.active_positions.get(self.symbol)

    @active_position.setter
    def active_position(self, pos: Optional[ActivePosition]) -> None:
        if pos is not None:
            self.active_positions[pos.symbol] = pos
        else:
            self.active_positions[self.symbol] = None

    @property
    def pending_order(self) -> Optional[PendingOrder]:
        return self.pending_orders.get(self.symbol)

    @pending_order.setter
    def pending_order(self, ord: Optional[PendingOrder]) -> None:
        if ord is not None:
            self.pending_orders[ord.symbol] = ord
        else:
            self.pending_orders[self.symbol] = None

    @property
    def traded_in_current_episode(self) -> bool:
        return self.traded_in_episodes.get(self.symbol, False)

    @traded_in_current_episode.setter
    def traded_in_current_episode(self, val: bool) -> None:
        self.traded_in_episodes[self.symbol] = val

    @property
    def last_state_id(self) -> Optional[int]:
        return self.last_state_ids.get(self.symbol, None)

    @last_state_id.setter
    def last_state_id(self, val: Optional[int]) -> None:
        self.last_state_ids[self.symbol] = val

    @property
    def state_age(self) -> int:
        return self.state_ages.get(self.symbol, 0)

    @state_age.setter
    def state_age(self, val: int) -> None:
        self.state_ages[self.symbol] = val

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

            if "active_positions" in data and isinstance(data["active_positions"], dict):
                for s, p_data in data["active_positions"].items():
                    if p_data:
                        self.active_positions[s] = ActivePosition(**p_data)
                        logger.info(f"[State] Restored active position for {s}: @ ${self.active_positions[s].entry_price:,.2f}")
            elif data.get("active_position"):
                pos = ActivePosition(**data["active_position"])
                self.active_positions[pos.symbol] = pos
                logger.info(f"[State] Restored active position: {pos.symbol} @ ${pos.entry_price:,.2f}")

            if "pending_orders" in data and isinstance(data["pending_orders"], dict):
                for s, o_data in data["pending_orders"].items():
                    if o_data:
                        self.pending_orders[s] = PendingOrder(**o_data)
                        logger.info(f"[State] Restored pending order for {s}: #{self.pending_orders[s].order_id}")
            elif data.get("pending_order"):
                ord_obj = PendingOrder(**data["pending_order"])
                self.pending_orders[ord_obj.symbol] = ord_obj
                logger.info(f"[State] Restored pending order: #{ord_obj.order_id}")

            if "last_state_ids" in data and isinstance(data["last_state_ids"], dict):
                self.last_state_ids.update(data["last_state_ids"])
            elif "last_state_id" in data:
                self.last_state_ids[self.symbol] = data["last_state_id"]

            if "state_ages" in data and isinstance(data["state_ages"], dict):
                self.state_ages.update(data["state_ages"])
            elif "state_age" in data:
                self.state_ages[self.symbol] = data["state_age"]

            if "traded_in_episodes" in data and isinstance(data["traded_in_episodes"], dict):
                self.traded_in_episodes.update(data["traded_in_episodes"])
            elif "traded_in_current_episode" in data:
                self.traded_in_episodes[self.symbol] = data["traded_in_current_episode"]

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
                "symbols": self.symbols,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "is_testnet": self.is_testnet,
                "max_concurrent_positions": self.max_concurrent_positions,
                "allocation_per_trade_usd": self.allocation_per_trade_usd,
                "active_positions": {
                    s: asdict(pos) if pos else None for s, pos in self.active_positions.items()
                },
                "active_position": asdict(self.active_position) if self.active_position else None,
                "pending_orders": {
                    s: asdict(ord) if ord else None for s, ord in self.pending_orders.items()
                },
                "pending_order": asdict(self.pending_order) if self.pending_order else None,
                "traded_in_episodes": self.traded_in_episodes,
                "traded_in_current_episode": self.traded_in_current_episode,
                "last_state_ids": self.last_state_ids,
                "last_state_id": self.last_state_id,
                "state_ages": self.state_ages,
                "state_age": self.state_age,
                "completed_trades": self.completed_trades[-100:],  # Store recent 100
            }

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            logger.debug(f"[State] State persisted to {self.state_file}")
        except Exception as e:
            logger.error(f"[State] Failed to save state to {self.state_file}: {e}")

    def fetch_recent_candles(self, symbol: Optional[str] = None, limit: int = 250) -> pd.DataFrame:
        """
        Fetches historical OHLCV candles from Binance via CCXT.
        Falls back to public mainnet API if testnet has insufficient data.
        """
        sym = symbol or self.symbol
        raw_candles = None
        try:
            raw_candles = self.exchange.fetch_ohlcv(sym, timeframe=self.timeframe, limit=limit)
        except Exception as e:
            logger.warning(f"[MarketData] Error fetching {sym} from current exchange configuration: {e}")
            if self.is_testnet:
                # Fallback to public mainnet for market data reading
                logger.info(f"[MarketData] Falling back to public Binance endpoint for {sym} OHLCV ingestion...")
                pub_ex = ccxt.binance({"enableRateLimit": True})
                raw_candles = pub_ex.fetch_ohlcv(sym, timeframe=self.timeframe, limit=limit)

        if not raw_candles or len(raw_candles) < 60:
            raise ValueError(f"Insufficient candle data fetched for {sym} ({len(raw_candles) if raw_candles else 0} bars). Minimum 60 required.")

        df = pd.DataFrame(
            raw_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.sort_values(by="datetime", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def evaluate_market(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        btc_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation for a specific symbol:
          1. Compute all indicators and causal microstructure features
          2. Forward-filter Gaussian HMM states (Zero Lookahead Bias)
          3. Evaluate Pilar 1 (Macro Gate) and Pilar 2 (Micro Pullback Trigger)
          4. Calculate Pilar 3 risk boundaries
        """
        sym = symbol or self.symbol
        strat = self.strategies.get(sym, self.strategy)

        if df is None:
            df = self.fetch_recent_candles(symbol=sym, limit=350)

        # 1. Compute Indicators
        df = strat.compute_indicators(df)
        df.dropna(subset=strat.feature_columns + ["atr", "ema_9", "rsi_14", "ema_200"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 2. Causal Online Inference
        causal_states, _ = strat.compute_causal_states(df)
        df["regime_state"] = causal_states

        latest_idx = len(df) - 1
        last_bar = df.iloc[latest_idx]
        current_state = int(last_bar["regime_state"])

        # Episode and State Age Tracking per symbol
        last_st = self.last_state_ids.get(sym)
        prev_age = self.state_ages.get(sym, 0)
        traded_in_ep = self.traded_in_episodes.get(sym, False)

        if current_state == strat.bullish_state_id:
            if last_st == strat.bullish_state_id:
                state_age = prev_age + 1
            else:
                state_age = 1
                traded_in_ep = False
        else:
            state_age = 0
            traded_in_ep = False

        self.traded_in_episodes[sym] = traded_in_ep

        # 3. Evaluate Gates (with btc_df for cross-asset features if applicable)
        signal_passed, gate_details = strat.evaluate_gates(
            df=df,
            idx=latest_idx,
            current_state=current_state,
            state_age=state_age,
            traded_in_episode=traded_in_ep,
            btc_df=btc_df,
        )

        close_p = float(last_bar["close"])
        ema_200 = float(last_bar["ema_200"])
        ema_9 = float(last_bar["ema_9"])
        rsi_14 = float(last_bar["rsi_14"])
        atr_14 = float(last_bar["atr_14"])

        # 4. Projected Pilar 3 Risk Boundaries
        entry_target_price = round(ema_9, 2)
        risk_brackets = strat.calculate_risk_brackets(entry_target_price, atr_14)

        return {
            "symbol": sym,
            "datetime": str(last_bar["datetime"]),
            "close": close_p,
            "ema_200": ema_200,
            "ema_9": ema_9,
            "rsi_14": rsi_14,
            "atr_14": atr_14,
            "current_state": current_state,
            "state_age": state_age,
            "traded_in_episode": traded_in_ep,
            "pilar_1": gate_details["pilar_1"],
            "pilar_2": gate_details["pilar_2"],
            "pilar_vol": gate_details.get("pilar_vol", True),
            "pilar_meta": gate_details.get("pilar_meta", True),
            "meta_prob": gate_details.get("meta_prob", 0.0),
            "signal_passed": signal_passed,
            "entry_target_price": entry_target_price,
            "risk_brackets": risk_brackets,
            "df": df,
        }

    def format_precision(self, amount: float, price: float, symbol: Optional[str] = None) -> Tuple[float, float]:
        """Formats order quantity and price conforming to exchange specifications."""
        sym = symbol or self.symbol
        try:
            if hasattr(self.exchange, "amount_to_precision") and callable(self.exchange.amount_to_precision):
                res = self.exchange.amount_to_precision(sym, amount)
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
                res = self.exchange.price_to_precision(sym, price)
                if isinstance(res, (int, float, str)) and not isinstance(res, bool):
                    price = float(res)
                else:
                    price = round(price, 2)
            else:
                price = round(price, 2)
        except Exception:
            price = round(price, 2)

        return amount, price

    def check_pending_order(self, current_bar_time: str, symbol: Optional[str] = None) -> None:
        """
        Monitors open limit order for a given symbol. If unfilled by the next hourly bar,
        cancel it to prevent stale execution out of signal context.
        """
        sym = symbol or self.symbol
        pending = self.pending_orders.get(sym)
        if pending is None:
            return

        order_id = pending.order_id
        logger.info(f"[Order] Checking status of pending Maker Limit Order #{order_id} for {sym}...")

        try:
            order = self.exchange.fetch_order(order_id, sym)
            status = order.get("status", "open").lower()

            if status == "filled":
                fill_price = float(order.get("average", order.get("price", pending.price)))
                filled_qty = float(order.get("filled", pending.amount))
                logger.info(f"[Order] Order #{order_id} ({sym}) FILLED! Price: ${fill_price:,.2f}, Qty: {filled_qty:.6f}")

                # Establish Pilar 3 Active Position
                strat = self.strategies.get(sym, self.strategy)
                brackets = strat.calculate_risk_brackets(fill_price, pending.signal_atr)
                self.active_positions[sym] = ActivePosition(
                    position_id=f"POS-{sym.replace('/', '')}-{int(time.time())}",
                    symbol=sym,
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
                self.pending_orders[sym] = None
                self.save_state()
                return

            elif status in ("canceled", "rejected", "expired"):
                logger.info(f"[Order] Order #{order_id} ({sym}) closed with status '{status}'. Clearing pending.")
                self.pending_orders[sym] = None
                self.save_state()
                return

            # If still unfilled and bar has rolled over -> Cancel
            if pending.expire_after_bar_time and current_bar_time != pending.expire_after_bar_time:
                logger.info(f"[Order] Order #{order_id} ({sym}) unfilled past bar {pending.expire_after_bar_time}. Canceling...")
                try:
                    self.exchange.cancel_order(order_id, sym)
                except Exception as cancel_err:
                    logger.warning(f"[Order] Cancel order returned: {cancel_err}")
                self.pending_orders[sym] = None
                self.save_state()

        except Exception as e:
            logger.error(f"[Order] Error querying order #{order_id} ({sym}): {e}")

    def place_maker_limit_buy(
        self,
        price: float,
        atr: float,
        current_bar_time: str,
        symbol: Optional[str] = None,
    ) -> Optional[PendingOrder]:
        """
        Submits passive Maker Limit Order at queue level (EMA 9) for a given symbol.
        """
        sym = symbol or self.symbol
        allocation = self.allocation_per_trade_usd
        raw_qty = allocation / price
        qty, clean_price = self.format_precision(raw_qty, price, symbol=sym)

        logger.info(
            f"[Order] Placing Maker Limit BUY Order: {sym} | "
            f"Qty: {qty:.6f} | Price: ${clean_price:,.2f} | Notional: ~${allocation:.2f} USDT"
        )

        try:
            res = self.exchange.create_limit_buy_order(sym, qty, clean_price)
            order_id = str(res["id"])
            pending_obj = PendingOrder(
                order_id=order_id,
                symbol=sym,
                price=clean_price,
                amount=qty,
                signal_atr=atr,
                created_at_time=datetime.now(timezone.utc).isoformat(),
                expire_after_bar_time=current_bar_time,
            )
            self.pending_orders[sym] = pending_obj
            self.traded_in_episodes[sym] = True
            self.save_state()
            logger.info(f"[Order] Limit BUY order successfully placed for {sym}. Order ID: #{order_id}")
            return pending_obj
        except Exception as e:
            logger.error(f"[Order] Failed to create limit buy order for {sym}: {e}")
            return None

    def monitor_active_position(
        self,
        current_price: float,
        current_regime_state: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """
        Pilar 3 Real-time Position Management & Scaling Out for a symbol:
          - TP1 (50% scale out): current_price >= tp1_price -> Sell 50%, lock BE ratchet
          - TP2 (Remaining 50%): current_price >= tp2_price -> Sell remaining 50%
          - SL (Stop Loss): current_price <= sl_price -> Sell all remaining
          - Emergency Regime Change: Bearish dump -> Immediate market exit
        """
        sym = symbol or self.symbol
        pos = self.active_positions.get(sym)
        if pos is None:
            return

        strat = self.strategies.get(sym, self.strategy)
        hit_sl = current_price <= pos.sl_price
        hit_tp1 = (not pos.tp1_hit) and (current_price >= pos.tp1_price)
        hit_tp2 = (pos.tp1_hit) and (current_price >= pos.tp2_price)
        hit_emergency = (current_regime_state == strat.bearish_state_id)

        # 1. Stop Loss Check
        if hit_sl:
            reason = "BREAK_EVEN_SL" if pos.tp1_hit else "STOP_LOSS"
            logger.warning(
                f"[Position] {reason} TRIGGERED for {sym}! Price: ${current_price:,.2f} <= SL: ${pos.sl_price:,.2f}. "
                f"Closing remaining {pos.remaining_quantity:.6f} {sym}..."
            )
            self._close_position_market(pos.remaining_quantity, current_price, reason, symbol=sym)
            return

        # 2. TP1 Scale-Out (50%)
        if hit_tp1:
            sell_qty = pos.initial_quantity * 0.5
            sell_qty, _ = self.format_precision(sell_qty, current_price, symbol=sym)
            logger.info(
                f"[Position] TARGET 1 (TP1) HIT for {sym}! Price: ${current_price:,.2f} >= TP1: ${pos.tp1_price:,.2f}. "
                f"Scaling out 50% ({sell_qty:.6f} {sym})..."
            )
            try:
                self.exchange.create_market_sell_order(sym, sell_qty)
                pos.remaining_quantity -= sell_qty
                pos.tp1_hit = True
                # Ratchet SL to Hard-Floored Break-Even (+0.25% net buffer)
                pos.sl_price = max(pos.sl_price, pos.be_price)
                logger.info(f"[Position] Stop loss ratcheted to Break-Even for {sym}: ${pos.sl_price:,.2f}")
                self.save_state()
            except Exception as e:
                logger.error(f"[Position] Failed to execute TP1 sell order for {sym}: {e}")
            return

        # 3. TP2 Full Take Profit (Remaining 50%)
        if hit_tp2:
            logger.info(
                f"[Position] TARGET 2 (TP2) HIT for {sym}! Price: ${current_price:,.2f} >= TP2: ${pos.tp2_price:,.2f}. "
                f"Closing remaining {pos.remaining_quantity:.6f} {sym} (Trade Complete)..."
            )
            self._close_position_market(pos.remaining_quantity, current_price, "TAKE_PROFIT_2", symbol=sym)
            return

        # 4. Emergency Regime Exit
        if hit_emergency:
            logger.warning(
                f"[Position] EMERGENCY REGIME CHANGE for {sym}! State {current_regime_state} (Bearish Dump). "
                f"Closing remaining {pos.remaining_quantity:.6f} {sym}..."
            )
            self._close_position_market(pos.remaining_quantity, current_price, "EMERGENCY_REGIME_CHANGE", symbol=sym)
            return

    def _close_position_market(
        self,
        quantity: float,
        exit_price: float,
        exit_reason: str,
        symbol: Optional[str] = None,
    ) -> None:
        """Executes full position closure at market price."""
        sym = symbol or self.symbol
        pos = self.active_positions.get(sym)
        if not pos:
            return

        clean_qty, _ = self.format_precision(quantity, exit_price, symbol=sym)
        try:
            self.exchange.create_market_sell_order(sym, clean_qty)
            gross_pnl = (exit_price - pos.entry_price) * clean_qty
            trade_record = {
                "position_id": pos.position_id,
                "symbol": sym,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "quantity": clean_qty,
                "gross_pnl": gross_pnl,
                "exit_reason": exit_reason,
                "entry_time": pos.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            }
            self.completed_trades.append(trade_record)
            self.active_positions[sym] = None
            self.save_state()
            logger.info(f"[Position] Position closed for {sym} ({exit_reason}). PnL: ${gross_pnl:+,.2f} USDT.")
        except Exception as e:
            logger.error(f"[Position] Error executing market sell order for {sym}: {e}")

    def scan_all_pairs(self, dry_run: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Sequentially scans all configured symbols (BTC/USDT, ETH/USDT, SOL/USDT):
          - Applies rate-limiting sleep (0.5s) between pairs
          - Isolated try-except per pair to protect loop integrity
          - Computes cross-asset BTC returns for altcoins
          - Enforces portfolio concurrent position quota (max 3 positions)
        """
        results: Dict[str, Dict[str, Any]] = {}

        # 1. Fetch BTC candles first to supply macro reference for cross-asset features
        btc_df: Optional[pd.DataFrame] = None
        btc_symbol = "BTC/USDT"
        try:
            btc_df = self.fetch_recent_candles(symbol=btc_symbol, limit=350)
        except Exception as btc_err:
            logger.warning(f"[Scanner] Failed to fetch benchmark BTC candles: {btc_err}")

        # Count existing exposure (active positions + pending orders)
        active_count = sum(1 for p in self.active_positions.values() if p is not None) + \
                       sum(1 for o in self.pending_orders.values() if o is not None)

        for sym in self.symbols:
            # Respect Binance API rate limits
            time.sleep(0.5)

            try:
                # Use cached btc_df if scanning BTC itself
                coin_df = btc_df if (sym == btc_symbol and btc_df is not None) else None
                eval_res = self.evaluate_market(df=coin_df, symbol=sym, btc_df=btc_df)
                results[sym] = eval_res

                self.last_state_ids[sym] = eval_res["current_state"]
                self.state_ages[sym] = eval_res["state_age"]

                # Process order logic if signal passed
                if eval_res["signal_passed"]:
                    has_exposure = (self.active_positions.get(sym) is not None) or (self.pending_orders.get(sym) is not None)

                    if not has_exposure:
                        if active_count < self.max_concurrent_positions:
                            if not dry_run:
                                logger.info(f"[Scanner] GATE OPEN for {sym}! Submitting Maker Limit Buy order...")
                                self.place_maker_limit_buy(
                                    price=eval_res["entry_target_price"],
                                    atr=eval_res["atr_14"],
                                    current_bar_time=eval_res["datetime"],
                                    symbol=sym,
                                )
                                active_count += 1
                            else:
                                logger.info(
                                    f"[Scanner] [DRY-RUN] GATE OPEN for {sym}! "
                                    f"Portfolio quota available ({active_count + 1}/{self.max_concurrent_positions})."
                                )
                        else:
                            logger.warning(
                                f"[Scanner] GATE OPEN for {sym}, but Portfolio Quota is FULL "
                                f"({active_count}/{self.max_concurrent_positions} active). Order skipped."
                            )
                    else:
                        logger.info(f"[Scanner] Symmetrical signal on {sym}, but position or order already active. Skipped.")

            except Exception as sym_err:
                logger.error(f"[Scanner] Error evaluating {sym}: {sym_err}", exc_info=True)

        self.save_state()
        return results

    def run_once(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Executes a single market scan across all configured pairs and prints
        a formatted diagnostic multi-pair table. Returns primary symbol result.
        """
        results = self.scan_all_pairs(dry_run=dry_run)

        # Terminal Multi-Pair Diagnostic Table
        print("\n" + "=" * 92)
        print(f"{'ADAPTIVE TRADING SYSTEM - MULTI-PAIR REGIME SCANNER':^92}")
        print("=" * 92)
        mode_str = "DRY-RUN (Simulated)" if dry_run else f"LIVE ({'TESTNET' if self.is_testnet else 'MAINNET'})"
        active_count = sum(1 for p in self.active_positions.values() if p is not None)
        pending_count = sum(1 for o in self.pending_orders.values() if o is not None)
        print(f"  Mode           : {mode_str} | Timeframe: {self.timeframe}")
        print(f"  Portfolio Quota: {active_count + pending_count}/{self.max_concurrent_positions} Slots Used "
              f"(Active: {active_count}, Pending: {pending_count}) | Allocation: ${self.allocation_per_trade_usd:,.2f}/trade")
        print("-" * 92)
        header = f"{'Symbol':<10} | {'Close':<11} | {'HMM State':<14} | {'Age':<4} | {'P1':<6} | {'P2':<6} | {'P_Vol':<6} | {'AI Decider':<16} | {'Signal'}"
        print(header)
        print("-" * 92)

        for sym in self.symbols:
            res = results.get(sym)
            if not res:
                print(f"{sym:<10} | {'ERROR':<11} | {'-':<14} | {'-':<4} | {'-':<6} | {'-':<6} | {'-':<6} | {'-':<16} | FAILED")
                continue

            strat = self.strategies.get(sym, self.strategy)
            st = res["current_state"]
            st_str = f"S{st} ({'BULL' if st == strat.bullish_state_id else 'BEAR/SIDE'})"
            p1_str = "OK" if res["pilar_1"] else "FAIL"
            p2_str = "OK" if res["pilar_2"] else "FAIL"
            p_vol_str = "OK" if res.get("pilar_vol", True) else "FAIL"
            meta_prob = res.get("meta_prob", 0.0)
            meta_str = f"{'OK' if res.get('pilar_meta', True) else 'FAIL'} ({meta_prob:.2f})"
            sig_str = "BUY" if res["signal_passed"] else "WAIT"

            row = (
                f"{sym:<10} | ${res['close']:<10,.2f} | {st_str:<14} | {res['state_age']:<4} | "
                f"{p1_str:<6} | {p2_str:<6} | {p_vol_str:<6} | {meta_str:<16} | {sig_str}"
            )
            print(row)

        print("-" * 92)
        print("  Active Positions & Brackets:")
        for sym in self.symbols:
            pos = self.active_positions.get(sym)
            ord_p = self.pending_orders.get(sym)
            status_desc = "FLAT"
            if pos:
                status_desc = f"LONG ({pos.position_id}) Entry: ${pos.entry_price:,.2f} | TP1: ${pos.tp1_price:,.2f} | TP2: ${pos.tp2_price:,.2f} | SL: ${pos.sl_price:,.2f}"
            elif ord_p:
                status_desc = f"PENDING (#{ord_p.order_id}) Limit Buy @ ${ord_p.price:,.2f}"
            print(f"    - {sym:<10}: {status_desc}")

        print(f"  State Ledger   : {self.state_file}")
        print("=" * 92 + "\n")

        return results.get(self.symbol, next(iter(results.values())))

    def start(self, poll_interval: int = 20) -> None:
        """
        Continuous multi-pair scheduling loop:
          - Hourly bar evaluation at minute 00:05 across BTC, ETH, and SOL
          - High-frequency tick monitoring every poll_interval seconds for all active positions
        """
        self.is_running = True
        logger.info(
            f"[LiveBot] Starting Multi-Pair Scanner loop for {self.symbols} [{self.timeframe}] "
            f"(Tick interval: {poll_interval}s, Quota: {self.max_concurrent_positions})"
        )

        last_evaluated_bar: Optional[str] = None

        while self.is_running:
            try:
                now_utc = datetime.now(timezone.utc)
                minute = now_utc.minute
                second = now_utc.second

                # Hourly Candle Close Trigger: At minute 00, between 05s and 35s
                is_hourly_cycle = (minute == 0) and (5 <= second <= 35)

                if is_hourly_cycle:
                    # Probe primary symbol bar time
                    probe_df = self.fetch_recent_candles(symbol=self.symbol, limit=5)
                    latest_bar_time = str(probe_df["datetime"].iloc[-1])

                    if latest_bar_time != last_evaluated_bar:
                        logger.info(f"[Hourly] New 1H candle closed: {latest_bar_time}. Running multi-pair scanner...")
                        last_evaluated_bar = latest_bar_time

                        # 1. Check & cancel expired pending orders for all symbols
                        for sym in self.symbols:
                            self.check_pending_order(latest_bar_time, symbol=sym)

                        # 2. Run full multi-pair scan and submit entries
                        self.scan_all_pairs(dry_run=False)

                # High-frequency tick monitor for all active positions
                for sym, pos in list(self.active_positions.items()):
                    if pos is not None:
                        try:
                            time.sleep(0.5)  # Rate limit safety between ticker requests
                            ticker = self.exchange.fetch_ticker(sym)
                            current_price = float(ticker.get("last", ticker.get("close", 0.0)))
                            if current_price > 0:
                                self.monitor_active_position(
                                    current_price=current_price,
                                    current_regime_state=self.last_state_ids.get(sym),
                                    symbol=sym,
                                )
                        except Exception as tick_err:
                            logger.warning(f"[Monitor] Error fetching ticker for {sym}: {tick_err}")

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
    parser.add_argument("--dry-run-once", action="store_true", help="Run a single multi-pair evaluation cycle in dry-run mode and exit")
    parser.add_argument("--symbol", type=str, default=None, help="Primary trading pair symbol (default: BTC/USDT)")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="List of trading pairs to scan (default: BTC/USDT ETH/USDT SOL/USDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Trading timeframe (default: 1h)")
    parser.add_argument("--poll-interval", type=int, default=20, help="Tick polling interval in seconds (default: 20)")
    parser.add_argument("--state-file", type=str, default="data/live_bot_state.json", help="Path to state JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = LiveTradingBot(
        symbols=args.symbols,
        symbol=args.symbol,
        timeframe=args.timeframe,
        state_file=Path(args.state_file),
    )

    if args.dry_run_once:
        logger.info("Executing dry-run multi-pair market evaluation cycle...")
        bot.run_once(dry_run=True)
        sys.exit(0)

    bot.start(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()

