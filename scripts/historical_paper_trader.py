#!/usr/bin/env python3
"""
Historical Paper Trader (Bar-by-Bar Replay Loop).

Simulates realistic, causal live paper trading against historical continuous OHLCV data
using the 3-Pillar Market Regime Corong:
  - Pilar 1: Causal Gaussian HMM Regime Gate (State 0 Bullish, Single-Shot, state_age <= 4)
  - Pilar 2: Micro Pullback Timing Trigger (Low <= EMA(9) or RSI(14) <= 52.0)
  - Pilar 3: Dynamic Risk Engine (Next-Bar Open BUY, Scaling Out 50:50, ATR TP1/TP2, BE Stop)

Zero Lookahead Bias: At bar index t, the bot only accesses data from bar 0 to t.
Execution: BUY is filled at Open of bar t+1 with slippage and taker fee applied.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-bootstrap into .venv if current interpreter is running outside of it
_venv_dir = Path(__file__).resolve().parent.parent / ".venv"
_venv_py = _venv_dir / "bin" / "python"
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_dir.resolve():
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import talib

from src.config import REGIME_FUNNEL, RegimeFunnelConfig
from src.strategies.regime_funnel import RegimeFunnelStrategy

# Configure minimal fallback logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("historical_paper_trader")


# ==============================================================================
# ANSI Color Formatting Helpers
# ==============================================================================

class Colors:
    """ANSI color codes for terminal visualizer."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    BG_BLUE = "\033[44m"


def colorize(text: str, color_code: str, use_color: bool = True) -> str:
    """Wraps text in ANSI color codes if use_color is True."""
    if not use_color:
        return text
    return f"{color_code}{text}{Colors.RESET}"


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class PaperPosition:
    """Tracks active position lifecycle with scaling-out state."""
    position_id: str
    entry_bar: int
    entry_time: str
    entry_price: float
    quantity: float
    notional_value: float
    entry_fee: float
    take_profit_2: float
    take_profit_1: float
    stop_loss: float
    atr: float
    original_quantity: float
    original_notional: float
    original_entry_fee: float = 0.0
    bars_held: int = 0
    tp1_taken: bool = False
    tp1_bar: Optional[int] = None
    tp1_time: Optional[str] = None
    tp1_exit_price: Optional[float] = None
    tp1_gross: float = 0.0
    tp1_fee: float = 0.0
    is_break_even: bool = False
    realized_gross: float = 0.0
    realized_fee: float = 0.0


@dataclass
class PaperTradeRecord:
    """Stores completed trade details for auditing and ledger persistence."""
    trade_id: str
    entry_bar: int
    entry_time: str
    entry_price: float
    allocated_notional: float
    initial_quantity: float
    entry_fee: float
    tp1_price: float
    tp1_hit: bool
    tp1_bar: Optional[int]
    tp1_time: Optional[str]
    tp1_exit_price: Optional[float]
    tp1_gross_pnl: float
    tp1_fee: float
    tp1_net_pnl: float
    tp2_price: float
    exit_bar: int
    exit_time: str
    exit_price: float
    exit_fee: float
    remaining_quantity_closed: float
    total_fee: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    bars_held: int





# ==============================================================================
# Historical Paper Trading Engine
# ==============================================================================

class HistoricalPaperTrader:
    """
    Executes a Bar-by-Bar Replay Paper Trading simulation adhering strictly
    to the 3-Pilar Corong architecture with zero lookahead bias.
    """

    def __init__(
        self,
        csv_path: Path,
        model_path: Path,
        start_bar: int = 1000,
        total_bars: int = 200,
        delay: float = 0.05,
        initial_capital: float = 100.0,
        trade_allocation: float = 30.0,
        fee_rate: float = 0.0015,         # 0.15% taker fee
        slippage_rate: float = 0.0004,    # 0.04% slippage
        tp1_multiplier: float = 0.80,     # TP1 = Entry + 0.80 * ATR
        tp2_multiplier: float = 1.20,     # TP2 = Entry + 1.20 * ATR
        sl_multiplier: float = 0.70,      # SL  = Entry - 0.70 * ATR
        be_net_buffer: float = 1.0025,    # BE Stop = Entry * 1.0025 (+0.25% net buffer)
        max_holding_bars: int = 24,       # Time expiry threshold
        output_ledger: Path = Path("data/paper_replay_ledger.json"),
        use_color: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.model_path = Path(model_path)
        self.start_bar = start_bar
        self.total_bars = total_bars
        self.delay = delay
        self.initial_capital = initial_capital
        self.trade_allocation = trade_allocation
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.tp1_multiplier = tp1_multiplier
        self.tp2_multiplier = tp2_multiplier
        self.sl_multiplier = sl_multiplier
        self.be_net_buffer = be_net_buffer
        self.max_holding_bars = max_holding_bars
        self.output_ledger = Path(output_ledger)
        self.use_color = use_color and sys.stdout.isatty()

        # Virtual Account State
        self.cash_balance = float(initial_capital)
        self.open_position: Optional[PaperPosition] = None
        self.pending_buy: Optional[Dict[str, Any]] = None
        self.completed_trades: List[PaperTradeRecord] = []
        self.trade_counter = 0

        # Centralized Strategy SSOT
        strategy_config = RegimeFunnelConfig(
            hmm_model_path=str(model_path),
            hmm_bullish_state_id=0,
            state_age_max=4,
            ema_trend_period=200,
            single_shot_per_episode=True,
            pullback_ema_period=9,
            pullback_rsi_period=14,
            pullback_rsi_threshold=52.0,
            risk_sl_mult=sl_multiplier,
            risk_tp1_mult=tp1_multiplier,
            risk_tp2_mult=tp2_multiplier,
            risk_be_buffer=be_net_buffer,
            max_hold_bars=max_holding_bars,
            capital_total=initial_capital,
            trade_allocation=trade_allocation,
            maker_fee=fee_rate,
            slippage=slippage_rate,
        )
        self.strategy = RegimeFunnelStrategy(config=strategy_config)

        # Pilar 1 Regime Tracking State
        self.bullish_state_id = self.strategy.bullish_state_id
        self.bearish_state_id = self.strategy.bearish_state_id
        self.state_age = 0
        self.traded_in_current_episode = False

        # Equity Tracking
        self.peak_equity = float(initial_capital)
        self.max_drawdown_pct = 0.0

    def _resolve_data_file(self) -> Path:
        """Resolves existing continuous CSV or Parquet file path."""
        candidates = [
            self.csv_path,
            Path("dataset/processed/btc/1h/btc_1h_continuous.csv"),
            Path("dataset/processed/btc/btc_1h_continuous.csv"),
            Path("dataset/processed/btc/1h/btc_1h_continuous.parquet"),
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not locate historical candle dataset at '{self.csv_path}' or fallback locations."
        )

    def _load_data_and_model(self) -> Tuple[pd.DataFrame, Any, Any, List[str]]:
        """Loads dataset and computes indicators via Centralized Strategy SSOT."""
        resolved_csv = self._resolve_data_file()
        if resolved_csv.suffix.lower() == ".parquet":
            df_raw = pd.read_parquet(resolved_csv)
        else:
            df_raw = pd.read_csv(resolved_csv)

        # Standardize columns
        df_raw.columns = [c.lower().strip() for c in df_raw.columns]
        if "datetime" not in df_raw.columns and "open_time" in df_raw.columns:
            df_raw["datetime"] = pd.to_datetime(df_raw["open_time"], unit="us" if df_raw["open_time"].iloc[0] > 1e14 else "ms")
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
        df_raw = df_raw.sort_values("datetime").reset_index(drop=True)

        # Compute causal features and indicators via centralized strategy SSOT
        df_feat = self.strategy.compute_indicators(df_raw)
        df_clean = df_feat.dropna(subset=self.strategy.feature_columns + ["ema_9", "rsi_14", "atr_14", "ema_200"]).reset_index(drop=True)

        # Compute online causal HMM states via centralized strategy SSOT
        causal_states, _ = self.strategy.compute_causal_states(df_clean)
        df_clean["regime_state"] = causal_states

        return df_clean, self.strategy.model, self.strategy.scaler, self.strategy.feature_columns

    def run(self) -> Dict[str, Any]:
        """Executes the bar-by-bar causal simulation loop."""
        df, _, _, _ = self._load_data_and_model()

        total_available = len(df)
        if self.start_bar < 50:
            print(colorize(f"[WARN] start_bar ({self.start_bar}) is < 50. Adjusting to 50 for indicator warmup.", Colors.YELLOW, self.use_color))
            self.start_bar = 50

        if self.start_bar >= total_available:
            raise ValueError(f"start_bar ({self.start_bar}) exceeds total available candles ({total_available}).")

        end_bar = min(self.start_bar + self.total_bars, total_available)
        actual_bars = end_bar - self.start_bar

        # Warm up regime state age up to start_bar
        self._warmup_regime_state(df, self.start_bar)

        self._print_header(df, self.start_bar, end_bar, actual_bars)

        # ----------------------------------------------------------------------
        # Bar-by-Bar Replay Loop
        # ----------------------------------------------------------------------
        for t in range(self.start_bar, end_bar):
            bar = df.iloc[t]
            dt_str = str(bar["datetime"])[:19]
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])
            atr = float(bar["atr"])
            ema_9 = float(bar["ema_9"])
            rsi_14 = float(bar["rsi_14"])
            ema_200 = float(bar["ema_200"])
            curr_state = int(bar["regime_state"])

            # ------------------------------------------------------------------
            # Step A: Next-Bar Open Execution (Fill Pending BUY from Bar t-1)
            # ------------------------------------------------------------------
            if self.pending_buy is not None and self.open_position is None:
                self._execute_buy_fill(t, dt_str, o, self.pending_buy["atr"])
                self.pending_buy = None

            # ------------------------------------------------------------------
            # Step B: Pilar 1 Regime Tracking (Episode & Age Update)
            # ------------------------------------------------------------------
            if curr_state == self.bullish_state_id:
                self.state_age += 1
            else:
                self.state_age = 0
                self.traded_in_current_episode = False

            # ------------------------------------------------------------------
            # Step C: Pilar 3 Position Management & Intra-bar Evaluation
            # ------------------------------------------------------------------
            if self.open_position is not None:
                self._process_active_position(t, dt_str, o, h, l, c, curr_state)

            # ------------------------------------------------------------------
            # Step D: Visual Status Update for Current Bar
            # ------------------------------------------------------------------
            self._print_bar_status(t, dt_str, c, curr_state)

            # ------------------------------------------------------------------
            # Step E: Pilar 1 & Pilar 2 Cascading Entry Gate Evaluation
            # ------------------------------------------------------------------
            self._evaluate_entry_signal(t, dt_str, c, l, ema_9, ema_200, rsi_14, atr, curr_state, df)

            # Update Equity & Drawdown
            self._update_equity_metrics(c)

            # Interactive Delay Simulation
            if self.delay > 0:
                time.sleep(self.delay)

        # Close any lingering open position at final candle close
        if self.open_position is not None:
            last_bar = df.iloc[end_bar - 1]
            self._close_position_on_exit(
                end_bar - 1,
                str(last_bar["datetime"])[:19],
                float(last_bar["close"]),
                reason="REPLAY_END",
            )

        # Generate final metrics and persist ledger
        summary = self._generate_and_save_ledger(df, self.start_bar, end_bar)
        self._print_report(summary)
        return summary

    def _warmup_regime_state(self, df: pd.DataFrame, start_bar: int) -> None:
        """Pre-computes state_age and single-shot flags for bars prior to start_bar."""
        self.state_age = 0
        self.traded_in_current_episode = False
        warmup_start = max(0, start_bar - 50)
        for i in range(warmup_start, start_bar):
            st = int(df.iloc[i]["regime_state"])
            if st == self.bullish_state_id:
                self.state_age += 1
            else:
                self.state_age = 0
                self.traded_in_current_episode = False

    def _execute_buy_fill(self, bar_idx: int, dt_str: str, open_price: float, signal_atr: float) -> None:
        """Executes BUY order at bar Open with slippage and taker fee."""
        self.trade_counter += 1
        entry_fill_price = open_price * (1.0 + self.slippage_rate)
        notional = self.trade_allocation
        entry_fee = notional * self.fee_rate
        quantity = notional / entry_fill_price

        # Pilar 3 Boundaries via Centralized Strategy SSOT
        brackets = self.strategy.calculate_risk_brackets(entry_fill_price, signal_atr)
        tp1_price = brackets["tp1_price"]
        tp2_price = brackets["tp2_price"]
        sl_price = brackets["sl_price"]

        self.cash_balance -= (notional + entry_fee)

        self.open_position = PaperPosition(
            position_id=f"PT-{self.trade_counter:04d}",
            entry_bar=bar_idx,
            entry_time=dt_str,
            entry_price=entry_fill_price,
            quantity=quantity,
            notional_value=notional,
            entry_fee=entry_fee,
            take_profit_2=tp2_price,
            take_profit_1=tp1_price,
            stop_loss=sl_price,
            atr=signal_atr,
            original_quantity=quantity,
            original_notional=notional,
            original_entry_fee=entry_fee,
        )

        action_tag = colorize("[ACTION: BUY]", Colors.BG_GREEN + Colors.WHITE + Colors.BOLD, self.use_color)
        fill_info = colorize(f"Filled at ${entry_fill_price:,.2f}", Colors.GREEN + Colors.BOLD, self.use_color)
        sl_info = colorize(f"SL: ${sl_price:,.2f}", Colors.RED, self.use_color)
        tp_info = colorize(f"TP1: ${tp1_price:,.2f} | TP2: ${tp2_price:,.2f}", Colors.CYAN, self.use_color)
        print(f"  {dt_str} {action_tag} {fill_info} | Allocated: ${notional:.2f} | {sl_info} | {tp_info}")

    def _process_active_position(
        self,
        bar_idx: int,
        dt_str: str,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        curr_state: int,
    ) -> None:
        """
        Manages active position through intra-bar price levels:
          1. Worst-case Stop Loss check (Low <= stop_loss)
          2. Scaling-out TP1 (High >= tp1_price) -> Close 50%, lock BE Stop
          3. Final Exit TP2 (High >= tp2_price) -> Close remaining 50%
          4. Bearish breakdown or time expiry
        """
        pos = self.open_position
        if pos is None:
            return

        pos.bars_held += 1

        # Intra-bar worst case ordering: Low checked before High
        hit_sl = low_p <= pos.stop_loss
        hit_regime_breakdown = (curr_state == self.bearish_state_id)
        hit_time_expiry = (pos.bars_held >= self.max_holding_bars)

        if hit_sl:
            exit_price = pos.stop_loss * (1.0 - self.slippage_rate)
            reason = "BREAK_EVEN_SL" if pos.tp1_taken else "STOP_LOSS"
            self._close_position_on_exit(bar_idx, dt_str, exit_price, reason=reason)
            return

        # 1. Target Tahap 1 (TP1 - Scaling out 50%)
        if (not pos.tp1_taken) and (high_p >= pos.take_profit_1):
            tp1_qty = pos.quantity * 0.5
            tp1_notional = pos.notional_value * 0.5
            tp1_entry_fee = pos.entry_fee * 0.5
            tp1_exit_price = pos.take_profit_1 * (1.0 - self.slippage_rate)
            tp1_exit_val = tp1_qty * tp1_exit_price
            tp1_exit_fee = tp1_exit_val * self.fee_rate
            tp1_gross = tp1_exit_val - tp1_notional
            tp1_fee = tp1_entry_fee + tp1_exit_fee

            # Credit realized cash
            self.cash_balance += (tp1_exit_val - tp1_exit_fee)

            pos.realized_gross += tp1_gross
            pos.realized_fee += tp1_fee
            pos.quantity -= tp1_qty
            pos.notional_value -= tp1_notional
            pos.entry_fee -= tp1_entry_fee
            pos.tp1_taken = True
            pos.tp1_bar = bar_idx
            pos.tp1_time = dt_str
            pos.tp1_exit_price = tp1_exit_price
            pos.tp1_gross = tp1_gross
            pos.tp1_fee = tp1_fee
            pos.is_break_even = True

            # Lock Break-Even Stop for remaining 50%: Entry * 1.0025 (+0.25% net buffer)
            be_stop = pos.entry_price * self.be_net_buffer
            pos.stop_loss = max(pos.stop_loss, be_stop)

            tp1_gain_pct = ((tp1_gross - tp1_fee) / tp1_notional) * 100.0
            event_tag = colorize("[EVENT: TP1 HIT]", Colors.BG_BLUE + Colors.WHITE + Colors.BOLD, self.use_color)
            gain_str = colorize(f"+{tp1_gain_pct:.2f}%", Colors.GREEN + Colors.BOLD, self.use_color)
            be_str = colorize(f"${pos.stop_loss:,.2f}", Colors.YELLOW + Colors.BOLD, self.use_color)
            print(f"  {dt_str} {event_tag} Closed 50% at ${tp1_exit_price:,.2f} ({gain_str}) | Locked BE Stop at {be_str}")

        # 2. Target Tahap 2 (TP2 - Sisa 50% Posisi)
        if pos.tp1_taken and (high_p >= pos.take_profit_2):
            exit_price = pos.take_profit_2 * (1.0 - self.slippage_rate)
            self._close_position_on_exit(bar_idx, dt_str, exit_price, reason="TAKE_PROFIT")
        elif hit_regime_breakdown:
            exit_price = close_p * (1.0 - self.slippage_rate)
            self._close_position_on_exit(bar_idx, dt_str, exit_price, reason="REGIME_CHANGE")
        elif hit_time_expiry:
            exit_price = close_p * (1.0 - self.slippage_rate)
            self._close_position_on_exit(bar_idx, dt_str, exit_price, reason="TIME_EXPIRY")

    def _close_position_on_exit(self, bar_idx: int, dt_str: str, exit_price: float, reason: str) -> None:
        """Closes remaining open position and logs trade record."""
        pos = self.open_position
        if pos is None:
            return

        exit_value = pos.quantity * exit_price
        exit_fee = exit_value * self.fee_rate
        total_fee = pos.entry_fee + exit_fee + pos.realized_fee
        gross_pnl = (exit_value - pos.notional_value) + pos.realized_gross
        net_pnl = gross_pnl - total_fee
        return_pct = (net_pnl / pos.original_notional) * 100.0

        self.cash_balance += (exit_value - exit_fee)

        record = PaperTradeRecord(
            trade_id=pos.position_id,
            entry_bar=pos.entry_bar,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            allocated_notional=pos.original_notional,
            initial_quantity=pos.original_quantity,
            entry_fee=pos.original_entry_fee,
            tp1_price=pos.take_profit_1,
            tp1_hit=pos.tp1_taken,
            tp1_bar=pos.tp1_bar,
            tp1_time=pos.tp1_time,
            tp1_exit_price=pos.tp1_exit_price,
            tp1_gross_pnl=pos.tp1_gross,
            tp1_fee=pos.tp1_fee,
            tp1_net_pnl=pos.tp1_gross - pos.tp1_fee,
            tp2_price=pos.take_profit_2,
            exit_bar=bar_idx,
            exit_time=dt_str,
            exit_price=exit_price,
            exit_fee=exit_fee,
            remaining_quantity_closed=pos.quantity,
            total_fee=total_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            exit_reason=reason,
            bars_held=pos.bars_held,
        )
        self.completed_trades.append(record)
        self.open_position = None

        # Format visual exit event
        pnl_color = Colors.GREEN if net_pnl >= 0 else Colors.RED
        pnl_str = colorize(f"{net_pnl:+.2f} USDT ({return_pct:+.2f}%)", pnl_color + Colors.BOLD, self.use_color)
        event_tag = colorize(f"[EVENT: {reason}]", Colors.BG_RED + Colors.WHITE + Colors.BOLD if net_pnl < 0 else Colors.BG_GREEN + Colors.WHITE + Colors.BOLD, self.use_color)
        print(f"  {dt_str} {event_tag} Final Exit at ${exit_price:,.2f} | Net PnL: {pnl_str} | Held: {pos.bars_held} bars")

    def _evaluate_entry_signal(
        self,
        bar_idx: int,
        dt_str: str,
        close_p: float,
        low_p: float,
        ema_9: float,
        ema_200: float,
        rsi_14: float,
        atr: float,
        curr_state: int,
        df: pd.DataFrame,
    ) -> None:
        """Evaluates Cascading Corong gates via Centralized Strategy SSOT."""
        signal_passed, gates = self.strategy.evaluate_gates(
            df=df,
            idx=bar_idx,
            current_state=curr_state,
            state_age=self.state_age,
            traded_in_episode=self.traded_in_current_episode,
        )

        can_trade = (
            signal_passed
            and (self.open_position is None)
            and (self.pending_buy is None)
            and (self.cash_balance >= self.trade_allocation)
        )

        if can_trade:
            self.traded_in_current_episode = True
            self.pending_buy = {
                "signal_bar": bar_idx,
                "signal_time": dt_str,
                "atr": atr,
            }
            sig_tag = colorize("[SIGNAL: BUY]", Colors.YELLOW + Colors.BOLD, self.use_color)
            details = colorize(
                f"State 0 (Age: {self.state_age}) | Close: ${close_p:,.2f} > EMA200: ${ema_200:,.2f} | Low: ${low_p:,.2f} <= EMA9: ${ema_9:,.2f} | RSI: {rsi_14:.1f}",
                Colors.CYAN,
                self.use_color,
            )
            print(f"  {dt_str} {sig_tag} Corong Passed -> {details} -> Queue BUY on Next Open")

    def _print_bar_status(self, bar_idx: int, dt_str: str, close_p: float, curr_state: int) -> None:
        """Formats and prints per-candle status line as requested."""
        state_color = Colors.CYAN if curr_state == self.bullish_state_id else (Colors.RED if curr_state == self.bearish_state_id else Colors.WHITE)
        regime_str = colorize(f"State {curr_state}", state_color + Colors.BOLD, self.use_color)
        price_str = f"${close_p:,.2f}"

        if self.open_position is None:
            status_str = colorize("WAITING", Colors.DIM, self.use_color)
            balance_str = f"${self.cash_balance:,.2f}"
            print(f"[{dt_str}] Bar #{bar_idx:<5d} | Price: {price_str:<11s} | Regime: {regime_str} | Status: {status_str} | Balance: {balance_str}")
        else:
            pos = self.open_position
            curr_val = pos.quantity * close_p
            unrealized = (curr_val - pos.notional_value) + pos.realized_gross
            unrealized_pct = (unrealized / pos.original_notional) * 100.0
            pnl_col = Colors.GREEN if unrealized >= 0 else Colors.RED
            status_desc = colorize(f"HOLDING (Pos: ${pos.notional_value:.2f}, PnL: {unrealized_pct:+.2f}%)", pnl_col + Colors.BOLD, self.use_color)
            total_equity = self.cash_balance + curr_val
            print(f"[{dt_str}] Bar #{bar_idx:<5d} | Price: {price_str:<11s} | Regime: {regime_str} | Status: {status_desc} | Equity: ${total_equity:,.2f}")

    def _update_equity_metrics(self, close_p: float) -> None:
        """Updates real-time equity peak and maximum drawdown percentage."""
        open_val = (self.open_position.quantity * close_p) if self.open_position else 0.0
        current_equity = self.cash_balance + open_val
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            dd = ((self.peak_equity - current_equity) / self.peak_equity) * 100.0
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

    def _print_header(self, df: pd.DataFrame, start: int, end: int, count: int) -> None:
        """Prints stylized terminal simulation header banner."""
        start_time = str(df.iloc[start]["datetime"])[:19]
        end_time = str(df.iloc[end - 1]["datetime"])[:19]
        print("\n" + "=" * 110)
        print(colorize("          HISTORICAL PAPER TRADER: BAR-BY-BAR CAUSAL REPLAYER          ", Colors.BOLD + Colors.CYAN, self.use_color))
        print("=" * 110)
        print(f"  Dataset Path     : {self.csv_path}")
        print(f"  Simulation Range : Bar #{start} -> #{end - 1} ({count} candles)")
        print(f"  Replay Window    : {start_time} to {end_time}")
        print(f"  Initial Capital  : ${self.initial_capital:.2f} USDT | Trade Allocation: ${self.trade_allocation:.2f} USDT")
        print(f"  Execution Specs  : Next-Bar Open | Fee: {self.fee_rate * 100:.2f}% | Slippage: {self.slippage_rate * 100:.2f}%")
        print(f"  Pilar Logic      : HMM Bullish State {self.bullish_state_id} (Age <= 4) & Close > EMA(200) -> Pullback EMA(9)/RSI(14) -> 50:50 Scaling Out")
        print(f"  Ledger Output    : {self.output_ledger}")
        print("-" * 110 + "\n")

    def _generate_and_save_ledger(self, df: pd.DataFrame, start: int, end: int) -> Dict[str, Any]:
        """Calculates performance statistics and persists ledger to JSON."""
        total_trades = len(self.completed_trades)
        wins = [t for t in self.completed_trades if t.net_pnl > 0]
        losses = [t for t in self.completed_trades if t.net_pnl <= 0]
        tp1_hits = [t for t in self.completed_trades if t.tp1_hit]
        tp2_hits = [t for t in self.completed_trades if t.exit_reason == "TAKE_PROFIT"]
        be_sl_hits = [t for t in self.completed_trades if t.exit_reason == "BREAK_EVEN_SL"]
        full_sl_hits = [t for t in self.completed_trades if t.exit_reason == "STOP_LOSS"]

        total_net_pnl = sum(t.net_pnl for t in self.completed_trades)
        total_fees = sum(t.total_fee for t in self.completed_trades)
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        final_balance = self.cash_balance

        summary = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": str(self.csv_path),
                "model": str(self.model_path),
                "start_bar": start,
                "end_bar": end - 1,
                "total_bars_simulated": end - start,
                "start_time": str(df.iloc[start]["datetime"])[:19],
                "end_time": str(df.iloc[end - 1]["datetime"])[:19],
                "initial_capital": self.initial_capital,
                "final_balance": round(final_balance, 4),
                "total_net_pnl": round(total_net_pnl, 4),
                "return_pct": round((total_net_pnl / self.initial_capital) * 100.0, 4),
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "total_trades": total_trades,
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate_pct": round(win_rate, 2),
                "tp1_hit_count": len(tp1_hits),
                "tp2_hit_count": len(tp2_hits),
                "be_sl_hit_count": len(be_sl_hits),
                "full_sl_hit_count": len(full_sl_hits),
                "total_fees_paid": round(total_fees, 4),
                "fee_rate": self.fee_rate,
                "slippage_rate": self.slippage_rate,
                "trade_allocation": self.trade_allocation,
            },
            "trades": [asdict(t) for t in self.completed_trades],
        }

        # Ensure directory exists and write JSON
        self.output_ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_ledger, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _print_report(self, summary: Dict[str, Any]) -> None:
        """Prints a comprehensive summary report to terminal."""
        meta = summary["metadata"]
        ret_color = Colors.GREEN if meta["total_net_pnl"] >= 0 else Colors.RED
        ret_str = colorize(f"{meta['total_net_pnl']:+.2f} USDT ({meta['return_pct']:+.2f}%)", ret_color + Colors.BOLD, self.use_color)

        print("\n" + "=" * 110)
        print(colorize("                    HISTORICAL PAPER TRADER SIMULATION REPORT                   ", Colors.BOLD + Colors.GREEN, self.use_color))
        print("=" * 110)
        print(f"  Simulation Range       : Bar #{meta['start_bar']} -> #{meta['end_bar']} ({meta['total_bars_simulated']} candles)")
        print(f"  Time Period            : {meta['start_time']} to {meta['end_time']}")
        print(f"  Initial Capital        : ${meta['initial_capital']:.2f} USDT")
        print(f"  Final Balance          : ${meta['final_balance']:.2f} USDT")
        print(f"  Total Net PnL          : {ret_str}")
        print(f"  Max Drawdown           : {meta['max_drawdown_pct']:.2f}%")
        print("-" * 110)
        print(f"  Total Executed Trades  : {meta['total_trades']}")
        print(f"  Winning Trades         : {meta['winning_trades']} ({meta['win_rate_pct']:.1f}%)")
        print(f"  Losing Trades          : {meta['losing_trades']}")
        print(f"  TP1 Hits (50% Scaled)  : {meta['tp1_hit_count']}")
        print(f"  TP2 Hits (Full TP)     : {meta['tp2_hit_count']}")
        print(f"  Break-Even SL Hits     : {meta['be_sl_hit_count']}")
        print(f"  Initial Full SL Hits   : {meta['full_sl_hit_count']}")
        print(f"  Total Fees Paid        : ${meta['total_fees_paid']:.4f} USDT")
        print(f"  Ledger Persisted To    : {self.output_ledger}")
        print("=" * 110 + "\n")


# ==============================================================================
# CLI Argument Parsing & Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    default_csv = Path("dataset/processed/btc/1h/btc_1h_continuous.csv")
    if not default_csv.exists():
        default_csv = Path("dataset/processed/btc/btc_1h_continuous.csv")

    parser = argparse.ArgumentParser(
        description="Historical Paper Trader (Bar-by-Bar Replay Loop with Corong 3-Pilar)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, default=str(default_csv), help="Path to continuous OHLCV dataset")
    parser.add_argument("--start", type=int, default=1000, help="Starting candle index for simulation replay")
    parser.add_argument("--bars", type=int, default=200, help="Number of candles to replay")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay in seconds per bar for live terminal animation")
    parser.add_argument("--capital", type=float, default=100.0, help="Initial virtual capital in USDT")
    parser.add_argument("--allocation", type=float, default=30.0, help="USDT allocated per trade")
    parser.add_argument("--fee", type=float, default=0.0015, help="Taker fee rate (0.0015 = 0.15%%)")
    parser.add_argument("--slippage", type=float, default=0.0004, help="Slippage rate (0.0004 = 0.04%%)")
    parser.add_argument("--model", type=str, default="models/btc_1h_regime_hmm.joblib", help="Path to trained Gaussian HMM model")
    parser.add_argument("--output-ledger", type=str, default="data/paper_replay_ledger.json", help="Path to save output trade ledger JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes in terminal")
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    trader = HistoricalPaperTrader(
        csv_path=Path(args.csv),
        model_path=Path(args.model),
        start_bar=args.start,
        total_bars=args.bars,
        delay=args.delay,
        initial_capital=args.capital,
        trade_allocation=args.allocation,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        output_ledger=Path(args.output_ledger),
        use_color=not args.no_color,
    )
    trader.run()


if __name__ == "__main__":
    main()
