#!/usr/bin/env python3
"""
Lightweight and informative CLI status monitor for Adaptive Trading System.
Reads and parses data/live_bot_state.json, calculates real-time floating PnL via CCXT,
and displays portfolio status in a formatted terminal dashboard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-bootstrap into .venv if current interpreter is running outside of it
_project_root = Path(__file__).resolve().parent.parent
_venv_py = _project_root / ".venv" / "bin" / "python"
if _venv_py.exists() and Path(sys.prefix).resolve() != (_project_root / ".venv").resolve():
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import ccxt

# ANSI Color codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"
BG_BLUE = "\033[44m"


def format_relative_time(iso_str: str) -> str:
    """Returns human-friendly relative time string (e.g. '12s ago', '4m ago')."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff_sec = int((now - dt).total_seconds())
        if diff_sec < 0:
            return "just now"
        if diff_sec < 60:
            return f"{diff_sec}s ago"
        if diff_sec < 3600:
            return f"{diff_sec // 60}m {diff_sec % 60}s ago"
        hours = diff_sec // 3600
        mins = (diff_sec % 3600) // 60
        return f"{hours}h {mins}m ago"
    except Exception:
        return iso_str


def fetch_current_ticker_price(symbol: str) -> Optional[float]:
    """Fetches real-time price from Binance public spot API via CCXT."""
    try:
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 4000})
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("last") or ticker.get("close")
        return float(price) if price else None
    except Exception:
        return None


def render_dashboard(state_path: Path) -> None:
    """Reads state JSON and renders structured terminal status dashboard."""
    if not state_path.exists():
        print(
            f"\n{C_YELLOW}[WARNING] '{state_path}' belum ditemukan.{C_RESET}\n"
            f"Pastikan bot sudah berjalan minimal 1 iterasi (misal: `python3 src/execution/live_bot.py --dry-run-once`).\n"
        )
        return

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state: Dict[str, Any] = json.load(f)
    except Exception as e:
        print(f"\n{C_RED}[ERROR] Gagal membaca state file {state_path}: {e}{C_RESET}\n")
        return

    timestamp_str = state.get("timestamp", "")
    rel_time = format_relative_time(timestamp_str) if timestamp_str else "N/A"
    clean_ts = timestamp_str.split(".")[0].replace("T", " ") + " UTC" if timestamp_str else "N/A"

    symbol = state.get("symbol", "BTC/USDT")
    timeframe = state.get("timeframe", "1h")
    is_testnet = state.get("is_testnet", True)
    testnet_str = f"{C_YELLOW}TESTNET{C_RESET}" if is_testnet else f"{C_GREEN}MAINNET{C_RESET}"

    last_state = state.get("last_state_id")
    state_age = state.get("state_age", 0)
    traded_episode = state.get("traded_in_current_episode", False)

    # Regime Interpretation
    if last_state == 0:
        regime_label = f"{C_GREEN}State 0 (BULLISH MOMENTUM){C_RESET}"
        gate_status = f"{C_GREEN}GATE OPEN / ACTIVE{C_RESET}" if not traded_episode and state_age <= 4 else f"{C_YELLOW}RESTRICTED (Age: {state_age}, Traded: {traded_episode}){C_RESET}"
    elif last_state == 3:
        regime_label = f"{C_RED}State 3 (BEARISH DUMP){C_RESET}"
        gate_status = f"{C_RED}GATE CLOSED (DUMP PROTECTION){C_RESET}"
    elif last_state is not None:
        regime_label = f"{C_YELLOW}State {last_state} (SIDEWAYS / CONSOLIDATION){C_RESET}"
        gate_status = f"{C_YELLOW}GATE CLOSED (WAITING BULLISH){C_RESET}"
    else:
        regime_label = f"{C_DIM}UNKNOWN / INITIALIZING{C_RESET}"
        gate_status = f"{C_DIM}WAITING{C_RESET}"

    active_pos = state.get("active_position")
    pending_order = state.get("pending_order")
    completed_trades = state.get("completed_trades", [])

    # Real-time Ticker Price
    current_price = fetch_current_ticker_price(symbol)

    # Render Header
    print("\n" + f"{C_CYAN}=" * 74 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}               ADAPTIVE SYSTEM - LIVE BOT STATUS MONITOR               {C_RESET}")
    print(f"{C_CYAN}=" * 74 + f"{C_RESET}")
    print(f"  {C_BOLD}Last Heartbeat{C_RESET} : {clean_ts} ({C_CYAN}{rel_time}{C_RESET})")
    print(f"  {C_BOLD}Target Market {C_RESET} : {C_BOLD}{symbol}{C_RESET} [{timeframe}] | Environment: {testnet_str}")
    print(f"  {C_BOLD}Market Regime {C_RESET} : {regime_label} | Age: {state_age} bar(s)")
    print(f"  {C_BOLD}Macro Gate    {C_RESET} : {gate_status}")
    print(f"{C_CYAN}-" * 74 + f"{C_RESET}")

    # Section 1: Active Position
    print(f"  {C_BOLD}{C_WHITE}[ACTIVE POSITION]{C_RESET}")
    if active_pos:
        entry_price = float(active_pos.get("entry_price", 0.0))
        rem_qty = float(active_pos.get("remaining_quantity", 0.0))
        init_qty = float(active_pos.get("initial_quantity", 0.0))
        tp1_hit = bool(active_pos.get("tp1_hit", False))
        tp1_p = float(active_pos.get("tp1_price", 0.0))
        tp2_p = float(active_pos.get("tp2_price", 0.0))
        sl_p = float(active_pos.get("sl_price", 0.0))
        be_p = float(active_pos.get("be_price", 0.0))

        stage_str = f"{C_GREEN}TP1 Hit (50% Scaled, BE Locked){C_RESET}" if tp1_hit else f"{C_YELLOW}Initial (Waiting TP1 50% Scale-Out){C_RESET}"
        print(f"    Status       : {C_GREEN}IN POSITION{C_RESET} | {stage_str}")
        print(f"    Position ID  : {active_pos.get('position_id', 'N/A')}")
        print(f"    Size         : {rem_qty:.6f} {symbol.split('/')[0]} (Initial: {init_qty:.6f})")
        print(f"    Entry Price  : ${entry_price:,.2f}")

        if current_price:
            unrealized_pnl = (current_price - entry_price) * rem_qty
            unrealized_pct = ((current_price - entry_price) / entry_price) * 100.0
            pnl_color = C_GREEN if unrealized_pnl >= 0 else C_RED
            sign = "+" if unrealized_pnl >= 0 else ""
            print(f"    Current Price: ${current_price:,.2f}")
            print(f"    Floating PnL : {pnl_color}{C_BOLD}{sign}${unrealized_pnl:,.2f} ({sign}{unrealized_pct:.2f}%){C_RESET}")
        else:
            print(f"    Current Price: {C_YELLOW}(Ticker API offline / unavailable){C_RESET}")

        # Risk & Targets Brackets
        sl_dist = (sl_p - entry_price) * rem_qty
        tp1_dist = (tp1_p - entry_price) * (init_qty * 0.5)
        tp2_dist = (tp2_p - entry_price) * (init_qty * 0.5)
        print(f"    Stop Loss    : ${sl_p:,.2f} (Risk: {C_RED}${sl_dist:,.2f}{C_RESET}) {'[BE LOCKED]' if tp1_hit else ''}")
        print(f"    Target TP1   : ${tp1_p:,.2f} (Reward: {C_GREEN}+${tp1_dist:,.2f}{C_RESET} on 50% size)")
        print(f"    Target TP2   : ${tp2_p:,.2f} (Reward: {C_GREEN}+${tp2_dist:,.2f}{C_RESET} on 50% size)")
    else:
        print(f"    Status       : {C_DIM}IDLE / WAITING FOR SETUP{C_RESET}")
        if current_price:
            print(f"    Current Price: ${current_price:,.2f}")

    # Section 2: Pending Orders
    if pending_order:
        print(f"\n  {C_BOLD}{C_YELLOW}[PENDING MAKER ORDER]{C_RESET}")
        print(f"    Order ID     : #{pending_order.get('order_id', 'N/A')}")
        print(f"    Type / Side  : LIMIT BUY (Maker Queue)")
        print(f"    Price        : ${float(pending_order.get('price', 0.0)):,.2f}")
        print(f"    Quantity     : {float(pending_order.get('amount', 0.0)):.6f}")
        print(f"    Expires At   : Next Bar ({pending_order.get('expire_after_bar_time', 'Next candle')})")

    print(f"{C_CYAN}-" * 74 + f"{C_RESET}")

    # Section 3: Portfolio & Closed Trades Summary
    total_trades = len(completed_trades)
    total_realized_pnl = sum(float(t.get("gross_pnl", 0.0)) for t in completed_trades)
    win_trades = sum(1 for t in completed_trades if float(t.get("gross_pnl", 0.0)) > 0)
    loss_trades = sum(1 for t in completed_trades if float(t.get("gross_pnl", 0.0)) <= 0)
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    pnl_hist_color = C_GREEN if total_realized_pnl >= 0 else C_RED
    sign_hist = "+" if total_realized_pnl >= 0 else ""

    # Exit reason counts
    exit_counts: Dict[str, int] = {}
    for t in completed_trades:
        reason = t.get("exit_reason", "UNKNOWN")
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    print(f"  {C_BOLD}{C_WHITE}[PORTFOLIO PERFORMANCE]{C_RESET}")
    print(f"    Closed Trades : {total_trades} trades (Win: {win_trades} | Loss: {loss_trades} | Win Rate: {win_rate:.1f}%)")
    print(f"    Realized PnL  : {pnl_hist_color}{C_BOLD}{sign_hist}${total_realized_pnl:,.2f} USDT{C_RESET}")

    if exit_counts:
        reasons_summary = ", ".join(f"{k}: {v}" for k, v in exit_counts.items())
        print(f"    Exit Breakdown: {C_DIM}{reasons_summary}{C_RESET}")

    print(f"{C_CYAN}=" * 74 + f"{C_RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Trading System - Live Bot Status Monitor")
    parser.add_argument(
        "--file",
        type=str,
        default="data/live_bot_state.json",
        help="Path to live_bot_state.json (default: data/live_bot_state.json)",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        help="Auto-refresh interval in seconds (e.g. --watch 5)",
    )
    args = parser.parse_args()
    state_path = Path(args.file)

    if args.watch is not None and args.watch > 0:
        interval = args.watch
        print(f"{C_CYAN}Entering continuous watch mode (Interval: {interval:.1f}s). Press Ctrl+C to stop.{C_RESET}")
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                render_dashboard(state_path)
                print(f"{C_DIM}[Live Watch Mode] Refreshing every {interval:.1f}s... (Ctrl+C to quit){C_RESET}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Exiting live watch mode.{C_RESET}")
            sys.exit(0)
    else:
        render_dashboard(state_path)


if __name__ == "__main__":
    main()
