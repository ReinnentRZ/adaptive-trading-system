"""
Adaptive Trading System - Real-time Multi-Pair Web Dashboard.

A decoupled, read-only Streamlit dashboard to monitor:
  - 3-Slot Portfolio Quota & Active Positions (BTC/USDT, ETH/USDT, SOL/USDT)
  - Dynamic Regime-Adaptive Brackets (SL, BE Lock, TP1, TP2)
  - Layer-2 AI Meta-Labeling Probabilities & HMM Market Regimes
  - Historical Trade Ledger, Cumulative Equity Curve, and Key Quant KPIs

Designed to run independently and non-blocking alongside the trading bot container.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import ccxt
import pandas as pd
import plotly.express as px
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Adaptive Quant Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark trading terminal aesthetics
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .badge-active {
        background-color: #238636;
        color: #ffffff;
    }
    .badge-pending {
        background-color: #d29922;
        color: #ffffff;
    }
    .badge-flat {
        background-color: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .badge-pass {
        color: #3fb950;
        font-weight: bold;
    }
    .badge-fail {
        color: #f85149;
        font-weight: bold;
    }
    .price-val {
        font-family: 'SF Mono', Monaco, Consolas, monospace;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_state_file_path() -> Path:
    """Resolves state file path from environment or default project location."""
    raw_path = os.getenv("STATE_FILE", "data/live_bot_state.json")
    path_obj = Path(raw_path)
    if not path_obj.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / raw_path
        if candidate.exists() or not path_obj.exists():
            return candidate
    return path_obj


def load_state() -> Dict[str, Any]:
    """
    Loads live bot state from JSON in a read-only, exception-safe manner.
    Returns standard fallback schema if file is missing, empty, or currently being written.
    """
    state_file = get_state_file_path()
    fallback = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "is_testnet": True,
        "max_concurrent_positions": 3,
        "allocation_per_trade_usd": 100.0,
        "active_positions": {
            "BTC/USDT": None,
            "ETH/USDT": None,
            "SOL/USDT": None,
            "BNB/USDT": None,
        },
        "pending_orders": {
            "BTC/USDT": None,
            "ETH/USDT": None,
            "SOL/USDT": None,
            "BNB/USDT": None,
        },
        "last_state_ids": {"BTC/USDT": None, "ETH/USDT": None, "SOL/USDT": None, "BNB/USDT": None},
        "state_ages": {"BTC/USDT": 0, "ETH/USDT": 0, "SOL/USDT": 0, "BNB/USDT": 0},
        "meta_probabilities": {"BTC/USDT": 0.0, "ETH/USDT": 0.0, "SOL/USDT": 0.0, "BNB/USDT": 0.0},
        "trade_history": [],
        "completed_trades": [],
    }

    if not state_file.exists():
        return fallback

    for _ in range(3):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return fallback
                data = json.loads(content)
                if isinstance(data, dict):
                    # Harmonize history fields
                    if "trade_history" not in data and "completed_trades" in data:
                        data["trade_history"] = data["completed_trades"]
                    elif "completed_trades" not in data and "trade_history" in data:
                        data["completed_trades"] = data["trade_history"]
                    return data
        except Exception:
            time.sleep(0.05)

    return fallback


@st.cache_data(ttl=5)
def fetch_public_tickers(symbols: Tuple[str, ...]) -> Dict[str, float]:
    """Fetches real-time market prices from Binance public API (cached for 5 seconds)."""
    prices: Dict[str, float] = {}
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        for sym in symbols:
            try:
                t = ex.fetch_ticker(sym)
                p = float(t.get("last", t.get("close", 0.0)))
                if p > 0:
                    prices[sym] = p
            except Exception:
                continue
    except Exception:
        pass
    return prices


# ==============================================================================
# SIDEBAR CONTROLS & SYSTEM CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.title("⚡ Control Panel")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh Dashboard", value=True)
    refresh_rate = st.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5, step=1)

    if st.button("🔄 Refresh Data Now", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(
        """
        - **Engine**: 3-Pilar Regime Corong
        - **Macro**: Gaussian HMM + Kalman
        - **Micro**: Dynamic Pullback
        - **Decider**: Layer-2 LightGBM AI
        - **Risk**: Dynamic Regime Brackets
        - **Sizing**: $100 / slot (Max 3)
        """
    )

    state_path = get_state_file_path()
    st.caption(f"**State Ledger**: `{state_path.name}`")
    st.caption(f"**Ledger Path**: `{state_path}`")


# Load current bot state
state_data = load_state()
configured_symbols = state_data.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
active_positions = state_data.get("active_positions", {})
pending_orders = state_data.get("pending_orders", {})
last_state_ids = state_data.get("last_state_ids", {})
state_ages = state_data.get("state_ages", {})
trade_history = state_data.get("trade_history", state_data.get("completed_trades", []))
max_slots = int(state_data.get("max_concurrent_positions", 3))
trade_alloc = float(state_data.get("allocation_per_trade_usd", 100.0))
is_testnet = bool(state_data.get("is_testnet", True))
timeframe = state_data.get("timeframe", "1h")
last_update = state_data.get("timestamp", "-")

# Fetch live tickers for unrealized PnL calculations
live_prices = fetch_public_tickers(tuple(configured_symbols))

# Calculate high-level KPIs
active_count = sum(1 for p in active_positions.values() if p is not None)
pending_count = sum(1 for o in pending_orders.values() if o is not None)
used_slots = active_count + pending_count

total_realized_pnl = sum(float(t.get("gross_pnl", 0.0)) for t in trade_history)
total_trades = len(trade_history)
winning_trades = sum(1 for t in trade_history if float(t.get("gross_pnl", 0.0)) > 0)
win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0


# ==============================================================================
# TOP HEADER & KPI METRICS BAR
# ==============================================================================
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("⚡ Adaptive Trading System")
    st.caption(
        f"Multi-Pair Regime Scanner (`BTC/USDT`, `ETH/USDT`, `SOL/USDT` [{timeframe}]) | "
        f"Layer-2 AI Decider & Regime-Adaptive Brackets"
    )

with col_status:
    env_label = "TESTNET (SANDBOX)" if is_testnet else "MAINNET LIVE"
    badge_style = "background-color: #1f6feb;" if is_testnet else "background-color: #238636;"
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="{badge_style} color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.85rem;">
                ● {env_label}
            </span>
            <div style="font-size: 0.75rem; color: #8b949e; margin-top: 6px;">Last Sync: {last_update[:19]} UTC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# 4 Key Metric KPIs
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric(
        label="Portfolio Exposure",
        value=f"{used_slots} / {max_slots} Slots",
        delta=f"{max_slots - used_slots} Slots Free",
        delta_color="normal",
    )
with m2:
    pnl_color = "normal" if total_realized_pnl >= 0 else "inverse"
    pnl_delta = (
        f"{(total_realized_pnl / 387.50 * 100.0):+,.2f}% (Cap $387.50)"
        if total_realized_pnl != 0
        else "0.00%"
    )
    st.metric(
        label="Total Realized PnL",
        value=f"${total_realized_pnl:+,.2f} USDT",
        delta=pnl_delta,
        delta_color=pnl_color,
    )
with m3:
    st.metric(
        label="Win Rate",
        value=f"{win_rate:.1f}%",
        delta=f"{winning_trades}W - {total_trades - winning_trades}L",
        delta_color="normal",
    )
with m4:
    st.metric(
        label="Completed Trades",
        value=f"{total_trades}",
        delta=f"${trade_alloc:,.0f} / Trade",
        delta_color="off",
    )
with m5:
    active_notional = active_count * trade_alloc
    st.metric(
        label="Active Capital Deployed",
        value=f"${active_notional:,.2f}",
        delta=f"{(active_notional / 387.50 * 100.0):.1f}% Total Capital",
        delta_color="off",
    )

st.markdown("---")


# ==============================================================================
# SECTION 1: 3-SLOT MULTI-PAIR SCANNER CARDS
# ==============================================================================
st.subheader("🎯 3-Slot Portfolio Allocation & Active Positions")

slot_cols = st.columns(len(configured_symbols))

for idx, sym in enumerate(configured_symbols):
    with slot_cols[idx]:
        pos = active_positions.get(sym)
        pending = pending_orders.get(sym)
        live_p = live_prices.get(sym, 0.0)
        st_id = last_state_ids.get(sym)
        st_age = state_ages.get(sym, 0)

        # Container Card
        with st.container(border=True):
            # Card Header
            h_left, h_right = st.columns([2, 1])
            with h_left:
                st.markdown(f"### {sym}")
            with h_right:
                if pos:
                    st.markdown("<span class='status-badge badge-active'>ACTIVE LONG</span>", unsafe_allow_html=True)
                elif pending:
                    st.markdown("<span class='status-badge badge-pending'>LIMIT BUY</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='status-badge badge-flat'>FLAT / SCAN</span>", unsafe_allow_html=True)

            # Live price display
            if live_p > 0:
                st.markdown(f"**Market Price**: <span class='price-val'>${live_p:,.2f}</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Market Price**: <span class='price-val'>Synchronizing...</span>", unsafe_allow_html=True)

            st.divider()

            # Scenario A: Position is ACTIVE
            if pos:
                entry_p = float(pos.get("entry_price", 0.0))
                qty = float(pos.get("remaining_quantity", pos.get("initial_quantity", 0.0)))
                tp1_p = float(pos.get("tp1_price", 0.0))
                tp2_p = float(pos.get("tp2_price", 0.0))
                sl_p = float(pos.get("sl_price", 0.0))
                tp1_hit = bool(pos.get("tp1_hit", False))

                unrealized_pnl = (live_p - entry_p) * qty if (live_p > 0 and entry_p > 0) else 0.0
                unrealized_pct = ((live_p - entry_p) / entry_p * 100.0) if (live_p > 0 and entry_p > 0) else 0.0

                pnl_color_style = "color: #3fb950;" if unrealized_pnl >= 0 else "color: #f85149;"

                st.markdown(
                    f"**Unrealized PnL**: <span class='price-val' style='{pnl_color_style} font-size: 1.15rem;'>"
                    f"${unrealized_pnl:+,.2f} ({unrealized_pct:+.2f}%)</span>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"• **Entry Price**: `${entry_p:,.2f}` | **Qty**: `{qty:.5f}`")

                # Brackets detail
                st.markdown("##### Dynamic Brackets:")
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    st.markdown(f"🔴 **Stop Loss**: `${sl_p:,.2f}`")
                    st.markdown(f"🟡 **BE Lock**: `{'LOCKED' if tp1_hit else 'ARMED'}`")
                with b_c2:
                    tp1_status = "✅ FILLED (50%)" if tp1_hit else f"`${tp1_p:,.2f}`"
                    st.markdown(f"🟢 **TP1 Target**: {tp1_status}")
                    st.markdown(f"🚀 **TP2 Target**: `${tp2_p:,.2f}`")

                # Mini Gauge for Price relative to SL and TP2
                if live_p > 0 and entry_p > 0 and tp2_p > sl_p:
                    norm_val = max(0.0, min(1.0, (live_p - sl_p) / (tp2_p - sl_p)))
                    st.progress(norm_val, text=f"Range: SL (${sl_p:,.0f}) ⟵ Current ⟶ TP2 (${tp2_p:,.0f})")

            # Scenario B: Order is PENDING
            elif pending:
                limit_p = float(pending.get("price", 0.0))
                amt = float(pending.get("amount", 0.0))
                sig_atr = float(pending.get("signal_atr", 0.0))
                created_t = pending.get("created_at_time", "-")

                info_msg = (
                    f"⏳ **Limit Buy In Queue**\n"
                    f"- **Target Price**: `${limit_p:,.2f}`\n"
                    f"- **Size**: `{amt:.5f}`\n"
                    f"- **Signal ATR**: `${sig_atr:,.2f}`"
                )
                st.info(info_msg)
                st.caption(f"Submitted at: {created_t[:19]}")

            # Scenario C: FLAT (Waiting / Scanning)
            else:
                st.markdown("##### Regime Status:")
                state_desc = f"State {st_id}" if st_id is not None else "Analyzing"
                st.markdown(f"• **HMM State**: `{state_desc}` (Age: `{st_age}` bars)")
                st.markdown("• **Gate Condition**: Waiting for Bullish Causal Setup")
                st.markdown("• **Pilar 3 Allocation**: Available ($100.00 ready)")



# ==============================================================================
# SECTION 1.5: LAYER-2 AI META-LABELER RADAR
# ==============================================================================
st.markdown("---")
st.subheader("🤖 Layer-2 AI Meta-Labeler Radar ($P(\\text{Win})$ Probability)")

meta_probs = state_data.get("meta_probabilities", {})
radar_cols = st.columns(len(configured_symbols))

for idx, sym in enumerate(configured_symbols):
    with radar_cols[idx]:
        with st.container(border=True):
            st.markdown(f"#### {sym}")
            prob = float(meta_probs.get(sym, 0.0))
            is_approved = prob >= 0.50
            status_text = "APPROVED" if is_approved else "FILTERED"
            badge_class = "badge-pass" if is_approved else "badge-fail"

            st.markdown(
                f"<div style='margin-bottom: 8px;'>"
                f"Prob: <span class='price-val' style='font-size: 1.1rem;'>{prob * 100:.1f}%</span> | "
                f"<span class='{badge_class}'>● {status_text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Visual progress gauge with threshold marker at 50%
            st.progress(min(1.0, max(0.0, prob)), text=f"P(Win): {prob:.2f} (Threshold: 0.50)")

            st_id = last_state_ids.get(sym)
            target_st = 3 if "BNB" in sym else 0
            is_bull = (st_id == target_st) if st_id is not None else False
            st_text = f"State {st_id}" if st_id is not None else "Scanning"
            regime_label = "Bullish Target" if is_bull else "Neutral / Bear"
            st.caption(f"HMM: `{st_text}` ({regime_label})")


# ==============================================================================
# SECTION 2: CHARTS & ANALYTICS
# ==============================================================================
st.markdown("---")
st.subheader("📈 Performance Analytics & Trade Evolution")

chart_tabs = st.tabs(["Cumulative Equity Curve", "Exit Reasons Breakdown", "PnL Distribution"])

with chart_tabs[0]:
    if trade_history:
        # Build cumulative dataframe
        df_trades = pd.DataFrame(trade_history)
        df_trades["gross_pnl"] = pd.to_numeric(df_trades["gross_pnl"], errors="coerce").fillna(0.0)
        df_trades["cum_pnl"] = df_trades["gross_pnl"].cumsum()
        df_trades["equity"] = 387.50 + df_trades["cum_pnl"]
        df_trades["trade_num"] = range(1, len(df_trades) + 1)

        fig_equity = px.area(
            df_trades,
            x="trade_num",
            y="cum_pnl",
            title="Cumulative Net PnL Progression (USDT)",
            labels={"trade_num": "Trade #", "cum_pnl": "Net PnL (USDT)"},
            color_discrete_sequence=["#238636" if total_realized_pnl >= 0 else "#f85149"],
        )
        fig_equity.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("No trade history available yet. Equity curve will populate as positions complete.")

with chart_tabs[1]:
    if trade_history:
        df_reasons = pd.DataFrame(trade_history)
        reasons_count = df_reasons["exit_reason"].value_counts().reset_index()
        reasons_count.columns = ["Exit Reason", "Count"]

        fig_reasons = px.pie(
            reasons_count,
            values="Count",
            names="Exit Reason",
            title="Exit Reason Distribution",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_reasons.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_reasons, use_container_width=True)
    else:
        st.info("No exit data available yet.")

with chart_tabs[2]:
    if trade_history:
        df_dist = pd.DataFrame(trade_history)
        df_dist["gross_pnl"] = pd.to_numeric(df_dist["gross_pnl"], errors="coerce").fillna(0.0)
        df_dist["pnl_sign"] = df_dist["gross_pnl"].apply(lambda x: "Win (Profit)" if x > 0 else "Loss")

        fig_dist = px.histogram(
            df_dist,
            x="gross_pnl",
            color="pnl_sign",
            title="PnL Distribution per Trade (USDT)",
            labels={"gross_pnl": "PnL ($ USDT)", "count": "Trades"},
            color_discrete_map={"Win (Profit)": "#238636", "Loss": "#da3633"},
            nbins=20,
        )
        fig_dist.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No distribution data available yet.")


# ==============================================================================
# SECTION 3: HISTORICAL TRADE LEDGER
# ==============================================================================
st.markdown("---")
st.subheader("📋 Completed Trades Ledger")

if trade_history:
    df_ledger = pd.DataFrame(trade_history)
    cols_to_show = [
        "position_id",
        "symbol",
        "entry_price",
        "exit_price",
        "quantity",
        "gross_pnl",
        "exit_reason",
        "entry_time",
        "exit_time",
    ]
    existing_cols = [c for c in cols_to_show if c in df_ledger.columns]
    df_display = df_ledger[existing_cols].copy()

    # Formatting numeric columns
    if "entry_price" in df_display:
        df_display["entry_price"] = df_display["entry_price"].map(lambda x: f"${float(x):,.2f}")
    if "exit_price" in df_display:
        df_display["exit_price"] = df_display["exit_price"].map(lambda x: f"${float(x):,.2f}")
    if "quantity" in df_display:
        df_display["quantity"] = df_display["quantity"].map(lambda x: f"{float(x):.6f}")
    if "gross_pnl" in df_display:
        df_display["gross_pnl"] = df_display["gross_pnl"].map(lambda x: f"${float(x):+,.2f}")

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Ledger is currently empty. Closed trades will automatically be logged here.")


# ==============================================================================
# AUTO-REFRESH TRIGGER
# ==============================================================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
