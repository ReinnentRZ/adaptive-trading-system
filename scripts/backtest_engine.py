#!/usr/bin/env python3
"""
Event-Driven Backtesting Engine for Spot Trading (Long-Only) Meta-Labeling System.

Simulates order execution, fee deduction, slippage, and dynamic ATR-based TP/SL
risk management on Out-of-Sample (OOS) Test datasets. Generates trade logs and
comprehensive daily performance metrics.

Author: Adaptive Trading System Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_features import (
    FeatureExtractor,
    _compute_lorentzian_signals_parallel,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BacktestEngine")

FEATURE_COLUMNS: List[str] = [
    "rsi",
    "cci",
    "adx",
    "wt_diff",
    "atr",
    "volatility_ratio",
    "normalized_atr",
    "lorentzian_signal",
]


@dataclass
class Position:
    """Represents an active Spot Long position."""
    position_id: str
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    notional_value: float
    entry_fee: float
    take_profit: float
    stop_loss: float
    atr: float
    ai_prob: float
    max_holding_bars: int = 12
    bars_held: int = 0


@dataclass
class TradeRecord:
    """Represents a completed trade."""
    trade_id: str
    coin: str
    timeframe: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    notional_value: float
    exit_value: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    entry_fee: float
    exit_fee: float
    total_fee: float
    exit_reason: str  # "TAKE_PROFIT", "STOP_LOSS", "TIME_EXPIRY", "BACKTEST_END"
    bars_held: int
    ai_prob: float


@dataclass
class DailySummary:
    """Daily trading performance summary."""
    date: str
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    ending_balance: float
    daily_return_pct: float


@dataclass
class BacktestResult:
    """Overall backtest performance metrics and statistics."""
    coin: str
    timeframe: str
    threshold: float
    initial_capital: float
    final_balance: float
    net_profit_usdt: float
    net_profit_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_fees_paid: float
    total_trading_days: int
    avg_trades_per_day: float
    avg_daily_pnl_usdt: float
    avg_daily_return_pct: float
    best_day_pnl: float
    worst_day_pnl: float
    trades_log_path: Path
    daily_log_path: Path


class EventDrivenBacktester:
    """
    Event-driven spot trading simulation engine for Out-of-Sample evaluation.
    """

    def __init__(
        self,
        coin: str,
        timeframe: str,
        initial_capital: float = 100.0,
        trade_allocation: float = 5.0,
        max_open_positions: int = 2,
        fee_rate: float = 0.00075,       # 0.075% Binance BNB tier
        slippage_rate: float = 0.0002,   # 0.02% execution slippage
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 12,
        threshold: Optional[float] = 0.50,
        auto_threshold: bool = False,
        processed_dir: Path = Path("dataset/processed"),
        models_dir: Path = Path("models"),
        logs_dir: Path = Path("logs"),
        test_ratio: float = 0.15,
        purge_buffer: int = 12,
    ) -> None:
        self.coin = coin.lower()
        self.timeframe = timeframe.lower()
        self.initial_capital = initial_capital
        self.trade_allocation = trade_allocation
        self.max_open_positions = max_open_positions
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.threshold = threshold if threshold is not None else 0.50
        self.auto_threshold = auto_threshold
        self.processed_dir = Path(processed_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.logs_dir = Path(logs_dir).resolve()
        self.test_ratio = test_ratio
        self.purge_buffer = purge_buffer

        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_oos_test_data(self) -> Tuple[pd.DataFrame, Any, float]:
        """
        Loads full continuous candle dataset, generates features and primary buy triggers,
        and isolates the Out-of-Sample test set partition.
        """
        data_path = (
            self.processed_dir
            / self.coin
            / self.timeframe
            / f"{self.coin}_{self.timeframe}_continuous.parquet"
        )
        model_path = (
            self.models_dir / f"{self.coin}_{self.timeframe}_dedication_lgbm.joblib"
        )
        metadata_path = (
            self.models_dir / f"{self.coin}_{self.timeframe}_metadata.json"
        )

        if not data_path.exists():
            raise FileNotFoundError(f"Continuous dataset not found: {data_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")

        df = pd.read_parquet(data_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df.sort_values(by="datetime", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Feature Extraction
        df = FeatureExtractor.extract_indicators(df)

        # Compute Lorentzian classification signals
        lorentz_features = np.column_stack((
            df["rsi"].to_numpy(dtype=np.float64),
            df["adx"].to_numpy(dtype=np.float64),
            df["cci"].to_numpy(dtype=np.float64),
            df["wt1"].to_numpy(dtype=np.float64),
        ))
        close_np = df["close"].to_numpy(dtype=np.float64)
        lorentzian_signals = _compute_lorentzian_signals_parallel(lorentz_features, close_np)
        df["lorentzian_signal"] = lorentzian_signals

        # Primary Buy Signal: (lorentzian_signal == 1) | ((rsi < 40) & (wt_diff > 0))
        primary_buy = (df["lorentzian_signal"] == 1.0) | (
            (df["rsi"] < 40.0) & (df["wt_diff"] > 0.0)
        )
        df["primary_signal"] = primary_buy

        # Drop warmup NaNs
        df.dropna(subset=FEATURE_COLUMNS, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Temporal Partitioning (OOS Test Set 15%)
        n = len(df)
        train_end = int(n * 0.70)
        val_start = train_end + self.purge_buffer
        val_end = val_start + int(n * 0.15)
        test_start = val_end + self.purge_buffer

        df_test = df.iloc[test_start:].copy().reset_index(drop=True)
        model = joblib.load(model_path)

        # Resolve threshold
        effective_threshold = self.threshold
        if self.auto_threshold and metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    effective_threshold = float(meta.get("recommended_threshold", self.threshold))
                    logger.info(
                        f"[{self.coin.upper()}] [{self.timeframe}] Using calibrated auto-threshold: {effective_threshold:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Could not load recommended threshold from metadata: {e}")

        logger.info(
            f"[{self.coin.upper()}] [{self.timeframe}] Loaded OOS Test Set: {len(df_test):,} candles "
            f"({df_test['datetime'].iloc[0]} to {df_test['datetime'].iloc[-1]}) | Threshold: {effective_threshold:.2f}"
        )

        return df_test, model, effective_threshold

    def run(self) -> BacktestResult:
        """Executes event-driven backtest simulation across OOS test candles."""
        df_test, model, effective_threshold = self.load_oos_test_data()

        # Compute AI Meta-Model Inference Probabilities across all test candles
        X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        raw_probs = model.predict_proba(X_test)
        ai_probs = raw_probs[:, 1] if raw_probs.ndim == 2 and raw_probs.shape[1] > 1 else raw_probs.ravel()
        df_test["ai_prob"] = ai_probs

        # Portfolio State
        cash_balance = self.initial_capital
        open_positions: List[Position] = []
        completed_trades: List[TradeRecord] = []
        equity_curve: List[Tuple[pd.Timestamp, float]] = []

        total_candles = len(df_test)
        trade_counter = 0

        for i in range(total_candles):
            current_bar = df_test.iloc[i]
            dt = current_bar["datetime"]
            o = float(current_bar["open"])
            h = float(current_bar["high"])
            l = float(current_bar["low"])
            c = float(current_bar["close"])
            atr = float(current_bar["atr"])
            primary_signal = bool(current_bar["primary_signal"])
            prob = float(current_bar["ai_prob"])

            # ------------------------------------------------------------------
            # 1. Check Exit Conditions for Active Positions
            # ------------------------------------------------------------------
            remaining_positions: List[Position] = []

            for pos in open_positions:
                pos.bars_held += 1
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                hit_tp = h >= pos.take_profit
                hit_sl = l <= pos.stop_loss

                # Realistic Intra-bar Order of Execution
                if hit_tp and hit_sl:
                    if o >= pos.take_profit:
                        exit_price = pos.take_profit * (1.0 - self.slippage_rate)
                        exit_reason = "TAKE_PROFIT"
                        exit_triggered = True
                    elif o <= pos.stop_loss:
                        exit_price = pos.stop_loss * (1.0 - self.slippage_rate)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    elif c >= o:
                        # Bullish candle: typically Open -> Low -> High -> Close
                        # If low <= stop loss, SL was hit during the initial dip before the rally
                        exit_price = pos.stop_loss * (1.0 - self.slippage_rate)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    else:
                        # Bearish candle: typically Open -> High -> Low -> Close
                        # If high >= take profit, TP was hit during initial rally before the drop
                        exit_price = pos.take_profit * (1.0 - self.slippage_rate)
                        exit_reason = "TAKE_PROFIT"
                        exit_triggered = True
                elif hit_tp:
                    exit_price = pos.take_profit * (1.0 - self.slippage_rate)
                    exit_reason = "TAKE_PROFIT"
                    exit_triggered = True
                elif hit_sl:
                    exit_price = pos.stop_loss * (1.0 - self.slippage_rate)
                    exit_reason = "STOP_LOSS"
                    exit_triggered = True
                elif pos.bars_held >= pos.max_holding_bars:
                    # Time barrier expired -> Market exit on current close
                    exit_price = c * (1.0 - self.slippage_rate)
                    exit_reason = "TIME_EXPIRY"
                    exit_triggered = True

                if exit_triggered:
                    exit_value = pos.quantity * exit_price
                    exit_fee = exit_value * self.fee_rate
                    total_fee = pos.entry_fee + exit_fee
                    gross_pnl = exit_value - pos.notional_value
                    net_pnl = gross_pnl - total_fee
                    return_pct = (net_pnl / pos.notional_value) * 100.0

                    cash_balance += exit_value - exit_fee

                    trade_rec = TradeRecord(
                        trade_id=pos.position_id,
                        coin=self.coin.upper(),
                        timeframe=self.timeframe,
                        entry_time=str(pos.entry_time),
                        exit_time=str(dt),
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        notional_value=pos.notional_value,
                        exit_value=exit_value,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        return_pct=return_pct,
                        entry_fee=pos.entry_fee,
                        exit_fee=exit_fee,
                        total_fee=total_fee,
                        exit_reason=exit_reason,
                        bars_held=pos.bars_held,
                        ai_prob=pos.ai_prob,
                    )
                    completed_trades.append(trade_rec)
                else:
                    remaining_positions.append(pos)

            open_positions = remaining_positions

            # ------------------------------------------------------------------
            # 2. Check Entry Conditions for New Buy Position
            # ------------------------------------------------------------------
            signal_buy = primary_signal and (prob >= effective_threshold)
            can_allocate = (
                len(open_positions) < self.max_open_positions
                and cash_balance >= self.trade_allocation
            )

            if signal_buy and can_allocate:
                trade_counter += 1
                entry_fill_price = c * (1.0 + self.slippage_rate)
                notional = self.trade_allocation
                entry_fee = notional * self.fee_rate
                quantity = (notional - entry_fee) / entry_fill_price

                tp_price = entry_fill_price + (self.tp_multiplier * atr)
                sl_price = entry_fill_price - (self.sl_multiplier * atr)

                cash_balance -= notional

                new_pos = Position(
                    position_id=f"T-{trade_counter:04d}",
                    entry_index=i,
                    entry_time=dt,
                    entry_price=entry_fill_price,
                    quantity=quantity,
                    notional_value=notional,
                    entry_fee=entry_fee,
                    take_profit=tp_price,
                    stop_loss=sl_price,
                    atr=atr,
                    ai_prob=prob,
                    max_holding_bars=self.max_holding_bars,
                )
                open_positions.append(new_pos)

            # Mark total equity at bar close
            unrealized_value = sum(p.quantity * c for p in open_positions)
            total_equity = cash_balance + unrealized_value
            equity_curve.append((dt, total_equity))

        # Close any remaining open positions at the end of the test set
        if open_positions:
            last_dt = df_test["datetime"].iloc[-1]
            last_close = float(df_test["close"].iloc[-1])
            for pos in open_positions:
                exit_price = last_close * (1.0 - self.slippage_rate)
                exit_value = pos.quantity * exit_price
                exit_fee = exit_value * self.fee_rate
                total_fee = pos.entry_fee + exit_fee
                gross_pnl = exit_value - pos.notional_value
                net_pnl = gross_pnl - total_fee
                return_pct = (net_pnl / pos.notional_value) * 100.0
                cash_balance += exit_value - exit_fee

                trade_rec = TradeRecord(
                    trade_id=pos.position_id,
                    coin=self.coin.upper(),
                    timeframe=self.timeframe,
                    entry_time=str(pos.entry_time),
                    exit_time=str(last_dt),
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    notional_value=pos.notional_value,
                    exit_value=exit_value,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    return_pct=return_pct,
                    entry_fee=pos.entry_fee,
                    exit_fee=exit_fee,
                    total_fee=total_fee,
                    exit_reason="BACKTEST_END",
                    bars_held=pos.bars_held,
                    ai_prob=pos.ai_prob,
                )
                completed_trades.append(trade_rec)
            open_positions = []

        final_balance = cash_balance

        # ----------------------------------------------------------------------
        # 3. Compute Comprehensive Performance Metrics
        # ----------------------------------------------------------------------
        net_profit_usdt = final_balance - self.initial_capital
        net_profit_pct = (net_profit_usdt / self.initial_capital) * 100.0

        total_trades = len(completed_trades)
        wins = [t for t in completed_trades if t.net_pnl > 0]
        losses = [t for t in completed_trades if t.net_pnl <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.gross_pnl for t in wins)
        gross_loss = abs(sum(t.gross_pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
        total_fees = sum(t.total_fee for t in completed_trades)

        # Max Drawdown Calculation
        equities = np.array([eq[1] for eq in equity_curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        max_drawdown_pct = abs(float(np.min(drawdowns))) * 100.0 if len(drawdowns) > 0 else 0.0

        # Daily Aggregation
        df_trades = pd.DataFrame([asdict(t) for t in completed_trades])
        daily_summaries = self._compute_daily_summaries(df_trades, equity_curve)

        daily_pnls = [d.net_pnl for d in daily_summaries]
        daily_returns = [d.daily_return_pct for d in daily_summaries]

        total_days = len(daily_summaries)
        avg_trades_day = (total_trades / total_days) if total_days > 0 else 0.0
        avg_daily_pnl = float(np.mean(daily_pnls)) if daily_pnls else 0.0
        avg_daily_ret = float(np.mean(daily_returns)) if daily_returns else 0.0
        best_day = float(np.max(daily_pnls)) if daily_pnls else 0.0
        worst_day = float(np.min(daily_pnls)) if daily_pnls else 0.0

        # Annualized Sharpe & Sortino (assuming 365 crypto trading days)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))
            downside_std = float(np.std([r for r in daily_returns if r < 0])) if any(r < 0 for r in daily_returns) else 1e-6
            sortino_ratio = float(np.mean(daily_returns) / (downside_std if downside_std > 0 else 1e-6) * np.sqrt(365))
        else:
            sharpe_ratio, sortino_ratio = 0.0, 0.0

        # ----------------------------------------------------------------------
        # 4. Export Logs
        # ----------------------------------------------------------------------
        trades_csv_path = self.logs_dir / f"backtest_trades_{self.coin}_{self.timeframe}.csv"
        daily_csv_path = self.logs_dir / f"backtest_daily_{self.coin}_{self.timeframe}.csv"

        if not df_trades.empty:
            df_trades.to_csv(trades_csv_path, index=False)
        else:
            pd.DataFrame(columns=list(TradeRecord.__annotations__.keys())).to_csv(trades_csv_path, index=False)

        df_daily = pd.DataFrame([asdict(d) for d in daily_summaries])
        df_daily.to_csv(daily_csv_path, index=False)

        return BacktestResult(
            coin=self.coin.upper(),
            timeframe=self.timeframe,
            threshold=effective_threshold,
            initial_capital=self.initial_capital,
            final_balance=final_balance,
            net_profit_usdt=net_profit_usdt,
            net_profit_pct=net_profit_pct,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_fees_paid=total_fees,
            total_trading_days=total_days,
            avg_trades_per_day=avg_trades_day,
            avg_daily_pnl_usdt=avg_daily_pnl,
            avg_daily_return_pct=avg_daily_ret,
            best_day_pnl=best_day,
            worst_day_pnl=worst_day,
            trades_log_path=trades_csv_path,
            daily_log_path=daily_csv_path,
        )

    def _compute_daily_summaries(
        self, df_trades: pd.DataFrame, equity_curve: List[Tuple[pd.Timestamp, float]]
    ) -> List[DailySummary]:
        """Aggregates trades and balance changes by calendar day."""
        if not equity_curve:
            return []

        df_eq = pd.DataFrame(equity_curve, columns=["datetime", "equity"])
        df_eq["date"] = df_eq["datetime"].dt.strftime("%Y-%m-%d")
        daily_closing_equity = df_eq.groupby("date")["equity"].last().to_dict()

        if not df_trades.empty:
            df_trades["exit_dt"] = pd.to_datetime(df_trades["exit_time"], utc=True)
            df_trades["date"] = df_trades["exit_dt"].dt.strftime("%Y-%m-%d")

        unique_dates = sorted(daily_closing_equity.keys())
        daily_list: List[DailySummary] = []
        prev_balance = self.initial_capital

        for dt_str in unique_dates:
            end_bal = daily_closing_equity[dt_str]

            if not df_trades.empty and dt_str in df_trades["date"].values:
                day_trades = df_trades[df_trades["date"] == dt_str]
                t_count = len(day_trades)
                w_count = int(np.count_nonzero(day_trades["net_pnl"] > 0))
                l_count = int(np.count_nonzero(day_trades["net_pnl"] <= 0))
                w_rate = (w_count / t_count * 100.0) if t_count > 0 else 0.0
                g_pnl = float(day_trades["gross_pnl"].sum())
                n_pnl = float(day_trades["net_pnl"].sum())
                t_fees = float(day_trades["total_fee"].sum())
            else:
                t_count, w_count, l_count, w_rate = 0, 0, 0, 0.0
                g_pnl, n_pnl, t_fees = 0.0, 0.0, 0.0

            d_ret = ((end_bal - prev_balance) / prev_balance) * 100.0 if prev_balance > 0 else 0.0

            daily_list.append(
                DailySummary(
                    date=dt_str,
                    total_trades=t_count,
                    wins=w_count,
                    losses=l_count,
                    win_rate_pct=w_rate,
                    gross_pnl=g_pnl,
                    net_pnl=n_pnl,
                    total_fees=t_fees,
                    ending_balance=end_bal,
                    daily_return_pct=d_ret,
                )
            )
            prev_balance = end_bal

        return daily_list


# ==============================================================================
# CLI INTERFACE & FORMATTED PRESENTATION
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parses CLI parameters."""
    parser = argparse.ArgumentParser(
        description="Event-Driven Spot Backtesting Engine with Meta-Model Decision Engine"
    )
    parser.add_argument("--coin", type=str, default=None, help="Target coin (e.g. btc, eth, sol)")
    parser.add_argument("--timeframe", type=str, default=None, help="Target timeframe (e.g. 5m, 15m, 30m)")
    parser.add_argument("--coins", type=str, nargs="+", default=["btc", "eth", "sol"], help="Coins list")
    parser.add_argument("--timeframes", type=str, nargs="+", default=["5m", "15m", "30m"], help="Timeframes list")
    parser.add_argument("--all", action="store_true", help="Run backtest across all coin-timeframe pairs")
    parser.add_argument("--threshold", type=float, default=0.50, help="AI Probability threshold (default: 0.50)")
    parser.add_argument("--auto-threshold", action="store_true", help="Use calibrated recommended threshold from model metadata")
    parser.add_argument("--capital", type=float, default=100.0, help="Initial capital in USDT (default: 100.0)")
    parser.add_argument("--trade_allocation", type=float, default=5.0, help="Trade allocation in USDT (default: 5.0)")
    parser.add_argument("--fee", type=float, default=0.00075, help="Trading fee rate (default: 0.00075)")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage rate (default: 0.0002)")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Processed continuous dataset directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory")
    return parser.parse_args()


def display_backtest_report(results: List[BacktestResult]) -> None:
    """Renders comprehensive performance and daily breakdown tables."""
    print("\n" + "=" * 145)
    print(
        f"{'Coin':<5} | {'TF':<5} | {'Thresh':<6} | {'Trades':<6} | {'Win Rate':<9} | "
        f"{'Profit Factor':<13} | {'Net PnL ($)':<12} | {'Return (%)':<11} | {'Max DD (%)':<10} | "
        f"{'Sharpe':<7} | {'Sortino':<8} | {'Trades/Day':<10}"
    )
    print("-" * 145)

    for r in results:
        print(
            f"{r.coin:<5} | {r.timeframe:<5} | {r.threshold:<6.2f} | {r.total_trades:<6} | {r.win_rate_pct:<8.2f}% | "
            f"{r.profit_factor:<13.2f} | {r.net_profit_usdt:<+12.2f} | {r.net_profit_pct:<+10.2f}% | "
            f"{r.max_drawdown_pct:<10.2f}% | {r.sharpe_ratio:<7.2f} | {r.sortino_ratio:<8.2f} | {r.avg_trades_per_day:<10.2f}"
        )
    print("=" * 145 + "\n")

    # Sample Daily Breakdown Report for pairs with executed trades
    active_results = [r for r in results if r.total_trades > 0]
    show_list = active_results[:3] if active_results else results[:3]

    for r in show_list:
        print(f"--- Daily Recap Sample (Recent 10 Days) for {r.coin} [{r.timeframe}] ---")
        df_daily = pd.read_csv(r.daily_log_path)
        print(
            f"{'Date':<12} | {'Trades':<7} | {'Wins':<5} | {'Losses':<6} | {'Net PnL (USDT)':<15} | {'Balance (USDT)':<15} | {'Return (%)':<10}"
        )
        print("-" * 80)
        for _, row in df_daily.tail(10).iterrows():
            print(
                f"{row['date']:<12} | {int(row['total_trades']):<7} | {int(row['wins']):<5} | {int(row['losses']):<6} | "
                f"{row['net_pnl']:<+15.2f} | {row['ending_balance']:<15.2f} | {row['daily_return_pct']:<+10.2f}%"
            )
        print(
            f"Daily Stats: Days={r.total_trading_days} | Avg Daily PnL=${r.avg_daily_pnl_usdt:+.2f} | "
            f"Best Day=${r.best_day_pnl:+.2f} | Worst Day=${r.worst_day_pnl:+.2f}\n"
        )


def main() -> None:
    """Main CLI execution flow."""
    args = parse_args()

    pairs_to_run: List[Tuple[str, str]] = []
    if args.coin and args.timeframe:
        pairs_to_run = [(args.coin.lower(), args.timeframe.lower())]
    elif args.all or (args.coins and args.timeframes):
        for c in args.coins:
            for tf in args.timeframes:
                pairs_to_run.append((c.lower(), tf.lower()))
    else:
        logger.error("Please specify --coin and --timeframe or --all.")
        sys.exit(1)

    all_results: List[BacktestResult] = []
    total_start = time.perf_counter()

    for coin, tf in pairs_to_run:
        engine = EventDrivenBacktester(
            coin=coin,
            timeframe=tf,
            initial_capital=args.capital,
            trade_allocation=args.trade_allocation,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            threshold=args.threshold,
            auto_threshold=args.auto_threshold,
            processed_dir=Path(args.processed_dir),
            models_dir=Path(args.models_dir),
            logs_dir=Path(args.logs_dir),
        )
        try:
            res = engine.run()
            all_results.append(res)
        except Exception as e:
            logger.error(f"Failed backtest for {coin} {tf}: {e}", exc_info=True)

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        f"=== Completed {len(all_results)} backtests in {total_elapsed:.2f}s ==="
    )

    if all_results:
        display_backtest_report(all_results)


if __name__ == "__main__":
    main()
