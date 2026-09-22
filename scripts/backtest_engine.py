#!/usr/bin/env python3
"""
Event-Driven Backtesting Engine for Spot Trading (Long-Only) Meta-Labeling System.

Supports both single-pair backtests and Unified Cross-Asset Portfolio Multi-Positioning
with Dynamic Risk Exposure (Max 50% Total Capital Exposure).

Simulates order execution, fee deduction, slippage, and dynamic ATR-based TP/SL
risk management on Out-of-Sample (OOS) Test datasets. Generates trade logs and
comprehensive daily performance metrics.

Author: Adaptive Trading System Team
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

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import talib

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_features import (
    FeatureExtractor,
    _compute_lorentzian_signals_parallel,
)
from src.config import REGIME_FUNNEL, RegimeFunnelConfig
from src.strategies.regime_funnel import RegimeFunnelStrategy

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
    coin: str
    timeframe: str
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
    sideways_bars: int = 0
    is_break_even: bool = False
    tp1_price: float = 0.0
    tp1_taken: bool = False
    original_quantity: float = 0.0
    original_notional: float = 0.0
    realized_gross: float = 0.0
    realized_fee: float = 0.0


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
    avg_concurrent_positions: float = 0.0
    peak_concurrent_positions: int = 0
    monthly_profit_usdt: float = 0.0
    monthly_profit_pct: float = 0.0
    trades_log_path: Path = field(default_factory=lambda: Path("logs/trades.csv"))
    daily_log_path: Path = field(default_factory=lambda: Path("logs/daily.csv"))


@dataclass
class UnifiedBacktestResult:
    """Performance metrics for Unified Multi-Asset Portfolio simulation."""
    scenario_name: str
    streams: List[str]
    trade_allocation: float
    max_exposure_pct: float
    initial_capital: float
    final_equity: float
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
    avg_concurrent_positions: float
    peak_concurrent_positions: int
    total_trading_days: int
    monthly_profit_usdt: float
    monthly_profit_pct: float
    total_fees_paid: float
    pair_breakdown: Dict[str, Dict[str, Any]]
    trades_log_path: Path
    daily_log_path: Path


class EventDrivenBacktester:
    """
    Event-driven spot trading simulation engine for Out-of-Sample evaluation on a single pair.
    Implements Dynamic Portfolio Multi-Positioning with Max 50% Capital Exposure.
    """

    def __init__(
        self,
        coin: str,
        timeframe: str,
        initial_capital: float = 100.0,
        trade_allocation: float = 15.0,
        max_exposure_pct: float = 0.50,
        max_open_positions: Optional[int] = 1,
        fee_rate: float = 0.00075,       # 0.075% Binance BNB tier
        slippage_rate: float = 0.0002,   # 0.02% execution slippage
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 12,
        cooldown_bars: int = 3,
        threshold: Optional[float] = None,
        auto_threshold: bool = True,
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
        self.max_exposure_pct = max_exposure_pct
        self.max_open_positions = max_open_positions or 1
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.cooldown_bars = cooldown_bars
        self.threshold = threshold
        self.auto_threshold = auto_threshold
        self.processed_dir = Path(processed_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.logs_dir = Path(logs_dir).resolve()
        self.test_ratio = test_ratio
        self.purge_buffer = purge_buffer

        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_oos_test_data(self) -> Tuple[pd.DataFrame, Any, float]:
        """
        Loads continuous candle dataset, generates features and primary buy triggers,
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
        if df_test.empty:
            raise ValueError(f"OOS test set is empty for {self.coin} {self.timeframe}")

        model = joblib.load(model_path)

        # Resolve threshold with priority:
        # 1. Explicit CLI / constructor parameter override (Highest Priority)
        # 2. Metadata recommended_threshold (if auto_threshold and metadata exists)
        # 3. Default fallback (0.44 for btc 5m, else 0.50)
        if self.threshold is not None:
            effective_threshold = float(self.threshold)
            logger.info(
                f"[{self.coin.upper()}] [{self.timeframe}] Using explicit CLI threshold override: {effective_threshold:.4f}"
            )
        elif self.auto_threshold and metadata_path.exists():
            default_th = 0.44 if self.coin == "btc" and self.timeframe == "5m" else 0.50
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    effective_threshold = float(meta.get("recommended_threshold", default_th))
                    logger.info(
                        f"[{self.coin.upper()}] [{self.timeframe}] Using calibrated auto-threshold from metadata: {effective_threshold:.4f}"
                    )
            except Exception as e:
                logger.warning(f"Could not load recommended threshold from metadata: {e}")
                effective_threshold = default_th
        else:
            effective_threshold = 0.44 if self.coin == "btc" and self.timeframe == "5m" else 0.50

        logger.info(
            f"[{self.coin.upper()}] [{self.timeframe}] Loaded OOS Test Set: {len(df_test):,} candles "
            f"({df_test['datetime'].iloc[0]} to {df_test['datetime'].iloc[-1]}) | Active Threshold: {effective_threshold:.4f}"
        )

        return df_test, model, effective_threshold

    def run(
        self,
        custom_signal_series: Optional[pd.Series] = None,
        use_ai_gate: bool = True,
    ) -> BacktestResult:
        """Executes event-driven backtest simulation across OOS test candles."""
        df_test, model, effective_threshold = self.load_oos_test_data()

        # Compute AI Meta-Model Inference Probabilities across all test candles
        X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        raw_probs = model.predict_proba(X_test)
        ai_probs = raw_probs[:, 1] if raw_probs.ndim == 2 and raw_probs.shape[1] > 1 else raw_probs.ravel()
        df_test["ai_prob"] = ai_probs

        if custom_signal_series is not None:
            df_test["primary_signal"] = custom_signal_series.values

        # Portfolio State
        cash_balance = self.initial_capital
        open_positions: List[Position] = []
        completed_trades: List[TradeRecord] = []
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        concurrent_positions_history: List[int] = []

        total_candles = len(df_test)
        trade_counter = 0
        last_exit_bar_idx: Optional[int] = None

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
                        exit_price = pos.stop_loss * (1.0 - self.slippage_rate)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    else:
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
                    last_exit_bar_idx = i
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

            # Current Portfolio Equity and Open Exposure
            unrealized_val = sum(p.quantity * c for p in open_positions)
            current_equity = cash_balance + unrealized_val
            current_open_exposure = sum(p.notional_value for p in open_positions)

            # ------------------------------------------------------------------
            # 2. Check Entry Conditions for New Buy Position (Strict Two-Stage Evaluation)
            # ------------------------------------------------------------------
            if use_ai_gate:
                signal_buy = primary_signal and (prob >= effective_threshold)
            else:
                signal_buy = primary_signal
            
            # Dynamic Multi-Positioning Risk Check: Max 1 active position per coin & Max 50% Capital Exposure
            has_active_pos = len(open_positions) >= self.max_open_positions
            exposure_limit = current_equity * self.max_exposure_pct
            
            # Anti-Cluster Cooldown Check: Minimum cooldown bars between consecutive trades
            is_cooldown_active = (last_exit_bar_idx is not None) and ((i - last_exit_bar_idx) < self.cooldown_bars)

            can_allocate = (
                (not has_active_pos)
                and (not is_cooldown_active)
                and (current_open_exposure + self.trade_allocation <= exposure_limit + 1e-6)
                and (cash_balance >= self.trade_allocation)
            )

            if signal_buy and can_allocate:
                trade_counter += 1
                entry_fill_price = c * (1.0 + self.slippage_rate)
                notional = self.trade_allocation
                entry_fee = notional * self.fee_rate
                quantity = notional / entry_fill_price

                tp_price = entry_fill_price + (self.tp_multiplier * atr)
                sl_price = entry_fill_price - (self.sl_multiplier * atr)

                cash_balance -= (notional + entry_fee)

                new_pos = Position(
                    position_id=f"T-{trade_counter:04d}",
                    coin=self.coin.upper(),
                    timeframe=self.timeframe,
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

                coin_symbol = f"{self.coin.upper()}USDT" if not self.coin.upper().endswith("USDT") else self.coin.upper()
                logger.info(
                    f"[TRADE ENTRY] Coin: {coin_symbol} | Time: {dt} | "
                    f"Primary Signal: YES | LightGBM Prob: {prob:.4f} >= Threshold {effective_threshold:.4f} | "
                    f"Status: APPROVED BY AI"
                )

            # Mark total equity & concurrent positions count at bar close
            unrealized_value = sum(p.quantity * c for p in open_positions)
            total_equity = cash_balance + unrealized_value
            equity_curve.append((dt, total_equity))
            concurrent_positions_history.append(len(open_positions))

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

        # Concurrent Positions Metrics
        avg_concurrent = float(np.mean(concurrent_positions_history)) if concurrent_positions_history else 0.0
        peak_concurrent = int(np.max(concurrent_positions_history)) if concurrent_positions_history else 0

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

        # Estimated Monthly Profit (normalized to 30.417 days/month)
        monthly_factor = 30.417 / total_days if total_days > 0 else 1.0
        monthly_profit_usdt = net_profit_usdt * monthly_factor
        monthly_profit_pct = net_profit_pct * monthly_factor

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
            avg_concurrent_positions=avg_concurrent,
            peak_concurrent_positions=peak_concurrent,
            monthly_profit_usdt=monthly_profit_usdt,
            monthly_profit_pct=monthly_profit_pct,
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
# UNIFIED CROSS-ASSET MULTI-POSITIONING PORTFOLIO ENGINE
# ==============================================================================

class UnifiedPortfolioBacktester:
    """
    Unified Multi-Asset Portfolio simulation engine.
    Synchronously manages a single capital pool ($100 USDT) across multiple live
    streams (e.g. BTC 5m, ETH 5m, SOL 15m) with Max 50% Capital Exposure.
    """

    def __init__(
        self,
        streams: Sequence[Tuple[str, str]] = (("btc", "15m"), ("eth", "15m"), ("sol", "15m")),
        initial_capital: float = 100.0,
        trade_allocation: float = 15.0,
        max_exposure_pct: float = 0.50,
        fee_rate: float = 0.00075,
        slippage_rate: float = 0.0002,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 12,
        cooldown_bars: int = 3,
        threshold: Optional[float] = None,
        processed_dir: Path = Path("dataset/processed"),
        models_dir: Path = Path("models"),
        logs_dir: Path = Path("logs"),
        scenario_name: str = "Scenario",
    ) -> None:
        self.streams = streams
        self.initial_capital = initial_capital
        self.trade_allocation = trade_allocation
        self.max_exposure_pct = max_exposure_pct
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.cooldown_bars = cooldown_bars
        self.threshold = threshold
        self.processed_dir = Path(processed_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.logs_dir = Path(logs_dir).resolve()
        self.scenario_name = scenario_name

        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_stream_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Loads and prepares OOS test datasets for all streams."""
        stream_data = {}

        for coin, tf in self.streams:
            key = f"{coin.upper()}_{tf}"
            engine = EventDrivenBacktester(
                coin=coin,
                timeframe=tf,
                initial_capital=self.initial_capital,
                trade_allocation=self.trade_allocation,
                cooldown_bars=self.cooldown_bars,
                threshold=self.threshold,
                auto_threshold=True,
                processed_dir=self.processed_dir,
                models_dir=self.models_dir,
                logs_dir=self.logs_dir,
            )
            df_test, model, threshold = engine.load_oos_test_data()

            # Predict probabilities
            X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            raw_probs = model.predict_proba(X_test)
            ai_probs = raw_probs[:, 1] if raw_probs.ndim == 2 and raw_probs.shape[1] > 1 else raw_probs.ravel()
            df_test["ai_prob"] = ai_probs
            df_test["coin"] = coin.upper()
            df_test["timeframe"] = tf
            df_test["threshold"] = threshold

            stream_data[key] = {
                "coin": coin.upper(),
                "timeframe": tf,
                "df": df_test,
                "threshold": threshold,
            }

        return stream_data

    def run(self) -> UnifiedBacktestResult:
        """Executes unified cross-asset multi-positioning simulation."""
        stream_data = self.load_stream_datasets()

        # Combine all candles into a chronological event stream
        candle_events: List[Dict[str, Any]] = []
        for key, s_info in stream_data.items():
            df = s_info["df"]
            for idx, row in df.iterrows():
                candle_events.append({
                    "datetime": row["datetime"],
                    "stream_key": key,
                    "coin": s_info["coin"],
                    "timeframe": s_info["timeframe"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "atr": float(row["atr"]),
                    "primary_signal": bool(row["primary_signal"]),
                    "ai_prob": float(row["ai_prob"]),
                    "threshold": float(s_info["threshold"]),
                    "bar_idx": idx,
                })

        # Sort strictly by timestamp ascending
        candle_events.sort(key=lambda x: x["datetime"])

        # Portfolio State
        cash_balance = self.initial_capital
        open_positions: List[Position] = []
        completed_trades: List[TradeRecord] = []
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        concurrent_positions_history: List[int] = []
        last_exit_by_coin: Dict[str, int] = {}

        # Latest close prices per coin for mark-to-market
        latest_prices: Dict[str, float] = {}
        trade_counter = 0

        # Group events by timestamp for synchronous intra-bar processing
        events_by_time: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
        for ev in candle_events:
            dt = ev["datetime"]
            events_by_time.setdefault(dt, []).append(ev)

        sorted_timestamps = sorted(events_by_time.keys())

        for dt in sorted_timestamps:
            current_events = events_by_time[dt]

            # Update latest known prices
            for ev in current_events:
                latest_prices[ev["coin"]] = ev["close"]

            # ------------------------------------------------------------------
            # 1. Process Exits across all active positions matching closing candles
            # ------------------------------------------------------------------
            remaining_positions: List[Position] = []

            for pos in open_positions:
                # Find matching candle event for this position's stream
                matching_ev = next(
                    (ev for ev in current_events if ev["coin"] == pos.coin and ev["timeframe"] == pos.timeframe),
                    None
                )

                if matching_ev is not None:
                    pos.bars_held += 1
                    o = matching_ev["open"]
                    h = matching_ev["high"]
                    l = matching_ev["low"]
                    c = matching_ev["close"]

                    hit_tp = h >= pos.take_profit
                    hit_sl = l <= pos.stop_loss
                    exit_triggered = False
                    exit_price = 0.0
                    exit_reason = ""

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
                            exit_price = pos.stop_loss * (1.0 - self.slippage_rate)
                            exit_reason = "STOP_LOSS"
                            exit_triggered = True
                        else:
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
                        exit_price = c * (1.0 - self.slippage_rate)
                        exit_reason = "TIME_EXPIRY"
                        exit_triggered = True

                    if exit_triggered:
                        last_exit_by_coin[pos.coin.upper()] = matching_ev["bar_idx"]
                        exit_value = pos.quantity * exit_price
                        exit_fee = exit_value * self.fee_rate
                        total_fee = pos.entry_fee + exit_fee
                        gross_pnl = exit_value - pos.notional_value
                        net_pnl = gross_pnl - total_fee
                        return_pct = (net_pnl / pos.notional_value) * 100.0

                        cash_balance += exit_value - exit_fee

                        trade_rec = TradeRecord(
                            trade_id=pos.position_id,
                            coin=pos.coin,
                            timeframe=pos.timeframe,
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
                else:
                    # No candle close for this stream at this exact timestamp
                    remaining_positions.append(pos)

            open_positions = remaining_positions

            # ------------------------------------------------------------------
            # 2. Process Entries for streams with valid AI Buy Signals
            # ------------------------------------------------------------------
            for ev in current_events:
                # Mark to market equity
                unrealized_val = sum(p.quantity * latest_prices.get(p.coin, p.entry_price) for p in open_positions)
                current_equity = cash_balance + unrealized_val
                current_open_exposure = sum(p.notional_value for p in open_positions)
                exposure_limit = current_equity * self.max_exposure_pct

                # Healthy Multi-Positioning: strictly Max 1 active position per coin
                has_coin_position = any(p.coin.upper() == ev["coin"].upper() for p in open_positions)
                last_exit_bar = last_exit_by_coin.get(ev["coin"].upper())
                is_cooldown_active = (last_exit_bar is not None) and ((ev["bar_idx"] - last_exit_bar) < self.cooldown_bars)

                signal_buy = ev["primary_signal"] and (ev["ai_prob"] >= ev["threshold"])
                can_allocate = (
                    (not has_coin_position)
                    and (not is_cooldown_active)
                    and (current_open_exposure + self.trade_allocation <= exposure_limit + 1e-6)
                    and (cash_balance >= self.trade_allocation)
                )

                if signal_buy and can_allocate:
                    trade_counter += 1
                    c = ev["close"]
                    atr = ev["atr"]
                    entry_fill_price = c * (1.0 + self.slippage_rate)
                    notional = self.trade_allocation
                    entry_fee = notional * self.fee_rate
                    quantity = notional / entry_fill_price

                    tp_price = entry_fill_price + (self.tp_multiplier * atr)
                    sl_price = entry_fill_price - (self.sl_multiplier * atr)

                    cash_balance -= (notional + entry_fee)

                    new_pos = Position(
                        position_id=f"UT-{trade_counter:04d}",
                        coin=ev["coin"],
                        timeframe=ev["timeframe"],
                        entry_index=ev["bar_idx"],
                        entry_time=dt,
                        entry_price=entry_fill_price,
                        quantity=quantity,
                        notional_value=notional,
                        entry_fee=entry_fee,
                        take_profit=tp_price,
                        stop_loss=sl_price,
                        atr=atr,
                        ai_prob=ev["ai_prob"],
                        max_holding_bars=self.max_holding_bars,
                    )
                    open_positions.append(new_pos)

                    coin_symbol = f"{ev['coin']}USDT" if not ev["coin"].endswith("USDT") else ev["coin"]
                    logger.info(
                        f"[TRADE ENTRY] Coin: {coin_symbol} | Time: {dt} | "
                        f"Primary Signal: YES | LightGBM Prob: {ev['ai_prob']:.4f} >= Threshold {ev['threshold']:.4f} | "
                        f"Status: APPROVED BY AI"
                    )

            # Record timestamp equity snapshot
            unrealized_val = sum(p.quantity * latest_prices.get(p.coin, p.entry_price) for p in open_positions)
            total_equity = cash_balance + unrealized_val
            equity_curve.append((dt, total_equity))
            concurrent_positions_history.append(len(open_positions))

        # Close any open positions at the end of the simulation
        if open_positions:
            last_dt = sorted_timestamps[-1]
            for pos in open_positions:
                last_close = latest_prices.get(pos.coin, pos.entry_price)
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
                    coin=pos.coin,
                    timeframe=pos.timeframe,
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

        final_equity = cash_balance
        net_profit_usdt = final_equity - self.initial_capital
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

        # Max Drawdown
        equities = np.array([eq[1] for eq in equity_curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        max_drawdown_pct = abs(float(np.min(drawdowns))) * 100.0 if len(drawdowns) > 0 else 0.0

        avg_concurrent = float(np.mean(concurrent_positions_history)) if concurrent_positions_history else 0.0
        peak_concurrent = int(np.max(concurrent_positions_history)) if concurrent_positions_history else 0

        # Daily summaries
        df_trades = pd.DataFrame([asdict(t) for t in completed_trades])
        dummy_engine = EventDrivenBacktester("btc", "5m", initial_capital=self.initial_capital)
        daily_summaries = dummy_engine._compute_daily_summaries(df_trades, equity_curve)
        total_days = len(daily_summaries)

        daily_returns = [d.daily_return_pct for d in daily_summaries]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))
            downside_std = float(np.std([r for r in daily_returns if r < 0])) if any(r < 0 for r in daily_returns) else 1e-6
            sortino_ratio = float(np.mean(daily_returns) / (downside_std if downside_std > 0 else 1e-6) * np.sqrt(365))
        else:
            sharpe_ratio, sortino_ratio = 0.0, 0.0

        monthly_factor = 30.417 / total_days if total_days > 0 else 1.0
        monthly_profit_usdt = net_profit_usdt * monthly_factor
        monthly_profit_pct = net_profit_pct * monthly_factor

        # Per-pair breakdown
        pair_breakdown: Dict[str, Dict[str, Any]] = {}
        for coin, tf in self.streams:
            key = f"{coin.upper()}_{tf}"
            pair_trades = [t for t in completed_trades if t.coin == coin.upper() and t.timeframe == tf]
            p_total = len(pair_trades)
            p_wins = len([t for t in pair_trades if t.net_pnl > 0])
            p_wr = (p_wins / p_total * 100.0) if p_total > 0 else 0.0
            p_pnl = sum(t.net_pnl for t in pair_trades)
            pair_breakdown[key] = {
                "total_trades": p_total,
                "wins": p_wins,
                "win_rate_pct": p_wr,
                "net_pnl_usdt": p_pnl,
            }

        alloc_tag = f"{self.trade_allocation:.0f}usdt"
        trades_csv_path = self.logs_dir / f"unified_backtest_trades_{alloc_tag}.csv"
        daily_csv_path = self.logs_dir / f"unified_backtest_daily_{alloc_tag}.csv"

        if not df_trades.empty:
            df_trades.to_csv(trades_csv_path, index=False)
        else:
            pd.DataFrame(columns=list(TradeRecord.__annotations__.keys())).to_csv(trades_csv_path, index=False)

        df_daily = pd.DataFrame([asdict(d) for d in daily_summaries])
        df_daily.to_csv(daily_csv_path, index=False)

        stream_labels = [f"{c.upper()}[{tf}]" for c, tf in self.streams]

        return UnifiedBacktestResult(
            scenario_name=self.scenario_name,
            streams=stream_labels,
            trade_allocation=self.trade_allocation,
            max_exposure_pct=self.max_exposure_pct,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
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
            avg_concurrent_positions=avg_concurrent,
            peak_concurrent_positions=peak_concurrent,
            total_trading_days=total_days,
            monthly_profit_usdt=monthly_profit_usdt,
            monthly_profit_pct=monthly_profit_pct,
            total_fees_paid=total_fees,
            pair_breakdown=pair_breakdown,
            trades_log_path=trades_csv_path,
            daily_log_path=daily_csv_path,
        )


# ==============================================================================
# CLI INTERFACE & FORMATTED PRESENTATION
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parses CLI parameters."""
    parser = argparse.ArgumentParser(
        description="Event-Driven Spot Backtesting Engine with Multi-Positioning & Unified Portfolio"
    )
    parser.add_argument("--coin", type=str, default=None, help="Target coin (e.g. btc, eth, sol)")
    parser.add_argument("--timeframe", type=str, default=None, help="Target timeframe (e.g. 5m, 15m, 30m)")
    parser.add_argument("--coins", type=str, nargs="+", default=["btc", "eth", "sol"], help="Coins list")
    parser.add_argument("--timeframes", type=str, nargs="+", default=["5m", "15m", "30m"], help="Timeframes list")
    parser.add_argument("--all", action="store_true", help="Run backtest across all coin-timeframe pairs")
    parser.add_argument("--unified", action="store_true", help="Run Unified Multi-Asset Portfolio simulation")
    parser.add_argument("--scenarios", action="store_true", help="Run multi-allocation scenarios ($5 and $10 USDT)")
    parser.add_argument("--ab-test", "--benchmark", dest="ab_test", action="store_true", help="Run comparative A/B testing benchmark (Control vs Experimental)")
    parser.add_argument("--regime", action="store_true", help="Run Unsupervised Market Regime State-Transition backtest")
    parser.add_argument("--regime-hmm", "--regime_hmm", dest="regime_hmm", action="store_true", help="Run Gaussian HMM Market Regime State-Transition backtest")
    parser.add_argument(
        "--streams",
        type=str,
        nargs="+",
        default=["btc:15m", "eth:15m", "sol:15m"],
        help="Streams for unified portfolio in coin:timeframe format (default: btc:15m eth:15m sol:15m)",
    )
    parser.add_argument("--threshold", type=float, default=None, help="AI Probability threshold (default: auto from metadata)")
    parser.add_argument("--auto-threshold", action="store_true", default=True, help="Use calibrated recommended threshold from metadata")
    parser.add_argument("--cooldown-bars", "--cooldown_bars", dest="cooldown_bars", type=int, default=3, help="Anti-cluster cooldown bars (default: 3)")
    parser.add_argument("--capital", type=float, default=100.0, help="Initial capital in USDT (default: 100.0)")
    parser.add_argument("--trade_allocation", "--trade-allocation", dest="trade_allocation", type=float, default=15.0, help="Trade allocation in USDT (default: 15.0)")
    parser.add_argument("--max-exposure-pct", type=float, default=0.50, help="Max open exposure pct (default: 0.50)")
    parser.add_argument("--fee", type=float, default=0.0002, help="Trading fee rate (default: 0.0002, 0.02%% maker tier)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage rate (default: 0.0000, 0.00%% maker limit order)")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory")
    return parser.parse_args()


def display_backtest_report(results: List[BacktestResult]) -> None:
    """Renders comprehensive performance table for single-pair runs."""
    print("\n" + "=" * 160)
    print(
        f"{'Coin':<5} | {'TF':<5} | {'Thresh':<6} | {'Trades':<6} | {'Win Rate':<9} | "
        f"{'Profit Factor':<13} | {'Net PnL ($)':<12} | {'Return (%)':<11} | {'Max DD (%)':<10} | "
        f"{'Avg Concur':<10} | {'Peak Pos':<8} | {'Est. Mo. PnL ($)':<16} | {'Sharpe':<7}"
    )
    print("-" * 160)

    for r in results:
        print(
            f"{r.coin:<5} | {r.timeframe:<5} | {r.threshold:<6.2f} | {r.total_trades:<6} | {r.win_rate_pct:<8.2f}% | "
            f"{r.profit_factor:<13.2f} | {r.net_profit_usdt:<+12.2f} | {r.net_profit_pct:<+10.2f}% | "
            f"{r.max_drawdown_pct:<10.2f}% | {r.avg_concurrent_positions:<10.2f} | {r.peak_concurrent_positions:<8} | "
            f"{r.monthly_profit_usdt:<+16.2f} | {r.sharpe_ratio:<7.2f}"
        )
    print("=" * 160 + "\n")


def display_unified_scenario_report(results: List[UnifiedBacktestResult]) -> None:
    """Renders comprehensive comparison report for Unified Portfolio scenarios."""
    print("\n" + "=" * 135)
    print("           UNIFIED MULTI-ASSET PORTFOLIO BACKTEST REPORT (MAX 50% CAPITAL EXPOSURE)           ")
    print("=" * 135)

    headers = [
        "Metrik Portofolio",
        "Skenario 1 ($5.0 USDT / Trade)",
        "Skenario 2 ($10.0 USDT / Trade)",
    ]

    r1 = results[0] if len(results) > 0 else None
    r2 = results[1] if len(results) > 1 else None

    def val(r: Optional[UnifiedBacktestResult], attr: str, fmt: str = "{}") -> str:
        if r is None:
            return "N/A"
        v = getattr(r, attr)
        return fmt.format(v)

    rows = [
        ("Portfolio Streams", ", ".join(r1.streams) if r1 else "N/A", ", ".join(r2.streams) if r2 else "N/A"),
        ("Modal Awal (Initial Capital)", f"${r1.initial_capital:.2f}" if r1 else "N/A", f"${r2.initial_capital:.2f}" if r2 else "N/A"),
        ("Alokasi per Posisi (Trade Allocation)", f"${r1.trade_allocation:.2f} USDT" if r1 else "N/A", f"${r2.trade_allocation:.2f} USDT" if r2 else "N/A"),
        ("Max Exposure Allowed", f"{r1.max_exposure_pct*100:.0f}% (${r1.initial_capital*r1.max_exposure_pct:.0f} USDT)" if r1 else "N/A", f"{r2.max_exposure_pct*100:.0f}% (${r2.initial_capital*r2.max_exposure_pct:.0f} USDT)" if r2 else "N/A"),
        ("Total Trades Dieksekusi", val(r1, "total_trades", "{:,}"), val(r2, "total_trades", "{:,}")),
        ("Winning Trades / Losing Trades", f"{r1.winning_trades} / {r1.losing_trades}" if r1 else "N/A", f"{r2.winning_trades} / {r2.losing_trades}" if r2 else "N/A"),
        ("Win Rate (%)", val(r1, "win_rate_pct", "{:.2f}%"), val(r2, "win_rate_pct", "{:.2f}%")),
        ("Rata-rata Posisi Serentak (Avg Concurrent)", val(r1, "avg_concurrent_positions", "{:.2f} trades"), val(r2, "avg_concurrent_positions", "{:.2f} trades")),
        ("Peak Posisi Terbuka Serentak (Max Peak)", val(r1, "peak_concurrent_positions", "{} trades"), val(r2, "peak_concurrent_positions", "{} trades")),
        ("Final Equity ($)", val(r1, "final_equity", "${:.2f}"), val(r2, "final_equity", "${:.2f}")),
        ("Net Profit ($ & %)", f"${r1.net_profit_usdt:+.2f} ({r1.net_profit_pct:+.2f}%)" if r1 else "N/A", f"${r2.net_profit_usdt:+.2f} ({r2.net_profit_pct:+.2f}%)" if r2 else "N/A"),
        ("Profit Factor", val(r1, "profit_factor", "{:.2f}"), val(r2, "profit_factor", "{:.2f}")),
        ("Max Drawdown Portofolio (%)", val(r1, "max_drawdown_pct", "{:.2f}%"), val(r2, "max_drawdown_pct", "{:.2f}%")),
        ("Sharpe Ratio (Annualized)", val(r1, "sharpe_ratio", "{:.2f}"), val(r2, "sharpe_ratio", "{:.2f}")),
        ("Sortino Ratio (Annualized)", val(r1, "sortino_ratio", "{:.2f}"), val(r2, "sortino_ratio", "{:.2f}")),
        ("Total Fees Paid ($)", val(r1, "total_fees_paid", "${:.2f}"), val(r2, "total_fees_paid", "${:.2f}")),
        ("Estimasi Profit Rata-rata per Bulan ($)", val(r1, "monthly_profit_usdt", "${:+.2f} / bln"), val(r2, "monthly_profit_usdt", "${:+.2f} / bln")),
        ("Estimasi Profit Rata-rata per Bulan (%)", val(r1, "monthly_profit_pct", "{:+.2f}% / bln"), val(r2, "monthly_profit_pct", "{:+.2f}% / bln")),
    ]

    print(f"{headers[0]:<45} | {headers[1]:<40} | {headers[2]:<40}")
    print("-" * 135)
    for title, c1, c2 in rows:
        print(f"{title:<45} | {c1:<40} | {c2:<40}")
    print("=" * 135 + "\n")

    # Per-pair breakdown print
    if r1 and r1.pair_breakdown:
        print("--- Rincian Kontribusi per Pair (Skenario 1 - $5.0 USDT Allocation) ---")
        print(f"{'Stream':<15} | {'Total Trades':<14} | {'Wins':<8} | {'Win Rate (%)':<14} | {'Net PnL (USDT)':<15}")
        print("-" * 75)
        for p_key, p_stat in r1.pair_breakdown.items():
            print(
                f"{p_key:<15} | {p_stat['total_trades']:<14} | {p_stat['wins']:<8} | "
                f"{p_stat['win_rate_pct']:<13.2f}% | {p_stat['net_pnl_usdt']:<+15.2f}"
            )
        print("-" * 75 + "\n")

    if r2 and r2.pair_breakdown:
        print("--- Rincian Kontribusi per Pair (Skenario 2 - $10.0 USDT Allocation) ---")
        print(f"{'Stream':<15} | {'Total Trades':<14} | {'Wins':<8} | {'Win Rate (%)':<14} | {'Net PnL (USDT)':<15}")
        print("-" * 75)
        for p_key, p_stat in r2.pair_breakdown.items():
            print(
                f"{p_key:<15} | {p_stat['total_trades']:<14} | {p_stat['wins']:<8} | "
                f"{p_stat['win_rate_pct']:<13.2f}% | {p_stat['net_pnl_usdt']:<+15.2f}"
            )
        print("-" * 75 + "\n")


def run_ab_benchmark(
    coin: str = "btc",
    timeframe: str = "5m",
    initial_capital: float = 100.0,
    trade_allocation: float = 20.0,
    threshold: Optional[float] = None,
    processed_dir: Path = Path("dataset/processed"),
    models_dir: Path = Path("models"),
    logs_dir: Path = Path("logs"),
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Executes Comparative A/B Testing Benchmark on Out-of-Sample (OOS) dataset:
    - Skenario 1 (Control Group): Pure Lorentzian Baseline (No AI Gatekeeper)
    - Skenario 2 (Experimental Group): Full Composite Primary Signal + LightGBM Meta-Gate (P >= threshold)
    """
    coin_clean = coin.lower()
    tf_clean = timeframe.lower()

    engine = EventDrivenBacktester(
        coin=coin_clean,
        timeframe=tf_clean,
        initial_capital=initial_capital,
        trade_allocation=trade_allocation,
        max_exposure_pct=0.50,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        max_holding_bars=12,
        cooldown_bars=3,
        threshold=threshold,
        auto_threshold=True,
        processed_dir=processed_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )

    df_test, _, effective_threshold = engine.load_oos_test_data()

    # Define Signal Masks:
    # 1. Pure Lorentzian Flip
    lorentz_flip = (df_test["lorentzian_signal"] == 1.0) & (df_test["lorentzian_signal"].shift(1) != 1.0)

    # 2. WaveTrend Crossover Up in Oversold Zone (RSI < 40)
    wt_cross = (df_test["rsi"] < 40.0) & (df_test["wt_diff"] > 0.0) & (df_test["wt_diff"].shift(1) <= 0.0)

    # Composite Signal (Lorentzian Flip OR RSI < 40 + WT Crossover)
    composite_signal = lorentz_flip | wt_cross

    logger.info(f"=== Running Skenario 1: Pure Lorentzian Baseline (Control Group) [{coin_clean.upper()} {tf_clean}] ===")
    res_control = engine.run(custom_signal_series=lorentz_flip, use_ai_gate=False)

    logger.info(f"=== Running Skenario 2: Full Composite + LightGBM Meta-Gate [{coin_clean.upper()} {tf_clean}] (P >= {effective_threshold:.4f}) ===")
    res_experimental = engine.run(custom_signal_series=composite_signal, use_ai_gate=True)

    return res_control, res_experimental


# Alias for backward compatibility and test runners
run_ab_test = run_ab_benchmark


def display_ab_test_report(res_control: BacktestResult, res_exp: BacktestResult) -> None:
    """Renders comparative A/B testing benchmark table and metrics."""
    coin_display = res_control.coin if res_control.coin.endswith("USDT") else f"{res_control.coin}USDT"
    report_title = f"A/B TESTING COMPARATIVE BENCHMARK REPORT (OOS {coin_display} {res_control.timeframe})"
    print("\n" + "=" * 125)
    print(f"{report_title:^125}")
    print("=" * 125)
    print(
        f"{'Skenario':<55} | {'Total Trades':<12} | {'Win Rate (%)':<12} | "
        f"{'Profit Factor':<13} | {'Net PnL ($)':<11} | {'Max DD (%)':<10} | {'Sharpe':<8}"
    )
    print("-" * 125)
    print(
        f"{'Skenario 1: Pure Lorentzian Baseline (Control)':<55} | "
        f"{res_control.total_trades:<12} | {res_control.win_rate_pct:<12.2f} | "
        f"{res_control.profit_factor:<13.2f} | {res_control.net_profit_usdt:<+11.2f} | "
        f"{res_control.max_drawdown_pct:<10.2f} | {res_control.sharpe_ratio:<8.2f}"
    )
    print(
        f"{'Skenario 2: Full Composite + LightGBM Gate (Experimental)':<55} | "
        f"{res_exp.total_trades:<12} | {res_exp.win_rate_pct:<12.2f} | "
        f"{res_exp.profit_factor:<13.2f} | {res_exp.net_profit_usdt:<+11.2f} | "
        f"{res_exp.max_drawdown_pct:<10.2f} | {res_exp.sharpe_ratio:<8.2f}"
    )
    print("=" * 125 + "\n")


def run_regime_backtest(
    coin: str = "btc",
    timeframe: str = "15m",
    model_type: str = "hmm",
    initial_capital: float = 100.0,
    trade_allocation: float = 20.0,
    max_exposure_pct: float = 0.50,
    fee_rate: float = 0.0002,
    slippage_rate: float = 0.0,
    tp_multiplier: Optional[float] = None,
    sl_multiplier: float = 0.70,
    be_trigger_multiplier: Optional[float] = None,
    max_holding_bars: int = 12,
    cooldown_bars: int = 3,
    processed_dir: Path = Path("dataset/processed"),
    models_dir: Path = Path("models"),
    logs_dir: Path = Path("logs"),
) -> Tuple[BacktestResult, Dict[str, int]]:
    """
    Executes Unsupervised Market Regime State-Transition Backtest on Out-of-Sample (OOS) dataset:
    - Ingests continuous microstructure features (log_return, volatility, volume_intensity)
    - Loads pre-trained Gaussian HMM / GMM regime model & scaler
    - Causal Online Inference: Computes forward-filtered state probabilities (Zero Lookahead Bias)
    - Pure Entry: Transition into Bullish Momentum State (learned transition, no manual heuristics)
    - Pure Exit: Transition out of Bullish State, OR TP 2.0x ATR / SL 1.0x ATR / Time Horizon
    """
    from scripts.train_regime_model import (
        MicrostructureFeatureExtractor,
        REGIME_FEATURE_COLUMNS,
        compute_causal_hmm_states,
        auto_resample_continuous_ohlcv,
    )

    coin_clean = coin.lower()
    tf_clean = timeframe.lower()

    if tp_multiplier is None:
        tp_multiplier = 1.20 if tf_clean == "1h" else 0.85
    if be_trigger_multiplier is None:
        be_trigger_multiplier = 0.80 if tf_clean == "1h" else 0.60

    data_path = processed_dir / coin_clean / tf_clean / f"{coin_clean}_{tf_clean}_continuous.parquet"
    if not data_path.exists():
        data_path = processed_dir / f"{coin_clean}_{tf_clean}_continuous.parquet"
    if not data_path.exists():
        try:
            data_path = auto_resample_continuous_ohlcv(processed_dir, coin_clean, tf_clean)
        except Exception:
            pass

    # Resolve model file
    model_path = models_dir / f"{coin_clean}_{tf_clean}_regime_{model_type.lower()}.joblib"
    if not model_path.exists():
        model_path = models_dir / f"{coin_clean}_{tf_clean}_regime_model.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Continuous dataset not found at: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained regime model not found at: {model_path}. "
            f"Please run `python3 scripts/train_regime_model.py --coin {coin_clean} --timeframe {tf_clean}` first."
        )

    df = pd.read_parquet(data_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.sort_values(by="datetime", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Initialize unified Strategy SSOT
    strategy_config = RegimeFunnelConfig(
        hmm_model_path=str(model_path),
        hmm_bullish_state_id=REGIME_FUNNEL.hmm_bullish_state_id,
        state_age_max=REGIME_FUNNEL.state_age_max,
        ema_trend_period=REGIME_FUNNEL.ema_trend_period,
        single_shot_per_episode=REGIME_FUNNEL.single_shot_per_episode,
        pullback_ema_period=REGIME_FUNNEL.pullback_ema_period,
        pullback_rsi_period=REGIME_FUNNEL.pullback_rsi_period,
        pullback_rsi_threshold=REGIME_FUNNEL.pullback_rsi_threshold,
        risk_sl_mult=sl_multiplier,
        risk_tp1_mult=be_trigger_multiplier,
        risk_tp2_mult=tp_multiplier,
        risk_be_buffer=REGIME_FUNNEL.risk_be_buffer,
        trade_allocation=trade_allocation,
        maker_fee=fee_rate,
        slippage=slippage_rate,
    )
    strategy = RegimeFunnelStrategy(config=strategy_config)

    df = strategy.compute_indicators(df)
    feature_cols = strategy.feature_columns
    bullish_state_id = strategy.bullish_state_id
    bearish_state_id = strategy.bearish_state_id
    sideways_state_id = strategy.sideways_state_id
    detected_model_type = "hmm" if hasattr(strategy.model, "transmat_") else "gmm"

    df.dropna(subset=feature_cols + ["atr", "ema_9", "rsi_14", "ema_200"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Chronological Split: 70% Train, Purge 12, 15% Val, Purge 12, 15% Test
    n = len(df)
    train_end = int(n * 0.70)
    val_start = train_end + 12
    val_end = val_start + int(n * 0.15)
    test_start = val_end + 12

    df_test = df.iloc[test_start:].copy().reset_index(drop=True)
    if df_test.empty:
        raise ValueError(f"OOS test set is empty for {coin_clean} {tf_clean}")

    # Causal Online Inference (Zero Lookahead Bias)
    if hasattr(strategy.model, "transmat_"):
        causal_states, _ = strategy.compute_causal_states(df_test)
        df_test["regime_state"] = causal_states
    else:
        X_test = strategy.scaler.transform(df_test[feature_cols].to_numpy(dtype=np.float64))
        df_test["regime_state"] = strategy.model.predict(X_test)

    logger.info(
        f"[{coin_clean.upper()}] [{tf_clean}] Loaded OOS {detected_model_type.upper()} Test Set: {len(df_test):,} candles "
        f"({df_test['datetime'].iloc[0]} to {df_test['datetime'].iloc[-1]}) | Bullish State: {bullish_state_id}"
    )

    # Portfolio State
    cash_balance = initial_capital
    open_positions: List[Position] = []
    completed_trades: List[TradeRecord] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    concurrent_positions_history: List[int] = []

    total_candles = len(df_test)
    trade_counter = 0
    last_exit_bar_idx: Optional[int] = None
    state_age = 0
    traded_in_current_episode = False
    pending_entry: Optional[Tuple[int, float, Any]] = None

    for i in range(total_candles):
        current_bar = df_test.iloc[i]
        dt = current_bar["datetime"]
        o = float(current_bar["open"])
        h = float(current_bar["high"])
        l = float(current_bar["low"])
        c = float(current_bar["close"])
        atr = float(current_bar["atr"])
        ema9 = float(current_bar["ema_9"])
        rsi14 = float(current_bar["rsi_14"])
        ema200 = float(current_bar["ema_200"])
        curr_state = int(current_bar["regime_state"])

        # Next-Bar Open Execution: Fill pending BUY orders at current bar Open (t+1)
        if pending_entry is not None and not open_positions:
            signal_bar_idx, signal_atr, signal_dt = pending_entry
            trade_counter += 1
            entry_fill_price = o * (1.0 + slippage_rate)
            notional = trade_allocation
            entry_fee = notional * fee_rate
            quantity = notional / entry_fill_price

            # Pilar 3 Initial Risk Boundaries (Scaling Out 50:50)
            brackets = strategy.calculate_risk_brackets(entry_fill_price, signal_atr)
            tp1_price = brackets["tp1_price"]
            tp2_price = brackets["tp2_price"]
            sl_price = brackets["sl_price"]

            cash_balance -= (notional + entry_fee)

            new_pos = Position(
                position_id=f"RT-{trade_counter:04d}",
                coin=coin_clean.upper(),
                timeframe=tf_clean,
                entry_index=i,
                entry_time=dt,
                entry_price=entry_fill_price,
                quantity=quantity,
                notional_value=notional,
                entry_fee=entry_fee,
                take_profit=tp2_price,
                stop_loss=sl_price,
                atr=signal_atr,
                ai_prob=1.0,
                max_holding_bars=max_holding_bars,
                sideways_bars=0,
                is_break_even=False,
                tp1_price=tp1_price,
                tp1_taken=False,
                original_quantity=quantity,
                original_notional=notional,
                realized_gross=0.0,
                realized_fee=0.0,
            )
            open_positions.append(new_pos)
            pending_entry = None

        # ==============================================================================
        # PILAR 1: Macro Regime Gate (Episode Tracking & Freshness Constraint)
        # ==============================================================================
        if curr_state == bullish_state_id:
            state_age += 1
        else:
            state_age = 0
            traded_in_current_episode = False

        # ==============================================================================
        # PILAR 3: Position Management & Dynamic Risk Engine (Scaling Out 50:50, SL, TP)
        # ==============================================================================
        remaining_positions: List[Position] = []
        for pos in open_positions:
            pos.bars_held += 1

            # Intra-bar Worst-Case Ordering Check: Low (Stop Loss) occurs first before High
            hit_sl = l <= pos.stop_loss
            hit_emergency_breakdown = (curr_state == bearish_state_id)
            hit_time_expiry = (pos.bars_held >= pos.max_holding_bars)

            exit_triggered = False
            exit_price = 0.0
            exit_reason = ""

            if hit_sl:
                # Worst-case ordering: Low breached Stop Loss
                exit_price = pos.stop_loss * (1.0 - slippage_rate)
                exit_reason = "BREAK_EVEN_SL" if (pos.tp1_taken or pos.is_break_even) else "STOP_LOSS"
                exit_triggered = True
            else:
                # Stop loss was NOT hit on this bar.
                # 1. Target Tahap 1 (TP1 - Amankan 50% Posisi): High >= Entry + 0.80 * ATR
                if (not pos.tp1_taken) and (h >= pos.tp1_price):
                    # Tutup 50% alokasi posisi pada harga TP1
                    tp1_qty = pos.quantity * 0.5
                    tp1_notional = pos.notional_value * 0.5
                    tp1_entry_fee = pos.entry_fee * 0.5
                    tp1_exit_price = pos.tp1_price * (1.0 - slippage_rate)
                    tp1_exit_val = tp1_qty * tp1_exit_price
                    tp1_exit_fee = tp1_exit_val * fee_rate
                    tp1_gross = tp1_exit_val - tp1_notional
                    tp1_fee = tp1_entry_fee + tp1_exit_fee

                    cash_balance += tp1_exit_val - tp1_exit_fee
                    pos.realized_gross += tp1_gross
                    pos.realized_fee += tp1_fee
                    pos.quantity -= tp1_qty
                    pos.notional_value -= tp1_notional
                    pos.entry_fee -= tp1_entry_fee
                    pos.tp1_taken = True
                    pos.is_break_even = True
                    # Geser Stop Loss sisa 50% posisi ke: Entry_Price * 1.0025 (+0.25% net fee lock)
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price * strategy.config.risk_be_buffer)

                # 2. Target Tahap 2 (TP2 - Sisa 50% Posisi): High >= Entry + 1.20 * ATR
                hit_tp2 = (pos.tp1_taken) and (h >= pos.take_profit)
                if hit_tp2:
                    exit_price = pos.take_profit * (1.0 - slippage_rate)
                    exit_reason = "TAKE_PROFIT"
                    exit_triggered = True
                elif hit_emergency_breakdown:
                    exit_price = c * (1.0 - slippage_rate)
                    exit_reason = "REGIME_CHANGE"
                    exit_triggered = True
                elif hit_time_expiry:
                    exit_price = c * (1.0 - slippage_rate)
                    exit_reason = "TIME_EXPIRY"
                    exit_triggered = True

            if exit_triggered:
                last_exit_bar_idx = i
                exit_value = pos.quantity * exit_price
                exit_fee = exit_value * fee_rate
                total_fee = pos.entry_fee + exit_fee + pos.realized_fee
                gross_pnl = (exit_value - pos.notional_value) + pos.realized_gross
                net_pnl = gross_pnl - total_fee
                orig_notional = pos.original_notional if pos.original_notional > 0 else pos.notional_value
                return_pct = (net_pnl / orig_notional) * 100.0

                cash_balance += exit_value - exit_fee

                trade_rec = TradeRecord(
                    trade_id=pos.position_id,
                    coin=coin_clean.upper(),
                    timeframe=tf_clean,
                    entry_time=str(pos.entry_time),
                    exit_time=str(dt),
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    notional_value=orig_notional,
                    exit_value=exit_value,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    return_pct=return_pct,
                    entry_fee=pos.entry_fee,
                    exit_fee=exit_fee,
                    total_fee=total_fee,
                    exit_reason=exit_reason,
                    bars_held=pos.bars_held,
                    ai_prob=1.0,
                )
                completed_trades.append(trade_rec)
            else:
                remaining_positions.append(pos)

        open_positions = remaining_positions

        unrealized_val = sum(p.quantity * c for p in open_positions)
        current_equity = cash_balance + unrealized_val
        current_open_exposure = sum(p.notional_value for p in open_positions)

        # ==============================================================================
        # PILAR 1 & 2: Cascading Entry Gates (SSOT Regime Funnel Strategy)
        # ==============================================================================
        cascading_buy_signal, _ = strategy.evaluate_gates(
            df=df_test,
            idx=i,
            current_state=curr_state,
            state_age=state_age,
            traded_in_episode=traded_in_current_episode,
        )

        # Portfolio Risk Constraints
        has_active_pos = len(open_positions) >= 1
        exposure_limit = current_equity * max_exposure_pct
        is_cooldown_active = (last_exit_bar_idx is not None) and ((i - last_exit_bar_idx) < cooldown_bars)

        can_allocate = (
            (not has_active_pos)
            and (pending_entry is None)
            and (not is_cooldown_active)
            and (current_open_exposure + trade_allocation <= exposure_limit + 1e-6)
            and (cash_balance >= trade_allocation)
        )

        if cascading_buy_signal and can_allocate:
            traded_in_current_episode = True
            pending_entry = (i, atr, dt)

        unrealized_value = sum(p.quantity * c for p in open_positions)
        total_equity = cash_balance + unrealized_value
        equity_curve.append((dt, total_equity))
        concurrent_positions_history.append(len(open_positions))

    # Close any open positions at end
    if open_positions:
        last_dt = df_test["datetime"].iloc[-1]
        last_close = float(df_test["close"].iloc[-1])
        for pos in open_positions:
            exit_price = last_close * (1.0 - slippage_rate)
            exit_value = pos.quantity * exit_price
            exit_fee = exit_value * fee_rate
            total_fee = pos.entry_fee + exit_fee + pos.realized_fee
            gross_pnl = (exit_value - pos.notional_value) + pos.realized_gross
            net_pnl = gross_pnl - total_fee
            orig_notional = pos.original_notional if pos.original_notional > 0 else pos.notional_value
            return_pct = (net_pnl / orig_notional) * 100.0
            cash_balance += exit_value - exit_fee

            trade_rec = TradeRecord(
                trade_id=pos.position_id,
                coin=coin_clean.upper(),
                timeframe=tf_clean,
                entry_time=str(pos.entry_time),
                exit_time=str(last_dt),
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                notional_value=orig_notional,
                exit_value=exit_value,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                return_pct=return_pct,
                entry_fee=pos.entry_fee,
                exit_fee=exit_fee,
                total_fee=total_fee,
                exit_reason="BACKTEST_END",
                bars_held=pos.bars_held,
                ai_prob=1.0,
            )
            completed_trades.append(trade_rec)
        open_positions = []

    final_balance = cash_balance
    net_profit_usdt = final_balance - initial_capital
    net_profit_pct = (net_profit_usdt / initial_capital) * 100.0

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

    equities = np.array([eq[1] for eq in equity_curve])
    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / running_max
    max_drawdown_pct = abs(float(np.min(drawdowns))) * 100.0 if len(drawdowns) > 0 else 0.0

    df_trades = pd.DataFrame([asdict(t) for t in completed_trades])
    dummy_engine = EventDrivenBacktester(coin_clean, tf_clean, initial_capital=initial_capital)
    daily_summaries = dummy_engine._compute_daily_summaries(df_trades, equity_curve)
    daily_returns = [d.daily_return_pct for d in daily_summaries]

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))
        downside_std = float(np.std([r for r in daily_returns if r < 0])) if any(r < 0 for r in daily_returns) else 1e-6
        sortino_ratio = float(np.mean(daily_returns) / (downside_std if downside_std > 0 else 1e-6) * np.sqrt(365))
    else:
        sharpe_ratio, sortino_ratio = 0.0, 0.0

    exit_breakdown = df_trades["exit_reason"].value_counts().to_dict() if not df_trades.empty else {}

    logs_dir.mkdir(parents=True, exist_ok=True)
    trades_csv_path = logs_dir / f"regime_backtest_trades_{coin_clean}_{tf_clean}.csv"
    daily_csv_path = logs_dir / f"regime_backtest_daily_{coin_clean}_{tf_clean}.csv"
    if not df_trades.empty:
        df_trades.to_csv(trades_csv_path, index=False)
    else:
        pd.DataFrame(columns=[f.name for f in fields(TradeRecord)]).to_csv(trades_csv_path, index=False)
    df_daily = pd.DataFrame([asdict(d) for d in daily_summaries])
    df_daily.to_csv(daily_csv_path, index=False)

    total_days = len(daily_summaries)
    monthly_factor = 30.417 / total_days if total_days > 0 else 1.0

    result = BacktestResult(
        coin=coin_clean.upper(),
        timeframe=tf_clean,
        threshold=1.0,
        initial_capital=initial_capital,
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
        avg_trades_per_day=(total_trades / total_days) if total_days > 0 else 0.0,
        avg_daily_pnl_usdt=float(np.mean([d.net_pnl for d in daily_summaries])) if daily_summaries else 0.0,
        avg_daily_return_pct=float(np.mean(daily_returns)) if daily_returns else 0.0,
        best_day_pnl=float(np.max([d.net_pnl for d in daily_summaries])) if daily_summaries else 0.0,
        worst_day_pnl=float(np.min([d.net_pnl for d in daily_summaries])) if daily_summaries else 0.0,
        avg_concurrent_positions=float(np.mean(concurrent_positions_history)) if concurrent_positions_history else 0.0,
        peak_concurrent_positions=int(np.max(concurrent_positions_history)) if concurrent_positions_history else 0,
        monthly_profit_usdt=net_profit_usdt * monthly_factor,
        monthly_profit_pct=net_profit_pct * monthly_factor,
        trades_log_path=trades_csv_path,
        daily_log_path=daily_csv_path,
    )
    return result, exit_breakdown


def display_regime_report(result: BacktestResult, exit_breakdown: Dict[str, int], model_title: str = "GAUSSIAN HMM") -> None:
    """Renders comprehensive summary report for Unsupervised Regime State-Transition Backtest."""
    coin_display = result.coin if result.coin.endswith("USDT") else f"{result.coin}USDT"
    title = f"UNSUPERVISED MARKET REGIME BACKTEST REPORT ({model_title} - OOS {coin_display} {result.timeframe})"
    print("\n" + "=" * 125)
    print(f"{title:^125}")
    print("=" * 125)
    print(
        f"{'Coin / TF':<15} | {'Total Trades':<13} | {'Win Rate (%)':<13} | "
        f"{'Profit Factor':<14} | {'Net PnL ($)':<12} | {'Max DD (%)':<11} | {'Sharpe':<8}"
    )
    print("-" * 125)
    print(
        f"{result.coin + '[' + result.timeframe + ']':<15} | "
        f"{result.total_trades:<13} | {result.win_rate_pct:<13.2f} | "
        f"{result.profit_factor:<14.2f} | {result.net_profit_usdt:<+12.2f} | "
        f"{result.max_drawdown_pct:<11.2f} | {result.sharpe_ratio:<8.2f}"
    )
    print("=" * 125)
    if exit_breakdown:
        print("\n--- Rincian Eksekusi Exit Posisi ---")
        for reason, cnt in exit_breakdown.items():
            pct = (cnt / result.total_trades * 100.0) if result.total_trades > 0 else 0.0
            print(f"  - {reason:<15}: {cnt:>5} trades ({pct:>6.2f}%)")
        print("-" * 50)
    print("\n")


def main() -> None:
    """Main CLI execution flow."""
    args = parse_args()

    if args.regime or args.regime_hmm:
        # Run Unsupervised Market Regime Backtest
        coin = (args.coin or "btc").lower()
        tf = (args.timeframe or "15m").lower()
        trade_alloc = args.trade_allocation if args.trade_allocation != 15.0 else (30.0 if tf == "1h" else 20.0)
        model_type = "hmm" if (args.regime_hmm or not args.regime) else "hmm"
        
        logger.info(
            f"Starting {model_type.upper()} Regime Backtest for {coin.upper()} [{tf}] (Capital: ${args.capital:.2f}, Trade: ${trade_alloc:.2f})"
        )
        res_regime, exit_breakdown = run_regime_backtest(
            coin=coin,
            timeframe=tf,
            model_type=model_type,
            initial_capital=args.capital,
            trade_allocation=trade_alloc,
            max_exposure_pct=args.max_exposure_pct,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            cooldown_bars=args.cooldown_bars,
            processed_dir=Path(args.processed_dir),
            models_dir=Path(args.models_dir),
            logs_dir=Path(args.logs_dir),
        )
        display_regime_report(res_regime, exit_breakdown, model_title="GAUSSIAN HMM")
        return

    if args.ab_test:
        # Run Comparative A/B Testing Benchmark
        coin = (args.coin or "btc").lower()
        tf = (args.timeframe or "5m").lower()
        th = args.threshold
        trade_alloc = args.trade_allocation if args.trade_allocation != 15.0 else 20.0
        
        logger.info(
            f"Starting A/B Benchmark for {coin.upper()} [{tf}] (Capital: ${args.capital:.2f}, Trade: ${trade_alloc:.2f}, Threshold: {'AUTO' if th is None else f'{th:.4f}'})"
        )
        res_ctrl, res_exp = run_ab_benchmark(
            coin=coin,
            timeframe=tf,
            initial_capital=args.capital,
            trade_allocation=trade_alloc,
            threshold=th,
            processed_dir=Path(args.processed_dir),
            models_dir=Path(args.models_dir),
            logs_dir=Path(args.logs_dir),
        )
        display_ab_test_report(res_ctrl, res_exp)
        return

    # Parse stream tuples
    parsed_streams: List[Tuple[str, str]] = []
    for s in args.streams:
        parts = s.split(":")
        if len(parts) == 2:
            parsed_streams.append((parts[0].lower(), parts[1].lower()))
        else:
            logger.warning(f"Invalid stream format: '{s}'. Expected 'coin:timeframe'.")

    if not parsed_streams:
        parsed_streams = [("btc", "5m"), ("eth", "5m"), ("sol", "15m")]

    if args.scenarios or args.unified:
        # Run Unified Multi-Asset Portfolio Simulation
        scenarios_to_run = [5.0, 10.0] if args.scenarios else [args.trade_allocation]
        unified_results: List[UnifiedBacktestResult] = []

        for alloc in scenarios_to_run:
            sc_name = f"Skenario ({alloc:.1f} USDT / Trade)"
            logger.info(f"=== Running Unified Portfolio Simulation: {sc_name} ===")
            u_engine = UnifiedPortfolioBacktester(
                streams=parsed_streams,
                initial_capital=args.capital,
                trade_allocation=alloc,
                max_exposure_pct=args.max_exposure_pct,
                fee_rate=args.fee,
                slippage_rate=args.slippage,
                cooldown_bars=args.cooldown_bars,
                threshold=args.threshold,
                processed_dir=Path(args.processed_dir),
                models_dir=Path(args.models_dir),
                logs_dir=Path(args.logs_dir),
                scenario_name=sc_name,
            )
            res = u_engine.run()
            unified_results.append(res)

        display_unified_scenario_report(unified_results)
        return

    # Single-pair mode
    pairs_to_run: List[Tuple[str, str]] = []
    if args.coin and args.timeframe:
        pairs_to_run = [(args.coin.lower(), args.timeframe.lower())]
    elif args.all or (args.coins and args.timeframes):
        for c in args.coins:
            for tf in args.timeframes:
                pairs_to_run.append((c.lower(), tf.lower()))
    else:
        logger.error("Please specify --coin and --timeframe, --all, --unified, or --scenarios.")
        sys.exit(1)

    all_results: List[BacktestResult] = []
    total_start = time.perf_counter()

    for coin, tf in pairs_to_run:
        engine = EventDrivenBacktester(
            coin=coin,
            timeframe=tf,
            initial_capital=args.capital,
            trade_allocation=args.trade_allocation,
            max_exposure_pct=args.max_exposure_pct,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            cooldown_bars=args.cooldown_bars,
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
