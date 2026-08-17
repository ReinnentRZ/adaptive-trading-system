---
name: backtest-strategy
description: >-
  Use this skill when the user asks to backtest a strategy, evaluate historical trading performance, or optimize hyperparameters of a technical indicator.
---

# Backtesting Strategies

This runbook guides you through historical performance evaluation and indicator validation.

## 1. Prerequisites
- Historical Klines data stored locally or fetched via `BinanceService.get_klines()`.
- Indicators (`rsi`, `adx`, `wt`, `cci`) imported from `src/indicators/`.
- Strategy rules imported from `src/strategies/`.

## 2. Standard Backtest Workflow
Follow these steps to run a backtest:

1. **Load Historical Data**:
   Ensure you have sufficient historical klines. A default limit of 2000 points is standard for initial tests:
   ```python
   from src.services.binance_client import BinanceService
   service = BinanceService()
   klines = service.get_klines(symbol="BTCUSDT", interval="1h", limit=1000)
   ```
2. **Calculate Indicators & Signals**:
   Pass the dataframe to your indicators, then evaluate strategy signals:
   ```python
   from src.strategies.lorentzian import LorentzianStrategy
   strategy = LorentzianStrategy(...)
   signals = strategy.generate_signals(df)
   ```
3. **Simulate Broker Actions**:
   Use `src/execution/mock_broker.py` or a custom backtest loop to track balances, entries, exits, and fees:
   - Account for maker/taker fees (Binance default is 0.1%).
   - Monitor drawdown, Win Rate, and Profit Factor.

## 3. Performance Metrics
Evaluate backtest results using the following standards:
- **Total Return**: Net percentage change in portfolio value.
- **Max Drawdown (MDD)**: Maximum peak-to-trough percentage drop.
- **Profit Factor**: Gross Profit divided by Gross Loss (targets > 1.5).
- **Win Rate**: Number of winning trades divided by total trades.
