# Welcome to the Adaptive Trading System Agent Hub

This repository is optimized for autonomous and collaborative AI development. This document guides visiting agents on how to navigate the codebase, understand the architecture, and follow project guidelines.

## 📁 Repository Structure

Here is a quick overview of the key folders in the workspace:
- **`src/config/`**: Local configuration files for APIs and settings.
- **`src/core/`**: Enums, state managers, and signal data structures.
- **`src/data/`**: Historical and real-time market data ingestion and WebSockets.
- **`src/execution/`**: Real and mock brokers, order validation, and position trackers.
- **`src/indicators/`**: Implementations of technical indicators (ADX, RSI, WaveTrend, CCI).
- **`src/services/`**: API wrapper services (e.g., BinanceClient wrapper).
- **`src/strategies/`**: Algorithmic trading strategies (e.g., Lorentzian classification).
- **`main.py`**: The entry point to start the live trading bot.

---

## 🛠️ Active Rules & Guidelines

When working in this repository, you **MUST** load and adhere to the following rules:
- **Coding Guidelines**: See [.agents/rules/coding_guidelines.md](file:///home/rei/reinn/projects/adaptive-trading-system/.agents/rules/coding_guidelines.md) for style conventions, clean architecture boundaries, and logging practices.
- **Trading Safety**: See [.agents/rules/trading_safety.md](file:///home/rei/reinn/projects/adaptive-trading-system/.agents/rules/trading_safety.md) for strict credential handling, rate-limiting rules, and order safety.

---

## 💡 Custom Skills & Runbooks

The following custom workflows are available for this workspace:
- **Backtesting Strategies**: Use the [backtest-strategy](file:///home/rei/reinn/projects/adaptive-trading-system/.agents/skills/backtest/SKILL.md) skill to simulate strategies on historical data.
- **Running the Bot**: Use the [run-bot](file:///home/rei/reinn/projects/adaptive-trading-system/.agents/skills/run-bot/SKILL.md) skill to start, verify, and monitor the bot's execution logs.
