---
name: run-bot
description: >-
  Use this skill when the user asks to start the trading bot, monitor websocket connection status, or inspect live market data feeds.
---

# Running and Monitoring the Trading Bot

This runbook guides you on running the live bot and troubleshooting connection states.

## 1. Startup Checklist
Before starting the bot, ensure:
1. **API Keys**: Make sure the `.env` file exists and contains valid `BINANCE_API_KEY` and `BINANCE_API_SECRET`.
2. **Virtual Environment**: Use the virtual environment `.venv` to run scripts:
   ```bash
   source .venv/bin/activate
   ```
3. **Connectivity**: Validate API connection using the ping tool:
   ```bash
   python3 -c "from src.services.binance_client import BinanceService; print(BinanceService().get_ping_latency())"
   ```

## 2. Launching the Bot
Start the main application loop:
```bash
python3 main.py
```
This initializes the historical data, starts the WebSocket stream for real-time klines, and processes live strategy signals.

## 3. Monitoring & Verification
- **Websocket Logs**: Verify kline data starts streaming. Logs will show incoming ticks and trade signals.
- **Connection Health**: If the connection drops, check if the websocket client library is throwing network or frame exceptions.
- **Process Status**: To run the bot in the background and track its progress, use a background runner (like nohup or tmux) or run it as a background task.
