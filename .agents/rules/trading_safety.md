# Trading Safety & Risk Management Rules

This document outlines safety protocols, risk management, and API practices to avoid financial loss, rate limit bans, or security compromises.

## 1. API Credentials & Authentication
- **No Hardcoding**: Under no circumstances should `API_KEY` or `API_SECRET` be committed to version control or hardcoded in codebase files.
- **Environment Variables**: Always load keys from environment variables using a `.env` file (which must be ignored in `.gitignore`).
- **Read-Only / Testnet by Default**: When developing and testing new features, prioritize using the Binance Testnet. Real trading accounts should only be enabled under explicit user configuration.

## 2. API Rate Limiting & Performance
- **Weight Awareness**: Binance imposes strict rate limits based on request weight. Limit calling expensive endpoints (e.g., fetching 2000 klines frequently) to prevent IP bans.
- **Caching**: Reuse fetched historical data and cache intermediate state instead of polling APIs.
- **WebSocket Throttling**: Manage WebSocket payloads appropriately. Do not perform heavy CPU operations on the main WebSocket thread, as it can cause buffers to overflow and disconnect.

## 3. Order Execution Safety
- **Dry-run Mode**: Always support a `dry_run` or paper trading mode in execution modules.
- **Order Constraints**: Respect Binance rules for minimum order sizes, tick sizes, and step sizes before sending order requests. Validate order boundaries in `src/execution/`.
- **Stop Loss & Position Sizing**: Ensure any strategy implementation enforces a strict maximum position size and automatic stop loss controls.
