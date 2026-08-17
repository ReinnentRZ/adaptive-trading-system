# Python & Workspace Coding Guidelines

This document outlines the coding standards, patterns, and conventions for the Adaptive Trading System project. All agents and developers must adhere to these guidelines.

## 1. Code Style & Standards
- **PEP 8 Compliance**: Follow PEP 8 guidelines for formatting. Indent with 4 spaces.
- **Type Hinting**: All new functions, methods, and classes must include type annotations (e.g., `def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:`).
- **Naming Conventions**:
  - Classes: PascalCase (e.g., `BinanceService`)
  - Functions/Variables/Modules: snake_case (e.g., `get_klines`, `data_manager.py`)
  - Constants: UPPERCASE_SNAKE (e.g., `API_KEY`, `SYMBOL`)

## 2. Architecture & Design Patterns
- **Separation of Concerns**: Maintain clean boundaries between project directories:
  - `src/config/`: Local configuration files for APIs and settings.
  - `src/core/`: Enums, state managers, and signal data structures.
  - `src/services/`: For external API clients (e.g., BinanceClient).
  - `src/data/`: Data ingestion, WebSocket handlers, and Pandas/NumPy processing.
  - `src/indicators/`: Mathematical indicators (RSI, ADX, WaveTrend, CCI).
  - `src/strategies/`: Strategic logic (e.g., Lorentzian classifier, trend-following).
  - `src/execution/`: Order execution, dry-runs, and portfolio tracking.
- **Asynchronous Operations**: When dealing with real-time WebSockets (`BinanceWebSocket`), ensure threads and async loops are safely managed without blocking the main event loop.

## 3. Exception Handling & Logging
- **Graceful Failures**: Never allow network errors (e.g., Binance request timeouts) to crash the bot. Wrap API interactions in robust try-except blocks.
- **Structured Logging**: Use Python's `logging` module to log messages. Avoid raw `print()` statements in production code. Prefer logging with proper severity levels (`INFO`, `WARNING`, `ERROR`).
- **Secret Management**: Never hardcode API keys, secrets, or passwords. Read them from the environment or `.env` files using `python-dotenv`.

## 4. Core Software Engineering Principles
- **KISS (Keep It Simple, Stupid)**: Avoid over-engineering by choosing the simplest design that completely solves the problem.
- **DRY (Don't Repeat Yourself)**: Ensure every piece of system knowledge or logic has a single, unambiguous representation.
- **YAGNI (You Aren't Gonna Need It)**: Do not add functionality or code until it is explicitly necessary for current requirements.
- **SOLID**: Apply these five object-oriented principles to ensure your code remains modular, highly flexible, and testable.
- **Fail Fast**: Halt program execution immediately when an unexpected error or bad state occurs to make debugging easier.

## 5. Financial Precision & Numeric Handling
- **No Naive Floats**: Never use native Python `float` types for nominal balances, asset prices, or position size calculations due to floating-point precision issues. Use `decimal.Decimal` or strict rounding helpers that comply with Binance exchange filters (`tickSize` for price step and `stepSize` for quantity step).
- **Vectorized Operations**: All technical indicator and feature calculations must be fully vectorized using NumPy or Pandas. Manual iterations (such as `iterrows()` or `itertuples()`) are strictly prohibited in the hot path to ensure low latency.

## 6. Quantitative & ML Data Integrity (Strategies/Lorentzian)
- **Zero Lookahead Bias**: To prevent data leakage and lookahead bias, always ensure that historical features and signals are computed using closed candles only, or explicitly shift the data by one index (`shift(1)`) if using high/low/close data of the current timestamp.
- **Immutability of Core States**: Core models representing trade execution states (`Signal`, `Order`, `Position`, `Trade`) must be immutable. Use `@dataclass(frozen=True)` or Pydantic's `frozen=True` configuration to prevent accidental state mutation during the execution lifecycle.

## 7. State Management, Safety & Execution Resilience
- **Order Idempotency**: All orders must use a deterministic, traceable `clientOrderId` (e.g., hash of timestamp, symbol, and signal type). This prevents duplicate order execution (double execution) when API retries or socket reconnections occur.
- **Heartbeat & Circuit Breaker**: Implement fail-safe mechanisms (circuit breakers) for websocket data feeds. If the live feed remains stale or disconnected beyond a defined threshold (e.g., 10 seconds), the system must fallback to a safe state, raise alerts, or cancel outstanding/hanging open orders.

## 8. Testing, Verification & Mocking Standards
- **Indicator Parity**: Any new technical indicator must be accompanied by unit tests that verify output accuracy against established benchmark values (such as TA-Lib or TradingView Pinescript output).
- **Mock Execution**: Unit and integration tests for the ordering and execution pipeline must run against the `mock_broker`. Live API network requests to Binance are strictly forbidden in the test suite to ensure safety and test isolation.

## 9. Cybersecurity, Secret Management & Container Hardening Standards

### 9.1 API Key & Secret Governance
- **Strict Least Privilege**: All exchange API keys (e.g., Binance) must not have withdrawal permissions enabled. Only enable Read-Only and Spot/Margin trading permissions based on the environment's requirements.
- **IP Access Restriction**: Production exchange API keys must be locked to the specific VPS/IP address whitelisting where the bot runs.
- **Zero Secrets in Source/Artifacts**: Hardcoded secrets, API keys, or JWT tokens are strictly prohibited in the codebase, docker image layers, and commit history. Inject secrets via environment variables at runtime (`.env` file must be ignored via `.gitignore` and `.dockerignore`).
- **Log Sanitization**: Raw authentication headers, HMAC signatures, passwords, and secret keys must be sanitized and never written to text logs, standard output, or SQLite telemetry databases.

### 9.2 Container Security & Hardening
- **Non-Root Execution**: Docker containers must run as a dedicated non-root user (e.g., `appuser` with `UID 1000`) instead of the default `root` user to limit container-escape vectors.
- **Minimal Base Image & Scoped Mounts**: Always use minimal base images (such as `slim` or `alpine` Python variants). Keep volume mounts scoped to the absolute minimum required directories (e.g., `./logs:/app/logs`). Do not mount system sockets like `/var/run/docker.sock` or parent host folders.
- **No Inbound Port Exposure**: Since this trading engine works purely outbound (polling REST and subscribing to WebSockets), do not expose any inbound ports (e.g., the `ports:` block in `docker-compose.yml`) unless an authenticated and secure web dashboard is explicitly required.

### 9.3 Network Integrity & Dependency Hygiene
- **Strict TLS/SSL Handshake Verification**: All outbound REST and WebSocket connections must enforce full SSL certificate verification. Disabling verification (e.g., `verify=False` or bypassing SSL/TLS certificate chains) is strictly prohibited. Use standard CA packages (e.g., Python's `certifi` CA bundle) to validate handshakes.
- **Vulnerability Auditing**: Perform regular dependency audits using security tools like `pip-audit` or `safety` to proactively identify and patch Common Vulnerabilities and Exposures (CVEs) in Python packages.

### 9.4 Execution Safety & Financial Circuit Breaker
- **Hard Drawdown Limits**: The execution engine must implement an automated financial circuit breaker (kill-switch). This safety mechanism must pause all trading activities and cancel open orders if daily drawdown thresholds are reached, extreme network latency anomalies occur, or major exchange connectivity failure is detected.

## 10. Security-First Development & Defensive Coding Mindset

### 10.1 Mandatory Threat Modeling in Every Task
- **Identify Attack Vectors**: Before writing or modifying any module (especially data ingestion, execution, and config), identify potential threat vectors: credentials exposure, data leakage, denial-of-service (buffer overflow/memory leak), and state manipulation.

### 10.2 Input Validation, Sanitization & Type Safety
- **Untrusted Input**: Treat all external inputs (Binance WebSocket payloads, REST API responses, local config files, and CLI parameters) as untrusted.
- **Runtime Schema Validation**: Validate input schemas using runtime parsers (such as `Pydantic` or strict `dataclasses`) before passing data to indicators/strategies.
- **Anomaly Rejection**: Instantly reject and drop data packets showing structural anomalies, stale timestamps (to prevent replay attacks or out-of-order latency), or logically out-of-bounds numbers (e.g., price <= 0, NaN, or infinity).

### 10.3 Defensive Resource Management (Anti-DoS & Crash Resistance)
- **Prevent Memory Exhaustion**: Set strict limits on buffer and cache capacities. Use bounded collections (such as `collections.deque(maxlen=N)` for historical candles) to limit memory growth.
- **Explicit Timeouts**: Apply explicit timeouts on all HTTP/REST requests and blocking socket operations. Never permit infinite blocking without active heartbeats.

### 10.4 Fail-Closed & Secure Defaults
- **Primacy of Fail-Closed**: All components must operate on the "fail-closed" principle. Upon authentication failures, checksum/integrity verification errors, or invalid state flags, the bot must immediately halt trading and cancel active orders instead of defaulting to a passive "safe" state.
- **Secure Defaults**: All default options must represent the most restrictive configuration (e.g., `dry_run = True` is default, and standard logger levels never expose raw API payloads).

### 10.5 Code Review & Automated Security Checks
- **Static Security Testing (SAST)**: All code changes must pass local security scans (such as running `bandit -r src/` or similar static analysis tools) to verify the absence of typical Python security flaws (e.g., usage of `eval()`, using `assert` for production logic checks, or hardcoding credentials).
