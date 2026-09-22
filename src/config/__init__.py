from dataclasses import dataclass, field
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from src.config.strategy import RegimeFunnelConfig


# ==============================================================================
# ENVIRONMENT VARIABLE HELPERS
# ==============================================================================
def _get_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


# ==============================================================================
# ZONE 1: QUICK USER SETTINGS (Sering Diubah)
# ==============================================================================
# Pengaturan dasar trading yang sering disesuaikan oleh pengguna.

# Simbol pasangan aset kripto yang diperdagangkan (Huruf kapital, tanpa spasi).
# Rekomendasi: Pasangan aktif dengan likuiditas tinggi seperti "BTCUSDT" atau "SOLUSDT".
USER_SYMBOL = _get_env_str("ACTIVE_PAIR", _get_env_str("TRADING_SYMBOL", "BTCUSDT")).upper()

# Interval timeframe candle untuk analisis sinyal.
# Pilihan Binance valid: "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w".
USER_INTERVAL = _get_env_str("ACTIVE_TIMEFRAME", _get_env_str("TRADING_INTERVAL", "5m")).lower()

# Nominal alokasi dana / margin per transaksi dalam USD/USDT.
# Rekomendasi: Mulai dari nilai minimum yang diijinkan Binance (misal 5.0 atau 10.0).
USER_TRADE_QUANTITY_USDT = _get_env_float("TRADE_AMOUNT", _get_env_float("TRADE_QUANTITY_USDT", 5.0))

# Mode paper trading / simulasi vs akun riil (Live Trading).
# BOT_MODE="paper" -> dry_run=True, BOT_MODE="live" -> dry_run=False.
_bot_mode_env = _get_env_str("BOT_MODE", "").lower()
if _bot_mode_env:
    USER_DRY_RUN = (_bot_mode_env != "live")
else:
    USER_DRY_RUN = _get_env_bool("DRY_RUN", True)

# Toggle manajemen risiko dinamis berbasis ATR (Average True Range).
# True = Level SL/TP dinamis mengikuti volatilitas pasar, False = Menggunakan fixed % profit target.
USER_USE_ATR_RISK_MANAGEMENT = _get_env_bool("USE_ATR_RISK_MANAGEMENT", True)


# ==============================================================================
# ZONE 2: ADVANCED QUANT & SYSTEM SETTINGS (Jarang Diubah)
# ==============================================================================
# Konfigurasi kuantitatif tingkat lanjut, ML model, dan parameter jaringan.

# --- MODEL ML & LORENTZIAN CLASSIFICATION ---
# Batas ambang probabilitas klasifikasi Meta-Labeling LightGBM (default 0.44 untuk BTC 5m)
QUANT_AI_THRESHOLD = _get_env_float("AI_THRESHOLD", 0.44)
# Batas persentase keyakinan sinyal sebelum eksekusi (50.0 - 100.0, default 70.0).
QUANT_CONFIDENCE_THRESHOLD = _get_env_float("ML_CONFIDENCE_THRESHOLD", 70.0)
# Jumlah tetangga terdekat dalam pencarian Lorentzian classifier (k) (3 - 25, default 8).
QUANT_NEIGHBORS_COUNT = _get_env_int("ML_NEIGHBORS_COUNT", 8)
# Kapasitas pencarian histori data candle untuk model klasifikasi (default 2000).
QUANT_MAX_BARS_BACK = _get_env_int("ML_MAX_BARS_BACK", 2000)
# Jumlah candle di masa depan untuk penentuan label arah pergerakan (default 4).
QUANT_LABEL_HORIZON = _get_env_int("ML_LABEL_HORIZON", 4)

# --- FILTER REZIM PASAR & VOLATILITAS ---
# Toggle penyaringan transaksi berdasarkan tingkat volatilitas.
QUANT_USE_VOLATILITY_FILTER = _get_env_bool("ML_USE_VOLATILITY_FILTER", True)
QUANT_VOLATILITY_MIN_LENGTH = _get_env_int("ML_VOLATILITY_MIN_LENGTH", 1)
QUANT_VOLATILITY_MAX_LENGTH = _get_env_int("ML_VOLATILITY_MAX_LENGTH", 10)
# Toggle penyaringan transaksi berdasarkan rezim tren pasar.
QUANT_USE_REGIME_FILTER = _get_env_bool("ML_USE_REGIME_FILTER", True)
# Batas ambang penyaringan tren rezim pasar (default -0.1).
QUANT_REGIME_THRESHOLD = _get_env_float("ML_REGIME_THRESHOLD", -0.1)

# --- TREN FILTER (EMA/SMA & ADX) ---
# Toggle penyaringan arah sinyal berbasis ADX (Average Directional Index).
QUANT_USE_ADX_FILTER = _get_env_bool("ML_USE_ADX_FILTER", False)
QUANT_ADX_LENGTH = _get_env_int("ML_ADX_LENGTH", 14)
# Batas tren kuat ADX sebelum sinyal diijinkan lewat (default 20).
QUANT_ADX_THRESHOLD = _get_env_int("ML_ADX_THRESHOLD", 20)
# Toggle filter tren jangka panjang EMA.
QUANT_USE_EMA_FILTER = _get_env_bool("ML_USE_EMA_FILTER", False)
QUANT_EMA_PERIOD = _get_env_int("ML_EMA_PERIOD", 200)
# Toggle filter tren jangka panjang SMA.
QUANT_USE_SMA_FILTER = _get_env_bool("ML_USE_SMA_FILTER", False)
QUANT_SMA_PERIOD = _get_env_int("ML_SMA_PERIOD", 200)

# --- KERNEL REGRESSION (Nadaraya-Watson) ---
QUANT_USE_KERNEL_FILTER = _get_env_bool("ML_USE_KERNEL_FILTER", True)
QUANT_USE_KERNEL_SMOOTHING = _get_env_bool("ML_USE_KERNEL_SMOOTHING", False)
QUANT_KERNEL_LOOKBACK = _get_env_int("ML_KERNEL_LOOKBACK", 8)
QUANT_KERNEL_RELATIVE_WEIGHT = _get_env_float("ML_KERNEL_RELATIVE_WEIGHT", 8.0)
QUANT_KERNEL_REGRESSION_LEVEL = _get_env_int("ML_KERNEL_REGRESSION_LEVEL", 25)
QUANT_KERNEL_LAG = _get_env_int("ML_KERNEL_LAG", 2)

# --- PROTEKSI RISIKO & MANAJEMEN PORTOFOLIO ---
# Periode perhitungan Average True Range (default 14).
RISK_ATR_PERIOD = _get_env_int("RISK_ATR_PERIOD", 14)
# Jarak pengaman Stop Loss dinamis (multiplied by ATR) (default 1.0).
RISK_ATR_SL_MULTIPLIER = _get_env_float("RISK_ATR_SL_MULTIPLIER", 1.0)
# Jarak target profit dinamis (multiplied by ATR) (default 2.0).
RISK_ATR_TP_MULTIPLIER = _get_env_float("RISK_ATR_TP_MULTIPLIER", 2.0)
# Time barrier batas maksimum holding candle (default 12 candle).
RISK_TIME_BARRIER_BARS = _get_env_int("TIME_BARRIER_BARS", 12)
# Target Take Profit persen statis jika filter ATR dimatikan (default 3.0%).
RISK_TAKE_PROFIT_PCT = _get_env_float("RISK_TAKE_PROFIT_PCT", 3.0)
# Pengaman Stop Loss persen statis jika filter ATR dimatikan (default 1.0%).
RISK_STOP_LOSS_PCT = _get_env_float("RISK_STOP_LOSS_PCT", 1.0)
# Potongan biaya komisi per transaksi (default 0.075% untuk Binance spot).
RISK_COMMISSION_FEE_PCT = _get_env_float("RISK_COMMISSION_FEE_PCT", 0.075)
# Saldo modal awal akun simulasi / mock portofolio (default 100.0 USDT).
RISK_INITIAL_BALANCE = _get_env_float("INITIAL_CAPITAL", _get_env_float("RISK_INITIAL_BALANCE", 100.0))

# --- SPESIFIKASI PERIODE INDIKATOR TEKNIKAL ---
IND_RSI_PERIOD = _get_env_int("IND_RSI_PERIOD", 14)
IND_RSI_MA_PERIOD = _get_env_int("IND_RSI_MA_PERIOD", 14)
IND_CCI_PERIOD = _get_env_int("IND_CCI_PERIOD", 20)
IND_CCI_SMOOTHING_PERIOD = _get_env_int("IND_CCI_SMOOTHING_PERIOD", 14)
IND_ADX_PERIOD = _get_env_int("IND_ADX_PERIOD", 14)
IND_ADX_SMOOTHING_PERIOD = _get_env_int("IND_ADX_SMOOTHING_PERIOD", 14)
IND_WT_CHANNEL_LENGTH = _get_env_int("IND_WT_CHANNEL_LENGTH", 10)
IND_WT_AVERAGE_LENGTH = _get_env_int("IND_WT_AVERAGE_LENGTH", 21)
IND_WT_SMA_LENGTH = _get_env_int("IND_WT_SMA_LENGTH", 4)

# --- NETWORK & WEBSOCKET ENGINE ---
SYS_BINANCE_WS_BASE_URL = _get_env_str("SYS_BINANCE_WS_BASE_URL", "wss://stream.binance.com:9443/ws")
SYS_PING_INTERVAL = _get_env_int("SYS_PING_INTERVAL", 30)
SYS_PING_TIMEOUT = _get_env_int("SYS_PING_TIMEOUT", 10)
SYS_RECONNECT_DELAY_SECONDS = _get_env_int("SYS_RECONNECT_DELAY_SECONDS", 5)

# --- TELEMETRY & LOGGING PATHS ---
SYS_LOG_FILE_PATH = _get_env_str("SYS_LOG_FILE_PATH", "logs/logs.txt")
SYS_TELEMETRY_DB_PATH = _get_env_str("SYS_TELEMETRY_DB_PATH", "logs/trading_telemetry.db")
SYS_ENABLE_CONSOLE_LOG = _get_env_bool("SYS_ENABLE_CONSOLE_LOG", True)


# ==============================================================================
# TYPE-SAFE DATACLASS REPRESENTATION & VALIDATION
# ==============================================================================
@dataclass(frozen=True)
class MarketConfig:
    symbol: str = USER_SYMBOL
    interval: str = USER_INTERVAL
    kline_limit: int = 2000

    def __post_init__(self) -> None:
        # Validasi format simbol Binance (Harus kapital, tanpa spasi)
        if not isinstance(self.symbol, str) or not self.symbol.isupper() or " " in self.symbol:
            raise ValueError(
                f"symbol must be an uppercase string without spaces (e.g. 'SOLUSDT'), got: '{self.symbol}'"
            )

    @property
    def base_asset(self) -> str:
        symbol_upper = self.symbol.upper()
        for quote in ["USDT", "BUSD", "BTC", "ETH"]:
            if symbol_upper.endswith(quote):
                return symbol_upper[:-len(quote)]
        return symbol_upper[:-4]

    @property
    def quote_asset(self) -> str:
        symbol_upper = self.symbol.upper()
        for quote in ["USDT", "BUSD", "BTC", "ETH"]:
            if symbol_upper.endswith(quote):
                return quote
        return "USDT"

    @property
    def ws_stream_name(self) -> str:
        return f"{self.symbol_lower}@kline_{self.interval}"

    @property
    def symbol_lower(self) -> str:
        return self.symbol.lower()


@dataclass(frozen=True)
class TradingConfig:
    initial_balance: float = RISK_INITIAL_BALANCE
    take_profit_pct: float = RISK_TAKE_PROFIT_PCT
    stop_loss_pct: float = RISK_STOP_LOSS_PCT
    commission_fee_pct: float = RISK_COMMISSION_FEE_PCT
    trade_quantity_usdt: float = USER_TRADE_QUANTITY_USDT
    atr_period: int = RISK_ATR_PERIOD
    atr_sl_multiplier: float = RISK_ATR_SL_MULTIPLIER
    atr_tp_multiplier: float = RISK_ATR_TP_MULTIPLIER
    time_barrier_bars: int = RISK_TIME_BARRIER_BARS
    use_atr_risk_management: bool = USER_USE_ATR_RISK_MANAGEMENT
    dry_run: bool = USER_DRY_RUN

    def __post_init__(self) -> None:
        # Validasi nominal margin perdagangan
        if self.trade_quantity_usdt <= 0:
            raise ValueError(f"trade_quantity_usdt must be positive, got {self.trade_quantity_usdt}")


@dataclass(frozen=True)
class IndicatorConfig:
    rsi_period: int = IND_RSI_PERIOD
    rsi_ma_period: int = IND_RSI_MA_PERIOD
    cci_period: int = IND_CCI_PERIOD
    cci_smoothing_period: int = IND_CCI_SMOOTHING_PERIOD
    adx_period: int = IND_ADX_PERIOD
    adx_smoothing_period: int = IND_ADX_SMOOTHING_PERIOD
    wt_channel_length: int = IND_WT_CHANNEL_LENGTH
    wt_average_length: int = IND_WT_AVERAGE_LENGTH
    wt_sma_length: int = IND_WT_SMA_LENGTH


@dataclass(frozen=True)
class LoggingConfig:
    log_file_path: str = SYS_LOG_FILE_PATH
    telemetry_db_path: str = SYS_TELEMETRY_DB_PATH
    enable_console_log: bool = SYS_ENABLE_CONSOLE_LOG


@dataclass(frozen=True)
class APIConfig:
    api_key: str = field(
        default_factory=lambda: os.getenv("BINANCE_API_KEY", "")
    )
    api_secret: str = field(
        default_factory=lambda: os.getenv("BINANCE_SECRET_KEY", "")
    )

    def __repr__(self) -> str:
        # Masking API key dan Secret key untuk menjaga privasi pada log files
        key = self.api_key
        if len(key) > 8:
            masked_key = f"{key[:4]}***{key[-4:]}"
        elif key:
            masked_key = f"{key[:2]}***"
        else:
            masked_key = "None"
        return f"APIConfig(api_key='{masked_key}', api_secret='***REDACTED***')"


@dataclass(frozen=True)
class StrategyConfig:
    ai_threshold: float = QUANT_AI_THRESHOLD
    confidence_threshold: float = QUANT_CONFIDENCE_THRESHOLD
    neighbors_count: int = QUANT_NEIGHBORS_COUNT
    max_bars_back: int = QUANT_MAX_BARS_BACK
    label_horizon: int = QUANT_LABEL_HORIZON
    time_barrier_bars: int = RISK_TIME_BARRIER_BARS
    
    use_volatility_filter: bool = QUANT_USE_VOLATILITY_FILTER
    volatility_min_length: int = QUANT_VOLATILITY_MIN_LENGTH
    volatility_max_length: int = QUANT_VOLATILITY_MAX_LENGTH
    
    use_regime_filter: bool = QUANT_USE_REGIME_FILTER
    regime_threshold: float = QUANT_REGIME_THRESHOLD
    
    use_adx_filter: bool = QUANT_USE_ADX_FILTER
    adx_length: int = QUANT_ADX_LENGTH
    adx_threshold: int = QUANT_ADX_THRESHOLD
    
    use_ema_filter: bool = QUANT_USE_EMA_FILTER
    ema_period: int = QUANT_EMA_PERIOD
    use_sma_filter: bool = QUANT_USE_SMA_FILTER
    sma_period: int = QUANT_SMA_PERIOD
    
    use_kernel_filter: bool = QUANT_USE_KERNEL_FILTER
    use_kernel_smoothing: bool = QUANT_USE_KERNEL_SMOOTHING
    kernel_lookback: int = QUANT_KERNEL_LOOKBACK
    kernel_relative_weight: float = QUANT_KERNEL_RELATIVE_WEIGHT
    kernel_regression_level: int = QUANT_KERNEL_REGRESSION_LEVEL
    kernel_lag: int = QUANT_KERNEL_LAG

    def __post_init__(self) -> None:
        # Validasi batas keyakinan sinyal Lorentzian
        if not (0.0 <= self.confidence_threshold <= 100.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 100.0, got {self.confidence_threshold}"
            )
        if not (0.0 <= self.ai_threshold <= 1.0):
            raise ValueError(
                f"ai_threshold must be between 0.0 and 1.0, got {self.ai_threshold}"
            )


@dataclass(frozen=True)
class WebSocketConfig:
    binance_ws_base_url: str = SYS_BINANCE_WS_BASE_URL
    ping_interval: int = SYS_PING_INTERVAL
    ping_timeout: int = SYS_PING_TIMEOUT
    reconnect_delay_seconds: int = SYS_RECONNECT_DELAY_SECONDS


@dataclass(frozen=True)
class AppConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    regime_funnel: RegimeFunnelConfig = field(default_factory=RegimeFunnelConfig)

    def __repr__(self) -> str:
        return (
            f"AppConfig(\n"
            f"  market={self.market},\n"
            f"  trading={self.trading},\n"
            f"  indicators={self.indicators},\n"
            f"  logging={self.logging},\n"
            f"  api={self.api},\n"
            f"  strategy={self.strategy},\n"
            f"  websocket={self.websocket},\n"
            f"  regime_funnel={self.regime_funnel}\n"
            f")"
        )


# ==============================================================================
# SINGLETON EXPORTS & BACKWARD COMPATIBILITY
# ==============================================================================
# Instansiasi Root Config
config = AppConfig()

# Expose global aliases agar tidak mengganggu import berkas konsumen lama
MARKET = config.market
TRADING = config.trading
INDICATORS = config.indicators
LOGGING = config.logging
WEBSOCKET = config.websocket
STRATEGY = config.strategy
REGIME_FUNNEL = config.regime_funnel
API_KEY = config.api.api_key
API_SECRET = config.api.api_secret
AI_THRESHOLD = config.strategy.ai_threshold

__all__ = [
    "config",
    "AppConfig",
    "MarketConfig",
    "TradingConfig",
    "IndicatorConfig",
    "LoggingConfig",
    "APIConfig",
    "StrategyConfig",
    "WebSocketConfig",
    "RegimeFunnelConfig",
    "MARKET",
    "TRADING",
    "INDICATORS",
    "LOGGING",
    "WEBSOCKET",
    "STRATEGY",
    "REGIME_FUNNEL",
    "API_KEY",
    "API_SECRET",
    "AI_THRESHOLD",
]
