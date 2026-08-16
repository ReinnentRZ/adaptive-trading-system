import os
import json
import sqlite3
import contextlib
import logging
import psutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from src.config.logging import LOGGING


@dataclass(frozen=True)
class CandleTelemetry:
    """
    Data model representing a snapshot of market data, technical indicators,
    and machine learning predictions for a closed candle.
    """
    timestamp: int
    datetime_utc: str
    datetime_wib: str
    close_price: float
    rsi: Optional[float]
    rsi_smoothing: Optional[float]
    adx: Optional[float]
    adx_smoothing: Optional[float]
    cci: Optional[float]
    cci_smoothing: Optional[float]
    wt1: Optional[float]
    wt2: Optional[float]
    neighbours: List[int]
    raw_prediction: int
    signal_name: str
    ram_usage_mb: float


class TelemetryLogger:
    """
    SQLite Telemetry Logger to persist structured quantitative and ML features
    for model analysis and offline dataset extraction.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure target directories exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry (
                        timestamp INTEGER PRIMARY KEY,
                        datetime_utc TEXT NOT NULL,
                        datetime_wib TEXT,
                        close_price REAL NOT NULL,
                        rsi REAL,
                        rsi_smoothing REAL,
                        adx REAL,
                        adx_smoothing REAL,
                        cci REAL,
                        cci_smoothing REAL,
                        wt1 REAL,
                        wt2 REAL,
                        neighbours TEXT,
                        raw_prediction INTEGER,
                        signal_name TEXT,
                        ram_usage_mb REAL
                    )
                """)
                # Migrasi otomatis jika kolom datetime_wib belum ada
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(telemetry);")
                columns = [info[1] for info in cursor.fetchall()]
                if "datetime_wib" not in columns:
                    conn.execute("ALTER TABLE telemetry ADD COLUMN datetime_wib TEXT;")

    def save_telemetry(self, data: CandleTelemetry) -> None:
        neighbours_json = json.dumps(data.neighbours)
        query = """
            INSERT OR REPLACE INTO telemetry (
                timestamp, datetime_utc, datetime_wib, close_price,
                rsi, rsi_smoothing, adx, adx_smoothing,
                cci, cci_smoothing, wt1, wt2,
                neighbours, raw_prediction, signal_name, ram_usage_mb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(query, (
                    data.timestamp,
                    data.datetime_utc,
                    data.datetime_wib,
                    data.close_price,
                    data.rsi,
                    data.rsi_smoothing,
                    data.adx,
                    data.adx_smoothing,
                    data.cci,
                    data.cci_smoothing,
                    data.wt1,
                    data.wt2,
                    neighbours_json,
                    data.raw_prediction,
                    data.signal_name,
                    data.ram_usage_mb
                ))


# Initialize Operational Logger
logger = logging.getLogger("trading_system")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Ensure log directories exist
    log_dir = os.path.dirname(LOGGING.log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Force logs to be displayed in WIB timezone (UTC+7)
    wib_tz = timezone(timedelta(hours=7))
    formatter.converter = lambda ts: datetime.fromtimestamp(ts, tz=wib_tz).timetuple()

    # File Handler
    file_handler = logging.FileHandler(LOGGING.log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    if LOGGING.enable_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

# Initialize Telemetry Logger Singleton
_telemetry_logger = TelemetryLogger(LOGGING.telemetry_db_path)


def log_close(
    data_kline: dict,
    time_close: int,
    rsi_value: Optional[float],
    rsi_smoothing: Optional[float],
    adx_value: Optional[float],
    adx_smoothing: Optional[float],
    cci_value: Optional[float],
    cci_smoothing: Optional[float],
    wt1_value: Optional[float],
    wt2_value: Optional[float],
    array_tetangga: List[int],
    raw_prediction: int,
    signal_name: str
) -> None:
    """
    Dual-Sink Logging Entrypoint:
    1. Operational Logging: Outputs concise status messages to console & log file.
    2. SQLite Telemetry Store: Persists full mathematical/ML feature state.
    """
    # 1. RAM Usage check
    process = psutil.Process(os.getpid())
    ram_usage_mb = float(process.memory_info().rss / (1024 * 1024))

    # 2. Construct Telemetry Object
    wib_tz = timezone(timedelta(hours=7))
    telemetry = CandleTelemetry(
        timestamp=int(time_close),
        datetime_utc=datetime.fromtimestamp(time_close / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        datetime_wib=datetime.fromtimestamp(time_close / 1000, tz=wib_tz).strftime('%Y-%m-%d %H:%M:%S'),
        close_price=float(data_kline["close"]),
        rsi=float(rsi_value) if rsi_value is not None else None,
        rsi_smoothing=float(rsi_smoothing) if rsi_smoothing is not None else None,
        adx=float(adx_value) if adx_value is not None else None,
        adx_smoothing=float(adx_smoothing) if adx_smoothing is not None else None,
        cci=float(cci_value) if cci_value is not None else None,
        cci_smoothing=float(cci_smoothing) if cci_smoothing is not None else None,
        wt1=float(wt1_value) if wt1_value is not None else None,
        wt2=float(wt2_value) if wt2_value is not None else None,
        neighbours=list(array_tetangga),
        raw_prediction=int(raw_prediction),
        signal_name=str(signal_name),
        ram_usage_mb=ram_usage_mb
    )

    # 3. Save to SQLite
    try:
        _telemetry_logger.save_telemetry(telemetry)
    except Exception as e:
        logger.error(f"Failed to persist telemetry data to SQLite: {e}")

    # 4. Format Operational Log Message
    vote_text = f"+{raw_prediction}" if raw_prediction > 0 else f"{raw_prediction}"
    if signal_name == "LONG":
        signal_text = "BUY"
    elif signal_name == "SHORT":
        signal_text = "SELL"
    else:
        signal_text = "HOLD"

    operational_msg = (
        f"Close: {telemetry.close_price:<9.2f} | "
        f"VOTE: {vote_text:<4} | "
        f"SIGNAL: {signal_text:<4} | "
        f"RAM: {ram_usage_mb:.1f} MB"
    )
    logger.info(operational_msg)