from dataclasses import dataclass

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
LOG_FILE_PATH = "logs/logs.txt"
TELEMETRY_DB_PATH = "logs/trading_telemetry.db"
ENABLE_CONSOLE_LOG = True

# ==============================================================================
# INTERNAL
# ==============================================================================
@dataclass(frozen=True)
class LoggingSettings:
    log_file_path: str
    telemetry_db_path: str
    enable_console_log: bool

LOGGING = LoggingSettings(
    log_file_path=LOG_FILE_PATH,
    telemetry_db_path=TELEMETRY_DB_PATH,
    enable_console_log=ENABLE_CONSOLE_LOG,
)
