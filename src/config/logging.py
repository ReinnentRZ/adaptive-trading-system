from dataclasses import dataclass

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
LOG_FILE_PATH = "logs/logs.txt"
ENABLE_CONSOLE_LOG = True

# ==============================================================================
# INTERNAL
# ==============================================================================
@dataclass(frozen=True)
class LoggingSettings:
    log_file_path: str
    enable_console_log: bool

LOGGING = LoggingSettings(
    log_file_path=LOG_FILE_PATH,
    enable_console_log=ENABLE_CONSOLE_LOG,
)
