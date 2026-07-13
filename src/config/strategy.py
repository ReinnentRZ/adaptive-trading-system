from dataclasses import dataclass

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
# General KNN Settings
CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence in percent to execute a trade
NEIGHBORS_COUNT = 8          # Number of neighbors to consider (k)
MAX_BARS_BACK = 2000         # Max historical bars back to search for neighbors
LABEL_HORIZON = 4            # Bars ahead to classify direction and neighbor gap

# Volatility Filter Settings
USE_VOLATILITY_FILTER = True
VOLATILITY_MIN_LENGTH = 1
VOLATILITY_MAX_LENGTH = 10

# Regime Filter Settings
USE_REGIME_FILTER = True
REGIME_THRESHOLD = -0.1

# ADX Filter Settings
USE_ADX_FILTER = False
ADX_LENGTH = 14
ADX_THRESHOLD = 20

# EMA & SMA Trend Confirmation Settings
USE_EMA_FILTER = False
EMA_PERIOD = 200
USE_SMA_FILTER = False
SMA_PERIOD = 200

# Kernel Regression Settings
USE_KERNEL_FILTER = True
USE_KERNEL_SMOOTHING = False
KERNEL_LOOKBACK = 8
KERNEL_RELATIVE_WEIGHT = 8.0
KERNEL_REGRESSION_LEVEL = 25
KERNEL_LAG = 2

# ==============================================================================
# INTERNAL
# ==============================================================================
@dataclass(frozen=True)
class StrategySettings:
    confidence_threshold: float
    neighbors_count: int
    max_bars_back: int
    label_horizon: int
    
    use_volatility_filter: bool
    volatility_min_length: int
    volatility_max_length: int
    
    use_regime_filter: bool
    regime_threshold: float
    
    use_adx_filter: bool
    adx_length: int
    adx_threshold: int
    
    use_ema_filter: bool
    ema_period: int
    use_sma_filter: bool
    sma_period: int
    
    use_kernel_filter: bool
    use_kernel_smoothing: bool
    kernel_lookback: int
    kernel_relative_weight: float
    kernel_regression_level: int
    kernel_lag: int

STRATEGY = StrategySettings(
    confidence_threshold=CONFIDENCE_THRESHOLD,
    neighbors_count=NEIGHBORS_COUNT,
    max_bars_back=MAX_BARS_BACK,
    label_horizon=LABEL_HORIZON,
    
    use_volatility_filter=USE_VOLATILITY_FILTER,
    volatility_min_length=VOLATILITY_MIN_LENGTH,
    volatility_max_length=VOLATILITY_MAX_LENGTH,
    
    use_regime_filter=USE_REGIME_FILTER,
    regime_threshold=REGIME_THRESHOLD,
    
    use_adx_filter=USE_ADX_FILTER,
    adx_length=ADX_LENGTH,
    adx_threshold=ADX_THRESHOLD,
    
    use_ema_filter=USE_EMA_FILTER,
    ema_period=EMA_PERIOD,
    use_sma_filter=USE_SMA_FILTER,
    sma_period=SMA_PERIOD,
    
    use_kernel_filter=USE_KERNEL_FILTER,
    use_kernel_smoothing=USE_KERNEL_SMOOTHING,
    kernel_lookback=KERNEL_LOOKBACK,
    kernel_relative_weight=KERNEL_RELATIVE_WEIGHT,
    kernel_regression_level=KERNEL_REGRESSION_LEVEL,
    kernel_lag=KERNEL_LAG,
)
