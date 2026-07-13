from dataclasses import dataclass

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
# RSI Settings
RSI_PERIOD = 14
RSI_MA_PERIOD = 14

# CCI Settings
CCI_PERIOD = 20
CCI_SMOOTHING_PERIOD = 14

# ADX Settings
ADX_PERIOD = 14
ADX_SMOOTHING_PERIOD = 14

# WaveTrend Settings
WT_CHANNEL_LENGTH = 10
WT_AVERAGE_LENGTH = 21
WT_SMA_LENGTH = 4

# ==============================================================================
# INTERNAL
# ==============================================================================
@dataclass(frozen=True)
class IndicatorSettings:
    rsi_period: int
    rsi_ma_period: int
    
    cci_period: int
    cci_smoothing_period: int
    
    adx_period: int
    adx_smoothing_period: int
    
    wt_channel_length: int
    wt_average_length: int
    wt_sma_length: int

INDICATORS = IndicatorSettings(
    rsi_period=RSI_PERIOD,
    rsi_ma_period=RSI_MA_PERIOD,
    cci_period=CCI_PERIOD,
    cci_smoothing_period=CCI_SMOOTHING_PERIOD,
    adx_period=ADX_PERIOD,
    adx_smoothing_period=ADX_SMOOTHING_PERIOD,
    wt_channel_length=WT_CHANNEL_LENGTH,
    wt_average_length=WT_AVERAGE_LENGTH,
    wt_sma_length=WT_SMA_LENGTH,
)
