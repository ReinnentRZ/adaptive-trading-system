"""
Unit tests for RegimeFunnelStrategy (Single Source of Truth 3-Pilar Engine).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config.strategy import RegimeFunnelConfig
from src.strategies.regime_funnel import RegimeFunnelStrategy


@pytest.fixture
def mock_ohlcv_data() -> pd.DataFrame:
    """Generates synthetic OHLCV data for strategy testing."""
    np.random.seed(42)
    n = 250
    base_price = 100.0
    returns = np.random.normal(0.001, 0.01, n)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open": prices * (1.0 - 0.001),
            "high": prices * (1.0 + 0.005),
            "low": prices * (1.0 - 0.005),
            "close": prices,
            "volume": np.random.uniform(100, 1000, n),
        }
    )
    return df


def test_regime_funnel_strategy_initialization():
    """Verify that RegimeFunnelStrategy initializes properly with config."""
    strategy = RegimeFunnelStrategy()
    assert strategy.config is not None
    assert strategy.model is not None
    assert strategy.scaler is not None
    assert strategy.bullish_state_id == 0


def test_regime_funnel_compute_indicators(mock_ohlcv_data):
    """Verify indicator computation produces all expected columns."""
    strategy = RegimeFunnelStrategy()
    df_res = strategy.compute_indicators(mock_ohlcv_data)

    required_cols = [
        "ema_200",
        "ema_9",
        "rsi_14",
        "atr_14",
        "atr",
        "volatility",
        "normalized_atr",
        "signed_volume",
        "volume_intensity",
        "log_return",
    ]
    for col in required_cols:
        assert col in df_res.columns, f"Missing column {col}"

    assert not df_res["ema_200"].dropna().empty
    assert not df_res["ema_9"].dropna().empty
    assert not df_res["rsi_14"].dropna().empty
    assert not df_res["atr"].dropna().empty


def test_regime_funnel_risk_brackets():
    """Verify calculation of SL, TP1, TP2, and BE ratchet prices."""
    config = RegimeFunnelConfig(
        risk_sl_mult=0.70,
        risk_tp1_mult=0.80,
        risk_tp2_mult=1.20,
        risk_be_buffer=1.0025,
    )
    strategy = RegimeFunnelStrategy(config=config)
    entry = 100000.0
    atr = 1000.0

    brackets = strategy.calculate_risk_brackets(entry, atr)
    assert brackets["sl_price"] == pytest.approx(100000.0 - 0.70 * 1000.0)
    assert brackets["tp1_price"] == pytest.approx(100000.0 + 0.80 * 1000.0)
    assert brackets["tp2_price"] == pytest.approx(100000.0 + 1.20 * 1000.0)
    assert brackets["be_price"] == pytest.approx(100000.0 * 1.0025)


def test_regime_funnel_evaluate_gates():
    """Verify Pilar 1 and Pilar 2 evaluation behavior."""
    strategy = RegimeFunnelStrategy()

    # Create dummy dataframe row
    df = pd.DataFrame(
        [
            {
                "close": 105.0,
                "low": 98.0,
                "ema_200": 100.0,
                "ema_9": 99.0,
                "rsi_14": 45.0,
            }
        ]
    )

    # Condition: State == 0, Close(105) > EMA200(100), state_age=2 <= 4, not traded, Low(98) <= EMA9(99)
    passed, details = strategy.evaluate_gates(
        df=df,
        idx=0,
        current_state=0,
        state_age=2,
        traded_in_episode=False,
    )
    assert passed is True
    assert details["pilar_1"] is True
    assert details["pilar_2"] is True

    # Reject if traded in episode (single-shot)
    passed_traded, details_traded = strategy.evaluate_gates(
        df=df,
        idx=0,
        current_state=0,
        state_age=2,
        traded_in_episode=True,
    )
    assert passed_traded is False
    assert details_traded["pilar_1"] is False

    # Reject if Close <= EMA200
    df_bear = df.copy()
    df_bear["close"] = 95.0
    passed_bear, _ = strategy.evaluate_gates(
        df=df_bear,
        idx=0,
        current_state=0,
        state_age=2,
        traded_in_episode=False,
    )
    assert passed_bear is False

    # Reject if state_age > 4
    passed_old, _ = strategy.evaluate_gates(
        df=df,
        idx=0,
        current_state=0,
        state_age=5,
        traded_in_episode=False,
    )
    assert passed_old is False

    # Test Volume Confirmation Gate
    df_vol = df.copy()
    df_vol["volume"] = 100.0
    df_vol["volume_sma20"] = 100.0  # volume (100) is NOT > 100 * 1.2 (120)
    passed_low_vol, details_vol = strategy.evaluate_gates(
        df=df_vol,
        idx=0,
        current_state=0,
        state_age=2,
        traded_in_episode=False,
    )
    assert passed_low_vol is False
    assert details_vol["pilar_vol"] is False

    df_vol_high = df.copy()
    df_vol_high["volume"] = 130.0  # volume (130) > 100 * 1.2 (120)
    df_vol_high["volume_sma20"] = 100.0
    passed_high_vol, details_high_vol = strategy.evaluate_gates(
        df=df_vol_high,
        idx=0,
        current_state=0,
        state_age=2,
        traded_in_episode=False,
    )
    assert passed_high_vol is True
    assert details_high_vol["pilar_vol"] is True


def test_regime_funnel_calculate_adaptive_brackets():
    """Verify dynamic regime-adaptive brackets calculation for all 3 market regimes."""
    from src.config.strategy import AdaptiveBracketParams
    from src.strategies.regime_funnel import calculate_adaptive_brackets

    strategy = RegimeFunnelStrategy()
    params = AdaptiveBracketParams()

    entry = 50000.0
    atr = 1000.0

    # 1. Test STRONG_BULL (close > ema200 * 1.015 and rsi >= 55)
    ema200 = 48000.0  # ema200 * 1.015 = 48720.0
    close_bull = 49000.0  # > 48720.0
    rsi_bull = 60.0  # >= 55.0
    res_bull = strategy.calculate_adaptive_brackets(
        entry_price=entry,
        atr_val=atr,
        current_close=close_bull,
        ema200_val=ema200,
        rsi_val=rsi_bull,
        natr_val=0.02,
        natr_median=0.02,
        params=params,
    )
    assert res_bull["regime_mode"] == "STRONG_BULL"
    assert res_bull["sl_price"] == pytest.approx(entry - 0.70 * atr)
    assert res_bull["tp1_price"] == pytest.approx(entry + 1.00 * atr)
    assert res_bull["tp2_price"] == pytest.approx(entry + 3.20 * atr)
    assert res_bull["tp1_ratio"] == pytest.approx(0.25)
    assert res_bull["tp2_ratio"] == pytest.approx(0.75)

    # 2. Test VOLATILE_CORRECTION (natr > natr_median * 1.25 and close < ema200 * 1.01)
    ema200 = 50000.0  # ema200 * 1.01 = 50500.0
    close_corr = 50200.0  # < 50500.0
    rsi_corr = 48.0
    natr_val = 0.030  # > 0.020 * 1.25 = 0.025
    natr_median = 0.020
    res_corr = calculate_adaptive_brackets(
        entry_price=entry,
        atr_val=atr,
        current_close=close_corr,
        ema200_val=ema200,
        rsi_val=rsi_corr,
        natr_val=natr_val,
        natr_median=natr_median,
        params=params,
    )
    assert res_corr["regime_mode"] == "VOLATILE_CORRECTION"
    assert res_corr["sl_price"] == pytest.approx(entry - 0.50 * atr)
    assert res_corr["tp1_price"] == pytest.approx(entry + 0.75 * atr)
    assert res_corr["tp2_price"] == pytest.approx(entry + 1.10 * atr)
    assert res_corr["tp1_ratio"] == pytest.approx(0.50)
    assert res_corr["tp2_ratio"] == pytest.approx(0.50)

    # 3. Test NORMAL_SIDEWAYS (Default)
    ema200 = 50000.0
    close_side = 50000.0
    rsi_side = 50.0
    natr_val = 0.020
    natr_median = 0.020
    res_side = strategy.calculate_adaptive_brackets(
        entry_price=entry,
        atr_val=atr,
        current_close=close_side,
        ema200_val=ema200,
        rsi_val=rsi_side,
        natr_val=natr_val,
        natr_median=natr_median,
        params=params,
    )
    assert res_side["regime_mode"] == "NORMAL_SIDEWAYS"
    assert res_side["sl_price"] == pytest.approx(entry - 0.60 * atr)
    assert res_side["tp1_price"] == pytest.approx(entry + 0.60 * atr)
    assert res_side["tp2_price"] == pytest.approx(entry + 0.90 * atr)
    assert res_side["tp1_ratio"] == pytest.approx(0.70)
    assert res_side["tp2_ratio"] == pytest.approx(0.30)

