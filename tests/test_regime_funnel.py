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
