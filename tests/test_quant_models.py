"""
Unit Tests for Quantitative Models: Adaptive Kalman Filter and GARCH(1,1) Volatility Forecaster.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.strategy import AdaptiveBracketParams, RegimeFunnelConfig
from src.strategies.quant_models import (
    GarchForecastResult,
    KalmanTrendResult,
    compute_rolling_garch_volatility,
    forecast_garch_volatility,
    kalman_trend_filter,
)
from src.strategies.regime_funnel import (
    RegimeFunnelStrategy,
    calculate_adaptive_brackets,
)


class TestKalmanTrendFilter:
    """Tests for 1D Adaptive Kalman Filter."""

    def test_kalman_output_dimensions_and_types(self):
        prices = np.linspace(50000, 55000, 100) + np.random.normal(0, 100, 100)
        res = kalman_trend_filter(prices, q=1e-5, r=0.01)

        assert isinstance(res, KalmanTrendResult)
        assert len(res.kalman_price) == 100
        assert len(res.kalman_trend) == 100
        assert len(res.kalman_slope) == 100
        assert not np.isnan(res.kalman_price[-1])
        assert not np.isnan(res.kalman_trend[-1])
        assert not np.isnan(res.kalman_slope[-1])

    def test_kalman_strict_causality_zero_lookahead(self):
        """Modifying future prices at t > 50 must NOT affect estimates at t <= 50."""
        np.random.seed(123)
        prices_orig = 60000.0 + np.cumsum(np.random.normal(10, 200, 100))
        prices_modified = prices_orig.copy()
        prices_modified[51:] += 10000.0  # Massive future shock at t=51

        res_orig = kalman_trend_filter(prices_orig, q=1e-5, r=0.01)
        res_mod = kalman_trend_filter(prices_modified, q=1e-5, r=0.01)

        # Prior history up to t=50 must be bitwise identical
        np.testing.assert_array_almost_equal(
            res_orig.kalman_price[:51],
            res_mod.kalman_price[:51],
            decimal=8,
        )
        np.testing.assert_array_almost_equal(
            res_orig.kalman_trend[:51],
            res_mod.kalman_trend[:51],
            decimal=8,
        )

    def test_kalman_scale_invariance(self):
        """Kalman filter should handle both high-price BTC ($60k) and sub-dollar altcoins ($0.05)."""
        btc_prices = np.linspace(60000, 62000, 50)
        penny_prices = np.linspace(0.05, 0.052, 50)

        res_btc = kalman_trend_filter(btc_prices)
        res_penny = kalman_trend_filter(penny_prices)

        assert res_btc.kalman_price[-1] > 55000.0
        assert res_penny.kalman_price[-1] < 0.10
        assert np.isfinite(res_btc.kalman_slope[-1])
        assert np.isfinite(res_penny.kalman_slope[-1])

    def test_kalman_edge_cases(self):
        empty_res = kalman_trend_filter([])
        assert len(empty_res.kalman_price) == 0

        single_res = kalman_trend_filter([100.0])
        assert len(single_res.kalman_price) == 1
        assert single_res.kalman_price[0] == 100.0


class TestGarchVolatilityForecaster:
    """Tests for GARCH(1,1) Volatility Forecaster."""

    def test_garch_forecast_validity(self):
        np.random.seed(42)
        # Synthetic daily/hourly returns with volatility clustering
        returns = np.random.normal(0, 0.015, 200)
        prices = 50000.0 * np.exp(np.cumsum(returns))

        res = forecast_garch_volatility(prices, lookback=150)

        assert isinstance(res, GarchForecastResult)
        assert res.sigma_pct > 0.0
        assert res.sigma_price > 0.0
        assert res.annualized_vol > 0.0
        assert res.alpha >= 0.0
        assert res.beta >= 0.0
        assert (res.alpha + res.beta) <= 1.0  # Stationarity constraint
        assert res.omega > 0.0

    def test_garch_insufficient_data_fallback(self):
        short_prices = [100.0, 101.0, 99.5]
        res = forecast_garch_volatility(short_prices, lookback=250)
        assert res.sigma_pct > 0.0
        assert res.sigma_price > 0.0
        assert not res.success  # Fallback triggered cleanly

    def test_rolling_garch_volatility(self):
        np.random.seed(99)
        prices = 1000.0 + np.cumsum(np.random.normal(0, 10, 60))
        rolling_vols = compute_rolling_garch_volatility(prices, lookback=40, stride=5)

        assert len(rolling_vols) == 60
        assert not np.isnan(rolling_vols[-1])
        assert np.all(rolling_vols > 0.0)


class TestRegimeFunnelQuantIntegration:
    """Tests integration of Kalman and GARCH inside RegimeFunnelStrategy."""

    def test_strategy_indicators_with_kalman_enabled(self):
        cfg = RegimeFunnelConfig(
            use_kalman_filter=True,
            kalman_q=1e-5,
            kalman_r=0.01,
        )
        strategy = RegimeFunnelStrategy(config=cfg)

        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "datetime": dates,
            "open": np.linspace(100, 120, n),
            "high": np.linspace(101, 121, n),
            "low": np.linspace(99, 119, n),
            "close": np.linspace(100, 120, n),
            "volume": np.full(n, 1000.0),
        })

        df_out = strategy.compute_indicators(df)
        assert "kalman_price" in df_out.columns
        assert "kalman_trend" in df_out.columns
        assert "kalman_slope" in df_out.columns

    def test_adaptive_brackets_with_garch_price_vol(self):
        brackets_standard = calculate_adaptive_brackets(
            entry_price=60000.0,
            atr_val=1000.0,
            current_close=62000.0,
            ema200_val=60000.0,
            rsi_val=60.0,
            natr_val=0.02,
            natr_median=0.02,
        )

        # Dynamic GARCH volatility higher than standard ATR (e.g., $1500)
        brackets_garch = calculate_adaptive_brackets(
            entry_price=60000.0,
            atr_val=1000.0,
            current_close=62000.0,
            ema200_val=60000.0,
            rsi_val=60.0,
            natr_val=0.02,
            natr_median=0.02,
            garch_price_vol=1500.0,
        )

        assert brackets_garch["vol_unit"] == 1500.0
        assert brackets_standard["vol_unit"] == 1000.0
        # TP2 should expand dynamically under higher GARCH forecasted volatility
        assert brackets_garch["tp2_price"] > brackets_standard["tp2_price"]
