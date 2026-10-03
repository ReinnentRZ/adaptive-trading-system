"""
Quantitative Models for Adaptive Trading Systems: Kalman Filter & GARCH(1,1).

Provides strictly causal, zero-lookahead mathematical models implemented in pure
NumPy and SciPy to maintain a lightweight, fast, and dependency-free deployment:
  1. 1D Adaptive Kalman Filter: Estimates true underlying price and instantaneous slope,
     providing low-lag replacements for micro timing (EMA 9) and macro trend (EMA 200).
  2. GARCH(1,1) Volatility Forecaster: Models conditional heteroskedasticity and forecasts
     1-step-ahead forward volatility for dynamic risk bracket sizing.

Authors: Adaptive Trading System Quant Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize


class KalmanTrendResult(NamedTuple):
    """Container for Kalman Filter outputs."""
    kalman_price: np.ndarray  # Responsive filtered price (EMA 9 replacement)
    kalman_trend: np.ndarray  # Smooth macro trend line (EMA 200 replacement)
    kalman_slope: np.ndarray  # Instantaneous slope / velocity (dPrice/dt)


@dataclass
class GarchForecastResult:
    """Container for GARCH(1,1) volatility forecast."""
    sigma_pct: float          # 1-step-ahead conditional volatility as fraction (e.g. 0.015 = 1.5%)
    sigma_price: float        # 1-step-ahead volatility in dollar price (Close * sigma_pct)
    annualized_vol: float     # Annualized volatility (assuming 24*365 crypto hourly periods)
    omega: float              # Baseline variance weight
    alpha: float              # ARCH coefficient (reaction to shocks)
    beta: float               # GARCH coefficient (persistence of variance)
    success: bool             # Whether MLE optimization converged cleanly


# ==============================================================================
# 1. 1D Adaptive Kalman Filter (Causal Trend & Micro Smoothing)
# ==============================================================================

def kalman_trend_filter(
    close: Union[Sequence[float], pd.Series, np.ndarray],
    q: float = 1e-5,
    r: float = 0.01,
    macro_q_mult: float = 0.01,
    macro_r_mult: float = 5.0,
) -> KalmanTrendResult:
    """
    Applies a scale-invariant, 2-speed 1D Adaptive Kalman Filter to price data.

    State Formulation:
        State vector: x_t = [p_t, v_t]^T (price and instantaneous velocity/slope)
        State transition: x_t = F x_{t-1} + w_t
            F = [[1, 1], [0, 1]]
        Measurement: z_t = H x_t + v_t
            H = [[1, 0]]

    Parameters:
        close: Input price sequence (strictly causal, evaluated up to bar t).
        q: Process noise covariance scalar for fast tracking (micro filter).
        r: Measurement noise covariance scalar for fast tracking.
        macro_q_mult: Multiplier applied to q for the macro trend filter (default: 0.01).
        macro_r_mult: Multiplier applied to r for the macro trend filter (default: 5.0).

    Returns:
        KalmanTrendResult(kalman_price, kalman_trend, kalman_slope)
    """
    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return KalmanTrendResult(empty, empty, empty)

    # Scale-invariance: Normalize by initial non-zero price
    p0 = float(close_arr[0]) if close_arr[0] != 0.0 else 1.0
    norm_close = close_arr / p0

    # 1. Fast Filter Setup (Micro: EMA 9 replacement)
    q_fast = max(float(q), 1e-9)
    r_fast = max(float(r), 1e-9)
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)

    Q_fast = np.array([[q_fast, 0.0], [0.0, q_fast * 0.1]], dtype=np.float64)
    R_fast = np.array([[r_fast]], dtype=np.float64)
    x_fast = np.array([[norm_close[0]], [0.0]], dtype=np.float64)
    P_fast = np.eye(2, dtype=np.float64) * 1.0

    # 2. Slow Filter Setup (Macro: EMA 200 replacement with minimal lag)
    q_slow = max(float(q * macro_q_mult), 1e-12)
    r_slow = max(float(r * macro_r_mult), 1e-9)
    Q_slow = np.array([[q_slow, 0.0], [0.0, q_slow * 0.01]], dtype=np.float64)
    R_slow = np.array([[r_slow]], dtype=np.float64)
    x_slow = np.array([[norm_close[0]], [0.0]], dtype=np.float64)
    P_slow = np.eye(2, dtype=np.float64) * 1.0

    I = np.eye(2, dtype=np.float64)

    kalman_price = np.empty(n, dtype=np.float64)
    kalman_trend = np.empty(n, dtype=np.float64)
    kalman_slope = np.empty(n, dtype=np.float64)

    # Strictly causal forward recursion
    for t in range(n):
        z = norm_close[t]
        if np.isnan(z):
            # Forward propagate on NaN
            x_fast = F @ x_fast
            x_slow = F @ x_slow
            kalman_price[t] = x_fast[0, 0] * p0
            kalman_trend[t] = x_slow[0, 0] * p0
            kalman_slope[t] = x_fast[1, 0] * p0
            continue

        # Fast Filter Step
        xf_pred = F @ x_fast
        Pf_pred = F @ P_fast @ F.T + Q_fast
        yf = z - (H @ xf_pred)[0, 0]
        Sf = (H @ Pf_pred @ H.T)[0, 0] + R_fast[0, 0]
        Kf = Pf_pred @ H.T / Sf
        x_fast = xf_pred + Kf * yf
        P_fast = (I - Kf @ H) @ Pf_pred

        # Slow Filter Step
        xs_pred = F @ x_slow
        Ps_pred = F @ P_slow @ F.T + Q_slow
        ys = z - (H @ xs_pred)[0, 0]
        Ss = (H @ Ps_pred @ H.T)[0, 0] + R_slow[0, 0]
        Ks = Ps_pred @ H.T / Ss
        x_slow = xs_pred + Ks * ys
        P_slow = (I - Ks @ H) @ Ps_pred

        # Denormalize outputs back to currency price
        kalman_price[t] = x_fast[0, 0] * p0
        kalman_trend[t] = x_slow[0, 0] * p0
        kalman_slope[t] = x_fast[1, 0] * p0

    return KalmanTrendResult(
        kalman_price=kalman_price,
        kalman_trend=kalman_trend,
        kalman_slope=kalman_slope,
    )


# ==============================================================================
# 2. GARCH(1,1) Volatility Forecaster (Conditional Variance Forward Prediction)
# ==============================================================================

def forecast_garch_volatility(
    close: Union[Sequence[float], pd.Series, np.ndarray],
    lookback: int = 250,
    annualization_factor: float = np.sqrt(24 * 365),  # 1H crypto annualizer ~93.59
) -> GarchForecastResult:
    """
    Fits a GARCH(1,1) model on recent log returns and forecasts 1-step-ahead volatility.

    Model:
        r_t = ln(P_t / P_{t-1})
        sigma_t^2 = omega + alpha * (r_{t-1} - mu)^2 + beta * sigma_{t-1}^2
        Forecast: sigma_{t+1}^2 = omega + alpha * (r_t - mu)^2 + beta * sigma_t^2

    Parameters:
        close: Historical price sequence up to current bar t (strictly causal).
        lookback: Rolling window of bars used for estimation (default: 250).
        annualization_factor: Volatility annualizer (default: sqrt(24 * 365) for 1h bars).

    Returns:
        GarchForecastResult with fractional volatility, dollar volatility, and parameters.
    """
    close_arr = np.asarray(close, dtype=np.float64)
    if len(close_arr) < 2:
        return GarchForecastResult(
            sigma_pct=0.01,
            sigma_price=0.01 * (close_arr[-1] if len(close_arr) > 0 else 1.0),
            annualized_vol=0.01 * annualization_factor,
            omega=1e-5,
            alpha=0.08,
            beta=0.90,
            success=False,
        )

    # Calculate log returns
    valid_close = close_arr[~np.isnan(close_arr)]
    if len(valid_close) < 15:
        # Fallback for insufficient data
        last_price = float(valid_close[-1]) if len(valid_close) > 0 else 1.0
        return GarchForecastResult(
            sigma_pct=0.015,
            sigma_price=0.015 * last_price,
            annualized_vol=0.015 * annualization_factor,
            omega=1e-5,
            alpha=0.08,
            beta=0.90,
            success=False,
        )

    log_returns = np.diff(np.log(valid_close))
    # Slice causal lookback window
    window_returns = log_returns[-lookback:] if len(log_returns) > lookback else log_returns
    n = len(window_returns)
    last_price = float(valid_close[-1])

    # Percentage scaling for numerical stability in MLE
    scaled_r = window_returns * 100.0
    sample_var = float(np.var(scaled_r))
    if sample_var <= 1e-12:
        # Stationary flatline market
        flat_pct = 0.005
        return GarchForecastResult(
            sigma_pct=flat_pct,
            sigma_price=flat_pct * last_price,
            annualized_vol=flat_pct * annualization_factor,
            omega=1e-6,
            alpha=0.05,
            beta=0.90,
            success=False,
        )

    # Initial parameter guess
    init_alpha = 0.08
    init_beta = 0.88
    init_omega = sample_var * (1.0 - init_alpha - init_beta)
    init_params = [max(init_omega, 1e-4), init_alpha, init_beta]

    # Negative log-likelihood objective under Gaussian innovations
    def _nll(params: Sequence[float]) -> float:
        om, al, be = params
        sigma2 = np.empty(n, dtype=np.float64)
        sigma2[0] = sample_var
        for i in range(1, n):
            sigma2[i] = om + al * (scaled_r[i - 1] ** 2) + be * sigma2[i - 1]
        sigma2 = np.maximum(sigma2, 1e-6)
        # Log likelihood
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (scaled_r ** 2) / sigma2)
        return -ll

    bounds = [(1e-6, None), (1e-4, 0.40), (0.50, 0.999)]
    # Stationarity constraint: alpha + beta <= 0.999
    constraints = ({"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2])})

    opt_success = False
    try:
        res = optimize.minimize(
            _nll,
            init_params,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 80, "ftol": 1e-5},
        )
        if res.success and np.isfinite(res.fun):
            omega_scaled, alpha, beta = res.x
            opt_success = True
        else:
            omega_scaled, alpha, beta = init_params
    except Exception:
        omega_scaled, alpha, beta = init_params

    # Filter conditional variance forward through the window
    sigma2_t = sample_var
    for i in range(1, n):
        sigma2_t = omega_scaled + alpha * (scaled_r[i - 1] ** 2) + beta * sigma2_t

    # 1-step-ahead forecast
    forecast_sigma2_scaled = omega_scaled + alpha * (scaled_r[-1] ** 2) + beta * sigma2_t
    forecast_sigma2_scaled = max(forecast_sigma2_scaled, 1e-6)

    # Convert back from percentage scale to raw fractional scale
    sigma_pct = float(np.sqrt(forecast_sigma2_scaled) / 100.0)
    sigma_price = float(last_price * sigma_pct)
    annualized_vol = float(sigma_pct * annualization_factor)
    omega_unscaled = float(omega_scaled / 10000.0)

    return GarchForecastResult(
        sigma_pct=sigma_pct,
        sigma_price=sigma_price,
        annualized_vol=annualized_vol,
        omega=omega_unscaled,
        alpha=float(alpha),
        beta=float(beta),
        success=opt_success,
    )


def compute_rolling_garch_volatility(
    close: Union[Sequence[float], pd.Series, np.ndarray],
    lookback: int = 250,
    stride: int = 24,
) -> np.ndarray:
    """
    Computes a causal rolling series of GARCH(1,1) 1-step-ahead volatility.

    Parameters (omega, alpha, beta) are periodically re-estimated via MLE every `stride` bars
    (default: 24 bars / once daily for 1H crypto), while conditional variance sigma_{t+1}^2
    is updated recursively at every bar t (O(1)), providing high performance.

    Parameters:
        close: Full price series.
        lookback: Rolling estimation window for MLE fitting.
        stride: Frequency of re-fitting MLE (default: 24 bars).

    Returns:
        Array of fractional 1-step forward volatility estimates.
    """
    close_arr = np.asarray(close, dtype=np.float64)
    n = len(close_arr)
    garch_vols = np.full(n, np.nan, dtype=np.float64)

    if n < 30:
        return garch_vols

    valid_mask = ~np.isnan(close_arr)
    clean_close = np.where(valid_mask, close_arr, 1.0)
    log_returns = np.zeros(n, dtype=np.float64)
    log_returns[1:] = np.diff(np.log(clean_close))

    current_sigma2 = float(np.var(log_returns[1:30])) if n >= 30 else 1e-4
    current_sigma2 = max(current_sigma2, 1e-8)
    om, al, be = 1e-6, 0.08, 0.90

    for t in range(30, n):
        # Periodic parameter re-estimation
        if (t % stride == 0) or (t == 30):
            hist = close_arr[: t + 1]
            res = forecast_garch_volatility(hist, lookback=lookback)
            om, al, be = res.omega, res.alpha, res.beta
            current_sigma2 = (res.sigma_pct) ** 2
        else:
            # Recursive exact GARCH(1,1) update
            r_prev = log_returns[t - 1]
            current_sigma2 = om + al * (r_prev ** 2) + be * current_sigma2
            current_sigma2 = max(current_sigma2, 1e-8)

        # 1-step ahead forecast: sigma_{t+1}^2 = omega + alpha * r_t^2 + beta * sigma_t^2
        r_curr = log_returns[t]
        f_sigma2 = om + al * (r_curr ** 2) + be * current_sigma2
        garch_vols[t] = np.sqrt(max(f_sigma2, 1e-8))

    # Backward fill warm-up period
    if np.isnan(garch_vols[0]) and np.any(~np.isnan(garch_vols)):
        first_valid = garch_vols[~np.isnan(garch_vols)][0]
        garch_vols[:30] = first_valid

    return garch_vols
