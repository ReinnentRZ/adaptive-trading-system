"""
Lorentzian Classification -- core ML strategy.

Reverse-engineered from the "Machine Learning: Lorentzian Classification"
Pine Script indicator (jdehorty). Feature engineering, Lorentzian distance,
and the approximate nearest-neighbor vote mirror the original indicator's
"Core ML Logic" section. The post-prediction decision logic (volatility /
regime / ADX filters, the sticky `signal` state machine, and kernel-
regression / EMA / SMA confirmation) is replayed in `_run_decision_engine`
so that `PredictionResult.signal` matches the original indicator's BUY /
SELL / HOLD behavior instead of just the sign of the raw vote.

Position/exit management (`endLongTrade` / `endShortTrade`) remains out of
scope: `Direction` has only three states and can't represent a fourth
"exit" signal distinct from NEUTRAL without changing that type.

Complexity
----------
- `approximate_nearest_neighbors`: O(max_bars_back) per call.
- `_compute_prediction_series` calls it once per bar in the dataset (to
  replay the sticky decision state machine), so one `analyze()` call is
  O(N * max_bars_back) where N is the number of candles passed in -- this
  mirrors the fact that the original Pine script also evaluates
  `prediction` at every bar as it walks the chart, not just the latest one.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, List, Optional, Sequence, Tuple

from src.config.strategy import STRATEGY
from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.rsi import RSIIndicator
from src.indicators.wt import WTIndicator

__all__ = [
    "Direction",
    "FeatureVector",
    "TrainingDataset",
    "LorentzianSettings",
    "DecisionEngineSettings",
    "PredictionResult",
    "lorentzian_distance",
    "generate_labels",
    "approximate_nearest_neighbors",
    "LorentzianStrategy",
]


class Direction(IntEnum):
    """
    Three-valued classification used both as the historical training label
    (did price rise/fall/stay flat `label_horizon` bars later) and as the
    final trading signal (LONG -> BUY, SHORT -> SELL, NEUTRAL -> HOLD). The
    original indicator reuses a single type for both concepts; this mirrors
    that instead of duplicating an identical three-state type.
    """

    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass(frozen=True)
class FeatureVector:
    """
    Immutable snapshot of normalized indicator readings for a single bar.

    Values are stored as a plain tuple rather than one named field per
    indicator so that every downstream calculation -- distance, neighbor
    search, prediction -- stays agnostic to which indicators are present or
    how many there are. Adding a new indicator to the model never requires
    touching this class or any function that consumes it (Open/Closed
    Principle).

    Precondition: callers are responsible for ensuring all values are on a
    comparable numeric scale (e.g. all normalized to a similar range).
    `lorentzian_distance` sums raw differences across dimensions, so a
    feature with a much larger numeric range than the others will silently
    dominate every distance computation.
    """

    values: Tuple[float, ...]
    feature_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feature_names and len(self.feature_names) != len(self.values):
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) must "
                f"match values length ({len(self.values)})."
            )

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    @property
    def is_valid(self) -> bool:
        """False if any feature is NaN (e.g. an indicator's warm-up period)."""
        return not any(math.isnan(value) for value in self.values)


@dataclass(frozen=True)
class TrainingDataset:
    """
    Chronological feature-vector series paired with realized direction
    labels, one entry per candle.

    Bundling the two sequences together (rather than passing them around as
    two separately-indexed parallel lists) makes it structurally impossible
    for them to fall out of sync.
    """

    feature_series: Tuple[FeatureVector, ...]
    labels: Tuple[Optional[Direction], ...]

    def __post_init__(self) -> None:
        if len(self.feature_series) != len(self.labels):
            raise ValueError(
                "feature_series and labels must have equal length "
                f"(got {len(self.feature_series)} and {len(self.labels)})."
            )

    def __len__(self) -> int:
        return len(self.feature_series)


@dataclass(frozen=True)
class LorentzianSettings:
    """
    Hyperparameters for the Lorentzian Classification model.

    `label_horizon` intentionally serves two roles, exactly as in the
    original indicator: it is both (a) the number of bars ahead used to
    label historical bars as LONG/SHORT/NEUTRAL, and (b) the minimum
    chronological spacing enforced between selected neighbors during the
    search. These two behaviors are conceptually linked -- spacing
    neighbors by less than the label horizon risks selecting "neighbors"
    whose forward-return windows overlap -- so a single field keeps them
    from silently drifting apart if one is tuned without the other.
    """

    neighbors_count: int = STRATEGY.neighbors_count
    max_bars_back: int = STRATEGY.max_bars_back
    label_horizon: int = STRATEGY.label_horizon

    def __post_init__(self) -> None:
        if self.neighbors_count < 1:
            raise ValueError("neighbors_count must be >= 1.")
        if self.max_bars_back < 1:
            raise ValueError("max_bars_back must be >= 1.")
        if self.label_horizon < 1:
            raise ValueError("label_horizon must be >= 1.")


@dataclass(frozen=True)
class DecisionEngineSettings:
    """
    Post-prediction filter/confirmation settings.

    Mirrors the original indicator's "Filters" and "Kernel Settings" input
    groups exactly, including their default values. None of this affects
    feature engineering, distance, or the neighbor vote -- it only governs
    how a vote is allowed to turn into a BUY/SELL transition (see
    `LorentzianStrategy._run_decision_engine`).
    """

    use_volatility_filter: bool = STRATEGY.use_volatility_filter
    volatility_min_length: int = STRATEGY.volatility_min_length
    volatility_max_length: int = STRATEGY.volatility_max_length

    use_regime_filter: bool = STRATEGY.use_regime_filter
    regime_threshold: float = STRATEGY.regime_threshold

    use_adx_filter: bool = STRATEGY.use_adx_filter
    adx_length: int = STRATEGY.adx_length
    adx_threshold: int = STRATEGY.adx_threshold

    use_ema_filter: bool = STRATEGY.use_ema_filter
    ema_period: int = STRATEGY.ema_period

    use_sma_filter: bool = STRATEGY.use_sma_filter
    sma_period: int = STRATEGY.sma_period

    use_kernel_filter: bool = STRATEGY.use_kernel_filter
    use_kernel_smoothing: bool = STRATEGY.use_kernel_smoothing
    kernel_lookback: int = STRATEGY.kernel_lookback
    kernel_relative_weight: float = STRATEGY.kernel_relative_weight
    kernel_regression_level: int = STRATEGY.kernel_regression_level
    kernel_lag: int = STRATEGY.kernel_lag


_DEFAULT_DECISION_ENGINE_SETTINGS = DecisionEngineSettings()


@dataclass(frozen=True)
class PredictionResult:
    """
    Outcome of the model for a single bar.

    `prediction` is the signed sum of the selected neighbors' labels (range
    -neighbors_count..+neighbors_count). Its magnitude is a usable
    confidence proxy -- how many of the k neighbors agreed on direction.
    `signal` is the result of the full post-prediction decision engine
    (filters + sticky state + kernel/EMA/SMA confirmation), NOT simply the
    sign of `prediction` -- see `LorentzianStrategy._run_decision_engine`.
    """

    prediction: int
    signal: Direction
    #tambahan
    neighbors_labels: List[int]



PriceExtractor = Callable[[Sequence[Any]], Sequence[float]]


def _default_close_extractor(candles: Sequence[Any]) -> List[float]:
    """
    Extract closing prices from a candle sequence.

    Supports attribute-style (`candle.close`) and mapping-style
    (`candle["close"]`) candles, so this module makes no hard assumption
    about the concrete Candle type used elsewhere in the system. Pass a
    different callable via `LorentzianStrategy(..., source_extractor=...)`
    if candles are structured differently (e.g. to use hlc3 as the source).
    """
    prices: List[float] = []
    for candle in candles:
        if hasattr(candle, "close"):
            prices.append(float(candle.close))
        elif isinstance(candle, dict) and "close" in candle:
            prices.append(float(candle["close"]))
        else:
            raise TypeError(
                "Each candle must expose a 'close' attribute or key to be "
                "used as the Lorentzian model's price source."
            )
    return prices


def _extract_price_field(candles: Sequence[Any], field: str) -> List[float]:
    """
    Generic OHLC field extractor used only by the decision engine's filters
    (volatility/regime/ADX/kernel all need `open`/`high`/`low` in addition
    to the `close` series `_default_close_extractor` already provides).
    Duck-types the same way `_default_close_extractor` does; does not touch
    or replace it.
    """
    values: List[float] = []
    
    for candle in candles:
        if hasattr(candle, field):
            values.append(float(getattr(candle, field)))
        elif isinstance(candle, dict) and field in candle:
            values.append(float(candle[field]))
        else:
            raise TypeError(
                f"Each candle must expose a '{field}' attribute or key for "
                "the decision engine's filters."
            )
    return values


def lorentzian_distance(a: FeatureVector, b: FeatureVector) -> float:
    """
    D(a, b) = sum_k log(1 + |a_k - b_k|)

    Pure function. Unlike Euclidean distance, a large discrepancy in any
    single feature dimension is compressed logarithmically rather than
    squared, so one outlier-ish feature reading cannot dominate the overall
    distance the way it would under a squared-difference metric.
    """
    if len(a) != len(b):
        raise ValueError(
            "Feature vectors must have equal length to compute distance "
            f"(got {len(a)} and {len(b)})."
        )
    if a.feature_names and b.feature_names and a.feature_names != b.feature_names:
        raise ValueError(
            f"Feature vectors describe different features: "
            f"{a.feature_names} vs {b.feature_names}."
        )
    return sum(math.log1p(abs(x - y)) for x, y in zip(a.values, b.values))


def generate_labels(
    source_prices: Sequence[float], horizon: int
) -> List[Optional[Direction]]:
    """
    Label bar i by comparing price[i] to price[i + horizon]: LONG if price
    rose over the horizon, SHORT if it fell, NEUTRAL if unchanged.

    Implementation note: the original Pine script computes this comparison
    with its `[]` history-referencing operator (`src[horizon] vs src[0]`),
    which can only look *backward* from whichever bar is currently being
    evaluated. Taken completely literally, that produces a value stored at
    bar k comparing close[k - horizon] to close[k] -- i.e. it ends up paired
    with bar k's own features but describes the price move *leading into*
    bar k, with LONG/SHORT swapped relative to the convention below. This
    function instead implements the standard, leak-free forward-return
    convention that the script's own comments describe ("predicting the
    direction of price action over the course of the next `horizon` bars"),
    pairing each bar's features with its own subsequent outcome.

    The final `horizon` bars cannot be labeled (their outcome is unknown)
    and are returned as None; callers must exclude these when selecting
    neighbors. Pure function: depends only on its arguments.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    n = len(source_prices)
    labels: List[Optional[Direction]] = [None] * n
    for i in range(n - horizon):
        future_price = source_prices[i + horizon]
        current_price = source_prices[i]
        if future_price > current_price:
            labels[i] = Direction.LONG
        elif future_price < current_price:
            labels[i] = Direction.SHORT
        else:
            labels[i] = Direction.NEUTRAL
    return labels


def _round_half_away_from_zero(value: float) -> int:
    """
    Round-half-away-from-zero, matching Pine Script's `math.round`.

    Python's built-in `round()` uses round-half-to-even ("banker's
    rounding"), which can disagree with Pine on exact `.5` boundaries. This
    only matters for `neighbors_count` values where `neighbors_count * 0.75`
    lands exactly on a half-integer (e.g. neighbors_count=6 -> 4.5).
    """
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def approximate_nearest_neighbors(
    current: FeatureVector,
    history: Sequence[FeatureVector],
    history_labels: Sequence[Optional[Direction]],
    settings: LorentzianSettings,
) -> Tuple[int, List[int]]:
    """
    Single-pass, greedy approximate nearest-neighbors search and vote.

    This deliberately reproduces the original indicator's specific behavior
    rather than textbook KNN:

    1. Chronological spacing: a candidate at index `i` is only eligible if
       `i % settings.label_horizon != 0`. Combined with the sliding window
       below, this enforces a minimum chronological gap between selected
       neighbors, avoiding neighbors whose forward-return label windows
       overlap each other.
    2. Monotonic acceptance: a candidate is accepted only if its distance is
       >= the last *accepted* distance. This is a cheap, single forward
       pass -- it does NOT guarantee the k globally-closest neighbors, only
       a set of k neighbors with non-decreasing distances in the order seen.
       This is what makes the search "approximate".
    3. Sliding-window eviction: once more than `neighbors_count` neighbors
       are held, the oldest is evicted, and the acceptance threshold
       (`last_distance`) is reset to the value at the 75th-percentile
       position of the remaining distances (not the maximum). Resetting to
       the max would make the threshold ratchet upward and eventually
       reject everything; resetting to the 75th percentile keeps the pool
       open to a wider range of future candidates.

    Returns the signed sum of the selected neighbors' labels.
    """
    if len(history) != len(history_labels):
        raise ValueError(
            "history and history_labels must have equal length "
            f"(got {len(history)} and {len(history_labels)})."
        )

    last_distance = -1.0
    distances: List[float] = []
    predictions: List[int] = []

    search_size = min(settings.max_bars_back, len(history))

    for i in range(search_size):
        label = history_labels[i]
        if label is None:
            continue  # Not yet labeled (too close to the end of history).

        candidate = history[i]
        if not candidate.is_valid:
            continue  # Indicator warm-up period for this historical bar.

        if i % settings.label_horizon == 0:
            continue  # Enforce chronological spacing between neighbors.

        distance = lorentzian_distance(current, candidate)
        if distance < last_distance:
            continue

        last_distance = distance
        distances.append(distance)
        predictions.append(int(label))

        if len(predictions) > settings.neighbors_count:
            cutoff_index = _round_half_away_from_zero(settings.neighbors_count * 0.75)
            last_distance = distances[cutoff_index]
            distances.pop(0)
            predictions.pop(0)

    return sum(predictions), predictions


# ---------------------------------------------------------------------------
# Decision-engine primitives.
#
# Everything below is transcribed directly from the MLExtensions /
# KernelFunctions Pine source: True Range / Wilder smoothing / EMA / SMA,
# the volatility / regime / ADX filters, and the Rational Quadratic /
# Gaussian kernels. None of it touches feature engineering, distance, or
# the neighbor vote above -- it only governs whether/when a vote is allowed
# to become a BUY/SELL transition (see `LorentzianStrategy._run_decision_engine`).
# ---------------------------------------------------------------------------


def _rma(values: Sequence[float], length: int) -> List[float]:
    """Wilder's moving average, matching Pine's `ta.rma` (SMA-seeded EMA)."""
    n = len(values)
    result: List[float] = [float("nan")] * n
    alpha = 1.0 / length
    for i in range(n):
        if i < length - 1:
            continue
        if i == length - 1:
            result[i] = sum(values[i - length + 1 : i + 1]) / length
        else:
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _ema_series(values: Sequence[float], length: int) -> List[float]:
    """Standard EMA, matching Pine's `ta.ema` (SMA-seeded EMA)."""
    n = len(values)
    result: List[float] = [float("nan")] * n
    alpha = 2.0 / (length + 1)
    for i in range(n):
        if i < length - 1:
            continue
        if i == length - 1:
            result[i] = sum(values[i - length + 1 : i + 1]) / length
        else:
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _sma_series(values: Sequence[float], length: int) -> List[float]:
    """Standard simple moving average, matching Pine's `ta.sma`."""
    n = len(values)
    result: List[float] = [float("nan")] * n
    for i in range(n):
        if i < length - 1:
            continue
        result[i] = sum(values[i - length + 1 : i + 1]) / length
    return result


def _wilder_cumulative_smooth(values: Sequence[float], length: int) -> List[float]:
    """
    Matches the specific recursion used inline in the original
    `filter_adx`/`n_adx` Pine code -- `x := nz(x[1]) - nz(x[1])/length + v`,
    seeded at 0 from the very first bar. This is distinct from `_rma`
    (which is SMA-seeded), so it is kept as a separate helper rather than
    reusing `_rma` for the sake of exact fidelity.
    """
    n = len(values)
    result: List[float] = [0.0] * n
    prev = 0.0
    for i in range(n):
        prev = prev - prev / length + values[i]
        result[i] = prev
    return result


def _true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float]) -> List[float]:
    """max(high-low, |high-close_prev|, |low-close_prev|); close_prev = 0 on bar 0 (nz default)."""
    n = len(close)
    tr: List[float] = [0.0] * n
    for i in range(n):
        prev_close = close[i - 1] if i > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
    return tr


def _average_true_range(
    high: Sequence[float], low: Sequence[float], close: Sequence[float], length: int
) -> List[float]:
    """Matches Pine's `ta.atr(length)` = RMA of True Range."""
    return _rma(_true_range(high, low, close), length)


def _volatility_filter(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    min_length: int,
    max_length: int,
    enabled: bool,
) -> List[bool]:
    """Pine: `useVolatilityFilter ? ta.atr(minLength) > ta.atr(maxLength) : true`."""
    n = len(close)
    if not enabled:
        return [True] * n
    recent_atr = _average_true_range(high, low, close, min_length)
    historical_atr = _average_true_range(high, low, close, max_length)
    # NaN comparisons are False in Python, which correctly fails the filter
    # during the ATR warm-up period without any special-casing.
    return [r > h for r, h in zip(recent_atr, historical_atr)]


def _regime_filter(
    regime_source: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    threshold: float,
    enabled: bool,
) -> List[bool]:
    """
    Matches MLExtensions' `regime_filter`: an adaptive (KLMF) trend line
    whose bar-to-bar slope is compared to its own 200-bar EMA. `regime_source`
    is `ohlc4` in the original script, NOT `close` -- callers must pass ohlc4.
    """
    n = len(regime_source)
    if not enabled:
        return [True] * n

    value1 = 0.0
    value2 = 0.0
    klmf = 0.0
    abs_curve_slope: List[float] = [0.0] * n

    for i in range(n):
        src = regime_source[i]
        src_prev = regime_source[i - 1] if i > 0 else src
        value1 = 0.2 * (src - src_prev) + 0.8 * value1
        value2 = 0.1 * (high[i] - low[i]) + 0.8 * value2
        omega = abs(value1 / value2) if value2 != 0 else 0.0
        alpha = (-(omega**2) + math.sqrt(omega**4 + 16 * omega**2)) / 8
        new_klmf = alpha * src + (1 - alpha) * klmf
        abs_curve_slope[i] = abs(new_klmf - klmf)
        klmf = new_klmf

    exp_avg_abs_curve_slope = _ema_series(abs_curve_slope, 200)

    result: List[bool] = [False] * n
    for i in range(n):
        denom = exp_avg_abs_curve_slope[i]
        if math.isnan(denom) or denom == 0:
            result[i] = False  # EMA(200) warm-up not complete yet.
            continue
        normalized_slope_decline = (abs_curve_slope[i] - denom) / denom
        result[i] = normalized_slope_decline >= threshold
    return result


def _adx_filter(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    length: int,
    threshold: float,
    enabled: bool,
) -> List[bool]:
    """Matches MLExtensions' `filter_adx`: classic DI+/DI-/DX/ADX, ADX > threshold."""
    n = len(close)
    if not enabled:
        return [True] * n

    tr: List[float] = [0.0] * n
    dm_plus: List[float] = [0.0] * n
    dm_minus: List[float] = [0.0] * n
    for i in range(n):
        prev_high = high[i - 1] if i > 0 else 0.0
        prev_low = low[i - 1] if i > 0 else 0.0
        prev_close = close[i - 1] if i > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
        up_move = high[i] - prev_high
        down_move = prev_low - low[i]
        dm_plus[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    tr_smooth = _wilder_cumulative_smooth(tr, length)
    dm_plus_smooth = _wilder_cumulative_smooth(dm_plus, length)
    dm_minus_smooth = _wilder_cumulative_smooth(dm_minus, length)

    di_plus = [(p / t * 100) if t != 0 else 0.0 for p, t in zip(dm_plus_smooth, tr_smooth)]
    di_minus = [(m / t * 100) if t != 0 else 0.0 for m, t in zip(dm_minus_smooth, tr_smooth)]
    dx = [
        (abs(p - m) / (p + m) * 100) if (p + m) != 0 else 0.0
        for p, m in zip(di_plus, di_minus)
    ]
    adx = _rma(dx, length)

    return [(a > threshold) if not math.isnan(a) else False for a in adx]


def _weighted_kernel_regression(
    source: Sequence[float], weight_fn: Callable[[int], float], window: int
) -> List[float]:
    """
    Shared sliding-window weighted average behind both kernels:
    yhat[t] = sum_i w(i)*source[t-i] / sum_i w(i), i = 0..window-1 (clipped
    at the start of the series). `window` is derived to match Pine's
    `for i = 0 to array.size(array.from(_src)) + startAtBar` loop, where
    `array.from(_src)` on a scalar always yields a 1-element array -- i.e.
    the window is fixed at `startAtBar + 2`, not the full chart length.
    """
    n = len(source)
    weights = [weight_fn(i) for i in range(window)]
    result: List[float] = [float("nan")] * n
    for t in range(n):
        max_i = min(window - 1, t)
        numerator = 0.0
        denominator = 0.0
        for i in range(max_i + 1):
            w = weights[i]
            numerator += source[t - i] * w
            denominator += w
        result[t] = numerator / denominator if denominator != 0 else source[t]
    return result


def _rational_quadratic_kernel(
    source: Sequence[float], lookback: int, relative_weight: float, start_at_bar: int
) -> List[float]:
    """Matches KernelFunctions' `rationalQuadratic`."""
    window = start_at_bar + 2

    def weight(i: int) -> float:
        return (1 + (i**2) / ((lookback**2) * 2 * relative_weight)) ** (-relative_weight)

    return _weighted_kernel_regression(source, weight, window)


def _gaussian_kernel(source: Sequence[float], lookback: int, start_at_bar: int) -> List[float]:
    """Matches KernelFunctions' `gaussian`."""
    window = start_at_bar + 2

    def weight(i: int) -> float:
        return math.exp(-(i**2) / (2 * (lookback**2)))

    return _weighted_kernel_regression(source, weight, window)


def _kernel_trend_filters(
    source: Sequence[float], settings: DecisionEngineSettings
) -> Tuple[List[bool], List[bool]]:
    """
    Matches the main script's `isBullish`/`isBearish`: either the Rational
    Quadratic line's own slope (`useKernelSmoothing=false`, the default), or
    a crossover-style comparison against the lagged Gaussian line
    (`useKernelSmoothing=true`).
    """
    n = len(source)
    if not settings.use_kernel_filter:
        return [True] * n, [True] * n

    yhat1 = _rational_quadratic_kernel(
        source, settings.kernel_lookback, settings.kernel_relative_weight,
        settings.kernel_regression_level,
    )
    yhat2 = _gaussian_kernel(
        source, settings.kernel_lookback - settings.kernel_lag, settings.kernel_regression_level
    )

    is_bullish: List[bool] = [True] * n
    is_bearish: List[bool] = [True] * n
    for t in range(n):
        if settings.use_kernel_smoothing:
            is_bullish[t] = yhat2[t] >= yhat1[t]
            is_bearish[t] = yhat2[t] <= yhat1[t]
        else:
            previous = yhat1[t - 1] if t > 0 else yhat1[t]
            is_bullish[t] = previous < yhat1[t]
            is_bearish[t] = previous > yhat1[t]
    return is_bullish, is_bearish


def _ema_trend_filter(
    close: Sequence[float], period: int, enabled: bool
) -> Tuple[List[bool], List[bool]]:
    """Matches `isEmaUptrend`/`isEmaDowntrend` (no-op `True, True` when disabled, as in Pine)."""
    n = len(close)
    if not enabled:
        return [True] * n, [True] * n
    ema = _ema_series(close, period)
    is_up = [(close[i] > ema[i]) if not math.isnan(ema[i]) else False for i in range(n)]
    is_down = [(close[i] < ema[i]) if not math.isnan(ema[i]) else False for i in range(n)]
    return is_up, is_down


def _sma_trend_filter(
    close: Sequence[float], period: int, enabled: bool
) -> Tuple[List[bool], List[bool]]:
    """Matches `isSmaUptrend`/`isSmaDowntrend` (no-op `True, True` when disabled, as in Pine)."""
    n = len(close)
    if not enabled:
        return [True] * n, [True] * n
    sma = _sma_series(close, period)
    is_up = [(close[i] > sma[i]) if not math.isnan(sma[i]) else False for i in range(n)]
    is_down = [(close[i] < sma[i]) if not math.isnan(sma[i]) else False for i in range(n)]
    return is_up, is_down


class LorentzianStrategy:
    """
    Approximate K-Nearest-Neighbors classifier using Lorentzian distance
    over a configurable set of normalized indicator features.

    Feature engineering, distance computation, and the neighbor vote mirror
    the original indicator's "Core ML Logic" section. `_run_decision_engine`
    additionally replays the original's post-prediction decision logic
    (volatility/regime/ADX filters, the sticky `signal` state machine, and
    kernel-regression/EMA/SMA confirmation), so `PredictionResult.signal`
    matches the original indicator's actual BUY/SELL/HOLD behavior rather
    than just the sign of the raw vote. Position/exit management
    (`endLongTrade`/`endShortTrade`) remains out of scope, since `Direction`
    has no fourth state to represent an exit distinct from NEUTRAL.

    Indicator instances are injected (Dependency Injection) rather than
    constructed internally, and this class never recomputes or duplicates
    indicator math -- it only calls each indicator's existing `calculate_*`
    method and combines the outputs.
    """

    _FEATURE_NAMES: Tuple[str, ...] = ("rsi", "adx", "cci", "wt")

    def __init__(
        self,
        rsi: RSIIndicator,
        adx: ADXIndicator,
        cci: CCIIndicator,
        wt: WTIndicator,
        settings: Optional[LorentzianSettings] = None,
        decision_settings: Optional[DecisionEngineSettings] = None,
        source_extractor: PriceExtractor = _default_close_extractor,
    ) -> None:
        self._rsi = rsi
        self._adx = adx
        self._cci = cci
        self._wt = wt
        self._settings = settings or LorentzianSettings()
        self._decision_settings = decision_settings or DecisionEngineSettings()
        self._extract_source_prices = source_extractor

    def analyze(self, candles: Sequence[Any]) -> PredictionResult:
        """
        Run the full pipeline for the most recent bar in `candles`.

        `signal` reflects the original indicator's full post-prediction
        decision logic (filters, sticky state, kernel/EMA/SMA confirmation
        -- see `_run_decision_engine`), not just the sign of the raw vote.
        """
        if not candles:
            raise ValueError("Cannot analyze an empty candle sequence.")

        dataset = self._build_feature_vector(candles)
        return self._predict(dataset, candles)

    def _build_feature_vector(self, candles: Sequence[Any]) -> TrainingDataset:
        """
        Build the chronological (features, label) dataset for `candles`.

        Returns one `FeatureVector` and one `Direction` label per candle,
        aligned 1:1. This assumes injected indicators return a value (NaN
        during their own warm-up) for every input candle rather than a
        truncated series -- `FeatureVector.is_valid` is how downstream code
        recognizes and skips warm-up bars.
        """
        feature_series = tuple(self._compute_feature_series(candles))
        source_prices = self._extract_source_prices(candles)
        labels = tuple(generate_labels(source_prices, self._settings.label_horizon))
        return TrainingDataset(feature_series=feature_series, labels=labels)

    def _compute_feature_series(self, candles: Sequence[Any]) -> List[FeatureVector]:
        """
        Delegate to the injected indicators and zip their outputs into a
        feature-vector series.

        Adding a new indicator to the model means adding one entry to
        `indicator_outputs` / `_FEATURE_NAMES` here -- nothing else in this
        module needs to change.
        """

        rsi_out = self._rsi.calculate_rsi(candles) or {}
        adx_out = self._adx.calculate_adx(candles) or {}
        cci_out = self._cci.calculate_cci(candles) or {}
        wt_out = self._wt.calculate_wt(candles) or {}

        rsi_series = rsi_out.get("series") if rsi_out.get("series") is not None else np.full(len(candles), np.nan)
        adx_series = adx_out.get("series") if adx_out.get("series") is not None else np.full(len(candles), np.nan)
        cci_series = cci_out.get("series") if cci_out.get("series") is not None else np.full(len(candles), np.nan)
        wt_series = wt_out.get("series") if wt_out.get("series") is not None else np.full(len(candles), np.nan)

        indicator_outputs = (
            rsi_series,
            adx_series,
            cci_series,
            wt_series,
        )

        lengths = {len(output) for output in indicator_outputs}
        if len(lengths) > 1:
            raise ValueError(
                "Indicator outputs must all be computed over the same "
                f"candle series; got lengths {[len(o) for o in indicator_outputs]} "
                f"for features {self._FEATURE_NAMES}."
            )

        return [
            FeatureVector(
                values=tuple(float(value) for value in row),
                feature_names=self._FEATURE_NAMES,
            )
            for row in zip(*indicator_outputs)
        ]

    def _compute_prediction_series(
        self, dataset: TrainingDataset
    ) -> Tuple[List[int], List[int]]:
        """
        Replay the unchanged `approximate_nearest_neighbors` vote for every
        bar in `dataset`, not just the most recent one.

        Why this is necessary: the original indicator's `signal` is sticky
        (`signal[t] = ... or signal[t-1]`), so whether *today* counts as a
        fresh BUY/SELL transition depends on the entire history of votes
        and filters, not just today's vote. `_run_decision_engine` needs
        that full series to replay the state machine correctly. This does
        NOT change how any individual vote is computed --
        `approximate_nearest_neighbors` is called exactly as before, for
        each bar -- only how many times it's called.

        Returns (votes_for_every_bar, neighbor_labels_for_the_last_bar).
        """
        votes: List[int] = []
        last_bar_neighbor_labels: List[int] = []
        last_index = len(dataset) - 1

        for index in range(len(dataset)):
            candidate = dataset.feature_series[index]
            if not candidate.is_valid:
                votes.append(0)
                continue

            history = dataset.feature_series[:index]
            history_labels = dataset.labels[:index]
            vote, neighbor_labels = approximate_nearest_neighbors(
                candidate, history, history_labels, self._settings
            )
            votes.append(vote)
            if index == last_index:
                last_bar_neighbor_labels = neighbor_labels

        return votes, last_bar_neighbor_labels

    def _run_decision_engine(self, votes: Sequence[int], candles: Sequence[Any]) -> Direction:
        """
        Replay the original indicator's post-prediction decision logic and
        return the resulting signal for the most recent bar.

        `votes[t]` is the unmodified Lorentzian KNN vote for bar t.
        Everything below operates strictly *after* that vote, mirroring the
        original script's "Prediction Filters" and "Entries and Exits"
        sections:

            filter_all[t]  = volatility_pass[t] and regime_pass[t] and adx_pass[t]
            signal[t]      = LONG  if votes[t] > 0 and filter_all[t]
                           = SHORT if votes[t] < 0 and filter_all[t]
                           = signal[t-1] otherwise   (sticky: a failed
                             filter freezes the signal, it does not reset it)
            startLongTrade  = signal flips to LONG on bar t, AND the EMA/SMA
                              trend filters agree, AND the kernel-regression
                              filter agrees, all on that same bar
            startShortTrade = mirror of the above for SHORT

        Only `startLongTrade`/`startShortTrade` on the *last* bar determine
        the returned `Direction` -- this is exactly what the original
        script's BUY/SELL plot markers are driven by.
        """
        if len(votes) != len(candles):
            raise ValueError(
                f"votes and candles must have equal length (got {len(votes)} "
                f"and {len(candles)})."
            )

        settings = self._decision_settings
        n = len(candles)

        open_ = _extract_price_field(candles, "open")
        high = _extract_price_field(candles, "high")
        low = _extract_price_field(candles, "low")
        close = self._extract_source_prices(candles)
        ohlc4 = [(o + h + l + c) / 4 for o, h, l, c in zip(open_, high, low, close)]

        volatility_pass = _volatility_filter(
            high, low, close,
            settings.volatility_min_length, settings.volatility_max_length,
            settings.use_volatility_filter,
        )
        regime_pass = _regime_filter(
            ohlc4, high, low, settings.regime_threshold, settings.use_regime_filter
        )
        adx_pass = _adx_filter(
            high, low, close, settings.adx_length, settings.adx_threshold,
            settings.use_adx_filter,
        )

        is_ema_uptrend, is_ema_downtrend = _ema_trend_filter(
            close, settings.ema_period, settings.use_ema_filter
        )
        is_sma_uptrend, is_sma_downtrend = _sma_trend_filter(
            close, settings.sma_period, settings.use_sma_filter
        )
        is_bullish, is_bearish = _kernel_trend_filters(close, settings)

        previous_signal = Direction.NEUTRAL
        final_signal = Direction.NEUTRAL

        for t in range(n):
            filter_all = volatility_pass[t] and regime_pass[t] and adx_pass[t]
            vote = votes[t]

            if vote > 0 and filter_all:
                current_signal = Direction.LONG
            elif vote < 0 and filter_all:
                current_signal = Direction.SHORT
            else:
                current_signal = previous_signal

            is_transition = current_signal != previous_signal

            is_new_buy_signal = (
                current_signal == Direction.LONG
                and is_ema_uptrend[t]
                and is_sma_uptrend[t]
                and is_transition
            )
            is_new_sell_signal = (
                current_signal == Direction.SHORT
                and is_ema_downtrend[t]
                and is_sma_downtrend[t]
                and is_transition
            )

            start_long_trade = (
                is_new_buy_signal and is_bullish[t] and is_ema_uptrend[t] and is_sma_uptrend[t]
            )
            start_short_trade = (
                is_new_sell_signal and is_bearish[t] and is_ema_downtrend[t] and is_sma_downtrend[t]
            )

            if t == n - 1:
                if start_long_trade:
                    final_signal = Direction.LONG
                elif start_short_trade:
                    final_signal = Direction.SHORT
                else:
                    final_signal = Direction.NEUTRAL

            previous_signal = current_signal

        return final_signal

    def _predict(self, dataset: TrainingDataset, candles: Sequence[Any]) -> PredictionResult:
        """
        Classify the most recent bar in `dataset`.

        `prediction`/`neighbors_labels` come straight from the unmodified
        Lorentzian vote for the current bar. `signal` is the result of
        replaying the original script's filter + sticky-state-machine
        decision logic in `_run_decision_engine` -- that is what actually
        drives BUY/SELL/HOLD in the original indicator, not the sign of
        `prediction` alone.
        """
        if len(dataset) == 0:
            return PredictionResult(prediction=0, signal=Direction.NEUTRAL, neighbors_labels=[])

        current_index = len(dataset) - 1
        current = dataset.feature_series[current_index]

        if not current.is_valid:
            # Indicators for the current bar haven't finished warming up.
            return PredictionResult(prediction=0, signal=Direction.NEUTRAL, neighbors_labels=[])

        votes, neighbor_labels = self._compute_prediction_series(dataset)
        raw_prediction = votes[current_index]
        final_signal = self._run_decision_engine(votes, candles)

        return PredictionResult(
            prediction=raw_prediction,
            signal=final_signal,
            neighbors_labels=neighbor_labels,
        )

    @staticmethod
    def _classify(prediction: int) -> Direction:
        """Map a signed vote total to a direction: >0 LONG, <0 SHORT, else NEUTRAL."""
        if prediction > 0:
            return Direction.LONG
        if prediction < 0:
            return Direction.SHORT
        return Direction.NEUTRAL
    
