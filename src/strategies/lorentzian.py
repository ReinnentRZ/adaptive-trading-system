from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, List, Optional, Sequence, Tuple

from src.indicators.adx import ADXIndicator
from src.indicators.cci import CCIIndicator
from src.indicators.rsi import RSIIndicator
from src.indicators.wt import WTIndicator

__all__ = [
    "Direction",
    "FeatureVector",
    "TrainingDataset",
    "LorentzianSettings",
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

    neighbors_count: int = 8
    max_bars_back: int = 2000
    label_horizon: int = 4

    def __post_init__(self) -> None:
        if self.neighbors_count < 1:
            raise ValueError("neighbors_count must be >= 1.")
        if self.max_bars_back < 1:
            raise ValueError("max_bars_back must be >= 1.")
        if self.label_horizon < 1:
            raise ValueError("label_horizon must be >= 1.")


@dataclass(frozen=True)
class PredictionResult:
    """
    Outcome of the model for a single bar.

    `prediction` is the signed sum of the selected neighbors' labels (range
    -neighbors_count..+neighbors_count). Its sign drives `signal`; its
    magnitude is a usable confidence proxy -- how many of the k neighbors
    agreed on direction.
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


class LorentzianStrategy:
    """
    Approximate K-Nearest-Neighbors classifier using Lorentzian distance
    over a configurable set of normalized indicator features.

    Scope: feature engineering, distance computation, neighbor search, and
    raw direction classification only -- this mirrors the "Core ML Logic"
    section of the original indicator. Trade-timing filters (volatility,
    regime, ADX, kernel regression, EMA/SMA trend filters) and position/exit
    management are intentionally out of scope and belong in separate,
    composable components that consume `PredictionResult`.

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
        source_extractor: PriceExtractor = _default_close_extractor,
    ) -> None:
        self._rsi = rsi
        self._adx = adx
        self._cci = cci
        self._wt = wt
        self._settings = settings or LorentzianSettings()
        self._extract_source_prices = source_extractor

    def analyze(self, candles: Sequence[Any]) -> PredictionResult:
        """
        Run the full pipeline for the most recent bar in `candles`.

        Designed to be called once per new bar (live), or repeatedly with a
        growing candle window (backtesting); see the module docstring for
        the complexity implications of the latter.
        """
        if not candles:
            raise ValueError("Cannot analyze an empty candle sequence.")

        dataset = self._build_feature_vector(candles)
        return self._predict(dataset)

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

    # def _predict(self, dataset: TrainingDataset) -> PredictionResult:
    #     """Classify the most recent bar in `dataset` against all prior bars."""
    #     if len(dataset) == 0:
    #         return PredictionResult(prediction=0, signal=Direction.NEUTRAL, neighbors_labels=[])

    #     current_index = len(dataset) - 1
    #     current = dataset.feature_series[current_index]

    #     if not current.is_valid:
    #         # Indicators for the current bar haven't finished warming up.
    #         return PredictionResult(prediction=0, signal=Direction.NEUTRAL, neighbors_labels=[])

    #     history = dataset.feature_series[:current_index]
    #     history_labels = dataset.labels[:current_index]

    #     raw_prediction, list_label_tetangga = approximate_nearest_neighbors(
    #         current, history, history_labels, self._settings
    #     )
    #     return PredictionResult(
    #         prediction=raw_prediction, signal=self._classify(raw_prediction), neighbors_labels=list_label_tetangga
    #     )

    def _predict(self, dataset: TrainingDataset) -> PredictionResult:
        """Classify the most recent bar in `dataset` against all prior bars."""
        
        if len(dataset) == 0:
            print("🚨 DEBUG ALARM: Keluar karena dataset kosong (len = 0)")
            return PredictionResult(prediction=0, signal=Direction.NEUTRAL, neighbors_labels=[])

        current_index = len(dataset) - 1
        current = dataset.feature_series[current_index]

        # 🎯 ALARM 1: Apakah nyangkut di Pintu Darurat?
        if not current.is_valid:
            print("🚨 DEBUG ALARM: Keluar karena current.is_valid = FALSE (Indikator menit ini dianggap cacat/NaN)")
            print(f"🚨 BONGKAR ISI CURRENT: {vars(current)}")
            return PredictionResult(prediction=0, signal=Direction.NEUTRAL, neighbors_labels=[])

        history = dataset.feature_series[:current_index]
        history_labels = dataset.labels[:current_index]

        raw_prediction, list_label_tetangga = approximate_nearest_neighbors(
            current, history, history_labels, self._settings
        )

        # 🎯 ALARM 2: Apakah lolos masuk, tapi gagal dapet tetangga?
        if len(list_label_tetangga) == 0:
            print("🚨 DEBUG ALARM: Masuk ke pencarian, tapi 1999 tetangga masa lalu gugur semua kena filter 'continue'!")

        return PredictionResult(
            prediction=raw_prediction, 
            signal=self._classify(raw_prediction), 
            neighbors_labels=list_label_tetangga
        )

    @staticmethod
    def _classify(prediction: int) -> Direction:
        """Map a signed vote total to a direction: >0 LONG, <0 SHORT, else NEUTRAL."""
        if prediction > 0:
            return Direction.LONG
        if prediction < 0:
            return Direction.SHORT
        return Direction.NEUTRAL