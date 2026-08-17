from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorResult:
    """
    Base dataclass for indicator results. Supports both attribute access and
    dictionary-style access for backward compatibility.
    """
    current: Optional[float]
    smoothing: Optional[float]
    series: np.ndarray

    def __getitem__(self, key: str) -> Any:
        key_map = {
            "rsi": "current",
            "rsi_smoothing": "smoothing",
            "adx": "current",
            "adx_smoothing": "smoothing",
            "cci": "current",
            "cci_smoothing": "smoothing",
            "wt1": "current",
            "wt2": "smoothing",
            "atr": "current",
            "current_atr": "current",
            "series": "series"
        }
        attr = key_map.get(key, key)
        if hasattr(self, attr):
            return getattr(self, attr)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass(frozen=True)
class RSIResult(IndicatorResult):
    @property
    def rsi(self) -> Optional[float]:
        return self.current

    @property
    def rsi_smoothing(self) -> Optional[float]:
        return self.smoothing


@dataclass(frozen=True)
class ADXResult(IndicatorResult):
    @property
    def adx(self) -> Optional[float]:
        return self.current

    @property
    def adx_smoothing(self) -> Optional[float]:
        return self.smoothing


@dataclass(frozen=True)
class CCIResult(IndicatorResult):
    @property
    def cci(self) -> Optional[float]:
        return self.current

    @property
    def cci_smoothing(self) -> Optional[float]:
        return self.smoothing


@dataclass(frozen=True)
class WTResult(IndicatorResult):
    @property
    def wt1(self) -> Optional[float]:
        return self.current

    @property
    def wt2(self) -> Optional[float]:
        return self.smoothing


@dataclass(frozen=True)
class ATRResult(IndicatorResult):
    @property
    def current_atr(self) -> Optional[float]:
        return self.current


class BaseIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate indicator values.

        Args:
            data: pd.DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  of type float64.

        Returns:
            IndicatorResult (or a subclass) containing current, smoothing, and series.
        """
        pass

    def _prepare_data(self, data: Union[pd.DataFrame, Sequence[dict]]) -> pd.DataFrame:
        """
        Validates and converts input data into a standardized pd.DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, (list, tuple)):
            df = pd.DataFrame(data)
        else:
            raise TypeError("Input data must be a pandas DataFrame or a sequence of dicts.")

        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}' in input data.")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        return df
