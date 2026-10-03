"""
Meta-Labeling Gate: Second-Stage Machine Learning Decider (Layer-2 AI).

Acts as the 'Supreme Court' / Final Decider for candidate trading signals:
  1. Ingests OHLCV and extracts the 8 core microstructure features.
  2. Loads pre-trained LightGBM Meta-Labeling Classifier (from models/).
  3. Predicts P(Win > Fees | Features) causally at bar index t.
  4. Applies calibrated probability threshold (default: 0.55).
  5. Returns (is_approved: bool, prob_win: float).

Authors: Adaptive Trading System Quant Team
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import talib

logger = logging.getLogger("MetaLabeler")

DEFAULT_FEATURE_COLUMNS: List[str] = [
    "rsi",
    "cci",
    "adx",
    "wt_diff",
    "atr",
    "volatility_ratio",
    "normalized_atr",
    "lorentzian_signal",
]


class PurgedTimeSeriesSplit:
    """
    Purged & Embargoed Time-Series Cross-Validation Splitter.
    Prevents information leakage between chronological folds by enforcing a purge buffer
    matching the maximum trade holding horizon.
    """

    def __init__(self, n_splits: int = 4, purge_buffer: int = 5) -> None:
        self.n_splits = n_splits
        self.purge_buffer = purge_buffer

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[Any] = None
    ):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            test_start = min(train_end + self.purge_buffer, n_samples)
            test_end = min((i + 1) * fold_size, n_samples)
            if test_start >= test_end:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_n_splits(
        self, X: Optional[Any] = None, y: Optional[Any] = None, groups: Optional[Any] = None
    ) -> int:
        return self.n_splits


@dataclass
class MetaLabelResult:
    """Evaluation result from the LightGBM Meta-Labeling Decider."""
    is_approved: bool
    prob_win: float
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""


class MetaLabelingGate:
    """
    LightGBM Meta-Labeling Decision Engine.
    Filters candidate trade signals by predicting whether forward trade
    outcomes will cover transaction fees with positive expectancy.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.55,
        coin: str = "btc",
        timeframe: str = "1h",
        models_dir: Union[str, Path] = "models",
    ) -> None:
        self.threshold = float(threshold)
        self.coin = coin.lower()
        self.timeframe = timeframe.lower()
        self.models_dir = Path(models_dir)
        self.feature_columns = list(DEFAULT_FEATURE_COLUMNS)

        self.model: Any = None
        self.resolved_model_path: Optional[Path] = None
        self.metadata: Dict[str, Any] = {}

        self._load_model(model_path)

    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        """Finds the best matching pre-trained LightGBM model artifact."""
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                return p
            # Relative to project root
            proj_root = Path(__file__).resolve().parent.parent.parent
            cand = proj_root / explicit_path
            if cand.exists():
                return cand

        # Auto-discovery candidates in priority order:
        candidates = [
            self.models_dir / f"{self.coin}_{self.timeframe}_funnel_metalabeler.joblib",
            self.models_dir / f"{self.coin}_{self.timeframe}_dedication_lgbm.joblib",
            self.models_dir / f"{self.coin}_30m_dedication_lgbm.joblib",
            self.models_dir / f"{self.coin}_15m_dedication_lgbm.joblib",
            self.models_dir / f"{self.coin}_5m_dedication_lgbm.joblib",
            self.models_dir / "btc_15m_dedication_lgbm.joblib",
        ]

        proj_root = Path(__file__).resolve().parent.parent.parent
        for c in candidates:
            if c.exists():
                return c
            c_root = proj_root / c
            if c_root.exists():
                return c_root

        # Scan models directory for any funnel_metalabeler or dedication_lgbm artifact
        if self.models_dir.exists():
            matches = list(self.models_dir.glob(f"*{self.coin}*{self.timeframe}*metalabeler*.joblib"))
            if not matches:
                matches = list(self.models_dir.glob("*metalabeler*.joblib"))
            if not matches:
                matches = list(self.models_dir.glob("*dedication_lgbm*.joblib"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"No trained LightGBM meta-labeling model found in '{self.models_dir}' "
            f"for {self.coin} {self.timeframe}."
        )

    def _load_model(self, model_path: Optional[Union[str, Path]]) -> None:
        """Loads model weights and metadata."""
        try:
            resolved = self._resolve_model_path(model_path)
            self.resolved_model_path = resolved
            self.model = joblib.load(resolved)
            logger.info(f"[MetaLabeler] Loaded LightGBM model from {resolved}")

            # Try to load companion metadata
            meta_path = resolved.with_name(resolved.name.replace(".joblib", "_metadata.json"))
            if not meta_path.exists():
                meta_path = resolved.with_name(resolved.name.replace("_dedication_lgbm.joblib", "_metadata.json"))
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    if "feature_columns" in self.metadata:
                        self.feature_columns = self.metadata["feature_columns"]
        except Exception as e:
            logger.warning(f"[MetaLabeler] Failed to load LightGBM model: {e}")
            self.model = None

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Layer-1 technical indicators and microstructure features
        required by the LightGBM Meta-Model. Causal and fully vectorized.
        """
        df = df.copy()
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)

        # 1. Momentum & Oscillators
        if "rsi" not in df.columns:
            df["rsi"] = talib.RSI(close, timeperiod=14)
        if "cci" not in df.columns:
            df["cci"] = talib.CCI(high, low, close, timeperiod=20)
        if "adx" not in df.columns:
            df["adx"] = talib.ADX(high, low, close, timeperiod=14)

        # 2. WaveTrend Difference (wt_diff)
        if "wt_diff" not in df.columns:
            hlc3 = (high + low + close) / 3.0
            esa = talib.EMA(hlc3, timeperiod=10)
            d = talib.EMA(np.abs(hlc3 - esa), timeperiod=10)
            divisor = 0.015 * d
            with np.errstate(divide="ignore", invalid="ignore"):
                ci = np.where(divisor != 0, (hlc3 - esa) / divisor, 0.0)
            wt1 = talib.EMA(ci, timeperiod=21)
            wt2 = talib.SMA(wt1, timeperiod=4)
            df["wt_diff"] = wt1 - wt2

        # 3. Volatility & Ratio
        if "atr" not in df.columns:
            df["atr"] = talib.ATR(high, low, close, timeperiod=14)
        if "volatility_ratio" not in df.columns:
            atr_sma50 = talib.SMA(df["atr"].to_numpy(dtype=np.float64), timeperiod=50)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["volatility_ratio"] = np.where(
                    (atr_sma50 != 0) & (~np.isnan(atr_sma50)), df["atr"] / atr_sma50, 1.0
                )
        if "normalized_atr" not in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["normalized_atr"] = np.where(close != 0, df["atr"] / close, 0.0)

        # 4. Lorentzian Signal (Directional bias: +1 for Long)
        if "lorentzian_signal" not in df.columns:
            df["lorentzian_signal"] = 1.0

        return df

    def evaluate(self, df: pd.DataFrame, idx: int) -> MetaLabelResult:
        """
        Evaluates a candidate trade signal at bar index `idx` using LightGBM.
        Strictly causal: inspects only bar `idx`.

        Returns:
            MetaLabelResult(is_approved, prob_win, threshold, features, model_name)
        """
        if self.model is None:
            # Fallback if model could not be loaded: approve by default
            return MetaLabelResult(
                is_approved=True,
                prob_win=0.50,
                threshold=self.threshold,
                model_name="NONE (FALLBACK APPROVED)",
            )

        bar = df.iloc[idx]
        feat_dict: Dict[str, float] = {}

        # Build feature vector
        vector: List[float] = []
        for col in self.feature_columns:
            val = float(bar[col]) if col in bar and not np.isnan(bar[col]) else 0.0
            feat_dict[col] = val
            vector.append(val)

        X = np.array([vector], dtype=np.float64)

        try:
            if hasattr(self.model, "predict_proba"):
                raw_prob = self.model.predict_proba(X)
                prob_win = float(raw_prob[0, 1]) if raw_prob.ndim == 2 and raw_prob.shape[1] > 1 else float(raw_prob[0])
            else:
                raw_pred = self.model.predict(X)
                prob_win = float(raw_pred[0])
        except Exception as e:
            logger.error(f"[MetaLabeler] Inference error at index {idx}: {e}")
            prob_win = 0.50

        is_approved = bool(prob_win >= self.threshold)
        model_name = self.resolved_model_path.name if self.resolved_model_path else "LightGBM"

        return MetaLabelResult(
            is_approved=is_approved,
            prob_win=prob_win,
            threshold=self.threshold,
            features=feat_dict,
            model_name=model_name,
        )

    def evaluate_series(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized batch inference across an entire DataFrame.

        Returns:
            (is_approved_array: bool[], probs_array: float[])
        """
        n = len(df)
        if self.model is None or n == 0:
            return np.ones(n, dtype=bool), np.full(n, 0.50, dtype=np.float64)

        # Prepare feature matrix
        df_feat = self.extract_features(df)
        X = df_feat[self.feature_columns].to_numpy(dtype=np.float64)

        if hasattr(self.model, "predict_proba"):
            raw_prob = self.model.predict_proba(X)
            probs = raw_prob[:, 1] if raw_prob.ndim == 2 and raw_prob.shape[1] > 1 else raw_prob.ravel()
        else:
            probs = self.model.predict(X).ravel()

        is_approved = probs >= self.threshold
        return is_approved, probs
