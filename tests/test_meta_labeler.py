"""
Unit Tests for Layer-2 AI Meta-Labeling Decider (LightGBM Gate).
"""

import numpy as np
import pandas as pd

from src.config.strategy import RegimeFunnelConfig
from src.strategies.meta_labeler import MetaLabelResult, MetaLabelingGate
from src.strategies.regime_funnel import RegimeFunnelStrategy


class TestMetaLabelingGate:
    """Tests for MetaLabelingGate."""

    def test_meta_labeler_initialization_and_loading(self):
        gate = MetaLabelingGate(coin="btc", timeframe="15m", threshold=0.55)
        assert gate.threshold == 0.55
        assert gate.model is not None
        assert len(gate.feature_columns) == 8

    def test_meta_labeler_extract_features(self):
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "datetime": dates,
            "open": np.linspace(50000, 52000, n),
            "high": np.linspace(50100, 52100, n),
            "low": np.linspace(49900, 51900, n),
            "close": np.linspace(50000, 52000, n),
            "volume": np.full(n, 1000.0),
        })

        df_feat = MetaLabelingGate.extract_features(df)
        required_cols = ["rsi", "cci", "adx", "wt_diff", "atr", "volatility_ratio", "normalized_atr", "lorentzian_signal"]
        for col in required_cols:
            assert col in df_feat.columns

    def test_meta_labeler_single_evaluation(self):
        gate = MetaLabelingGate(coin="btc", timeframe="15m", threshold=0.55)
        n = 60
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "datetime": dates,
            "open": np.linspace(50000, 52000, n),
            "high": np.linspace(50100, 52100, n),
            "low": np.linspace(49900, 51900, n),
            "close": np.linspace(50000, 52000, n),
            "volume": np.full(n, 1000.0),
        })
        df = MetaLabelingGate.extract_features(df)

        res = gate.evaluate(df, idx=n - 1)
        assert isinstance(res, MetaLabelResult)
        assert 0.0 <= res.prob_win <= 1.0
        assert isinstance(res.is_approved, bool)
        assert res.threshold == 0.55

    def test_regime_funnel_with_meta_labeler_integration(self):
        cfg = RegimeFunnelConfig(
            use_meta_labeler=True,
            meta_label_threshold=0.50,
        )
        strategy = RegimeFunnelStrategy(config=cfg)
        assert strategy.meta_gate is not None

        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "datetime": dates,
            "open": np.linspace(60000, 65000, n),
            "high": np.linspace(60200, 65200, n),
            "low": np.linspace(59800, 64800, n),
            "close": np.linspace(60000, 65000, n),
            "volume": np.full(n, 1000.0),
        })

        df_out = strategy.compute_indicators(df)
        assert "wt_diff" in df_out.columns
        assert "volatility_ratio" in df_out.columns

        passed, details = strategy.evaluate_gates(
            df=df_out,
            idx=n - 1,
            current_state=0,
            state_age=1,
            traded_in_episode=False,
        )
        assert "pilar_meta" in details
        assert "meta_prob" in details
