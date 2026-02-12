"""Tests for FundamentalProcess."""

import numpy as np
import pytest

from market_abm.fundamental import FundamentalProcess


class TestFundamentalProcess:
    def test_initial_value(self):
        fp = FundamentalProcess(100.0, 0.0, 0.01)
        assert fp.value == 100.0
        assert fp.history == [100.0]

    def test_step_updates_value(self):
        rng = np.random.default_rng(42)
        fp = FundamentalProcess(100.0, 0.0, 0.01, rng=rng)
        v1 = fp.step()
        assert v1 != 100.0
        assert fp.value == v1
        assert len(fp.history) == 2

    def test_positive_values(self):
        """Geometric random walk should always produce positive values."""
        rng = np.random.default_rng(123)
        fp = FundamentalProcess(100.0, 0.0, 0.05, rng=rng)
        for _ in range(1000):
            fp.step()
        assert all(v > 0 for v in fp.history)

    def test_drift(self):
        """With positive drift and no volatility, value should increase."""
        rng = np.random.default_rng(0)
        fp = FundamentalProcess(100.0, 0.01, 0.0, rng=rng)
        for _ in range(100):
            fp.step()
        assert fp.value > 100.0

    def test_apply_shock(self):
        fp = FundamentalProcess(100.0, 0.0, 0.0)
        result = fp.apply_shock(0.10)
        assert result == pytest.approx(110.0)
        assert fp.value == pytest.approx(110.0)

    def test_apply_negative_shock(self):
        fp = FundamentalProcess(100.0, 0.0, 0.0)
        fp.apply_shock(-0.20)
        assert fp.value == pytest.approx(80.0)

    def test_log_value(self):
        fp = FundamentalProcess(100.0, 0.0, 0.0)
        assert fp.log_value == pytest.approx(np.log(100.0))

    def test_reproducibility(self):
        """Same seed should produce same trajectory."""
        fp1 = FundamentalProcess(100.0, 0.001, 0.01, rng=np.random.default_rng(99))
        fp2 = FundamentalProcess(100.0, 0.001, 0.01, rng=np.random.default_rng(99))
        for _ in range(50):
            fp1.step()
            fp2.step()
        assert fp1.history == fp2.history
