"""Tests for FundamentalProcess (Ornstein-Uhlenbeck)."""

import numpy as np
import pytest

from market_abm.fundamental import FundamentalProcess


class TestFundamentalProcess:
    def test_initial_value(self):
        fp = FundamentalProcess(100.0, kappa=0.01, mu=100.0, sigma=0.5)
        assert fp.value == 100.0
        assert fp.history == [100.0]

    def test_step_updates_value(self):
        rng = np.random.default_rng(42)
        fp = FundamentalProcess(100.0, kappa=0.01, mu=100.0, sigma=0.5, rng=rng)
        v1 = fp.step()
        assert v1 != 100.0
        assert fp.value == v1
        assert len(fp.history) == 2

    def test_mean_reversion(self):
        rng = np.random.default_rng(42)
        fp = FundamentalProcess(200.0, kappa=0.1, mu=100.0, sigma=0.1, rng=rng)
        for _ in range(500):
            fp.step()
        assert abs(fp.value - 100.0) < abs(200.0 - 100.0)

    def test_zero_volatility_converges(self):
        fp = FundamentalProcess(200.0, kappa=0.1, mu=100.0, sigma=0.0)
        for _ in range(200):
            fp.step()
        assert fp.value == pytest.approx(100.0, abs=0.01)

    def test_zero_kappa_random_walk(self):
        rng = np.random.default_rng(42)
        fp = FundamentalProcess(100.0, kappa=0.0, mu=100.0, sigma=0.5, rng=rng)
        for _ in range(100):
            fp.step()
        assert fp.value != 100.0

    def test_apply_shock(self):
        fp = FundamentalProcess(100.0, kappa=0.01, mu=100.0, sigma=0.0)
        result = fp.apply_shock(0.10)
        assert result == pytest.approx(110.0)
        assert fp.value == pytest.approx(110.0)

    def test_apply_negative_shock(self):
        fp = FundamentalProcess(100.0, kappa=0.01, mu=100.0, sigma=0.0)
        fp.apply_shock(-0.20)
        assert fp.value == pytest.approx(80.0)

    def test_reproducibility(self):
        fp1 = FundamentalProcess(100.0, kappa=0.01, mu=100.0, sigma=0.5,
                                  rng=np.random.default_rng(99))
        fp2 = FundamentalProcess(100.0, kappa=0.01, mu=100.0, sigma=0.5,
                                  rng=np.random.default_rng(99))
        for _ in range(50):
            fp1.step()
            fp2.step()
        assert fp1.history == fp2.history
