"""Tests of the photopeak fitter and the resolution metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import least_squares

from dcalib import metrics


def _gauss_sample(n: int, mu: float, sigma: float, seed: int, bkg: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bkg = int(bkg * n)
    return np.concatenate([rng.normal(mu, sigma, n - n_bkg), rng.uniform(0.6, 1.3, n_bkg)])


class TestHistogram:
    def test_histogram_edges_and_nan(self) -> None:
        centres, counts = metrics.histogram(
            [0.8, 0.8024, 0.8026, 1.1499, 1.15, np.nan], 0.8, 1.15, 0.0025
        )
        assert len(centres) == 140 and centres[0] == pytest.approx(0.80125)
        assert counts[0] == 2 and counts[1] == 1 and counts[-1] == 1 and counts.sum() == 4

    def test_smooth_preserves_sum_inside(self) -> None:
        c = np.zeros(20)
        c[10] = 9.0
        s = metrics.smooth(c)
        assert s.sum() == pytest.approx(9.0) and s[10] == pytest.approx(3.0)

    def test_deviance(self) -> None:
        d = metrics.poisson_deviance([0.0, 5.0, 5.0, 5.0], [2.0, 5.0, 5.0 * (1 + 1e-6), 9.0])
        assert d[0] == pytest.approx(4.0) and d[1] == 0.0
        assert 0.0 < d[2] < 1e-10 and d[3] == pytest.approx(2 * (9 - 5 + 5 * math.log(5 / 9)))


class TestFitPhotopeak:
    def test_recovers_gaussian_on_background(self) -> None:
        fit = metrics.fit_photopeak(_gauss_sample(50_000, 0.99, 0.022, 1, bkg=0.2))
        assert fit.ok and fit.message == ""
        assert abs(fit.mu - 0.99) < 4 * fit.mu_err
        assert fit.sigma == pytest.approx(0.022, rel=0.03)
        assert fit.fwhm_pct == pytest.approx(100 * 2.3548 * fit.sigma / fit.mu, rel=1e-4)
        assert fit.window[0] < fit.mu < fit.window[1] and fit.chi2ndf < 2.0
        curve = fit.curve(np.array([fit.mu]))
        assert curve[0] == pytest.approx(fit.amplitude + np.mean(fit.background), rel=0.05)

    def test_error_calibration(self) -> None:
        pulls = []
        for seed in range(150):
            fit = metrics.fit_photopeak(_gauss_sample(800, 1.0, 0.024, seed, bkg=0.1))
            assert fit.ok
            pulls.append((fit.mu - 1.0) / fit.mu_err)
        # The asymmetric window biases a pure Gaussian by about +0.3 sigma (module doc).
        assert abs(np.mean(pulls)) < 0.5
        assert 0.7 < np.std(pulls) < 1.25

    def test_window_width(self) -> None:
        values = _gauss_sample(5000, 1.0, 0.02, 3)
        fit = metrics.fit_photopeak(values, seed_sigma=0.03)
        assert fit.window[1] - fit.window[0] == pytest.approx(5 * 0.03, abs=2 * 0.0025)
        adapted = metrics.fit_photopeak(values, seed_sigma=0.03, adapt_width=True)
        assert adapted.window[1] - adapted.window[0] == pytest.approx(5 * 0.02, abs=2 * 0.0025)

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (np.full(10, 1.0), "too few counts in the histogram"),
            (np.linspace(0.0, 0.5, 1000), "too few counts in the histogram"),
        ],
    )
    def test_failures(self, values: np.ndarray, message: str) -> None:
        fit = metrics.fit_photopeak(values)
        assert not fit.ok and fit.message == message
        assert math.isnan(fit.fwhm_pct) and np.isnan(fit.curve(np.array([1.0]))).all()

    def test_lm_matches_scipy_least_squares(self) -> None:
        values = _gauss_sample(3000, 1.0, 0.024, 7, bkg=0.15)
        centres, counts = metrics.histogram(values, 0.8, 1.15, 0.0025)
        window = (centres > 0.95) & (centres < 1.07)
        x, c = centres[window], counts[window]
        params, _, deviance = metrics._fit_window(x, c, 1.0, 0.024)
        z0 = (x - 1.0) / 0.024
        t = (x - x[0]) / (x[-1] - x[0])

        def fun(theta: np.ndarray) -> np.ndarray:
            return metrics._model_and_residuals.py_func(theta, z0, t, c)[1]

        def jac(theta: np.ndarray) -> np.ndarray:
            return metrics._model_and_residuals.py_func(theta, z0, t, c)[2]

        theta0 = np.array([math.log(c.max()), 0.0, 0.0, 1.0, 1.0])
        ref = least_squares(
            fun,
            theta0,
            jac=jac,
            bounds=([-np.inf] * 3 + [0.0, 0.0], [np.inf] * 5),
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        assert deviance == pytest.approx(2 * ref.cost, rel=1e-6, abs=1e-8)
        assert params[1] == pytest.approx(1.0 + 0.024 * ref.x[1], abs=1e-6)
        assert params[2] == pytest.approx(0.024 * math.exp(ref.x[2]), rel=1e-5)


class TestWidths:
    def test_gaussian_widths(self) -> None:
        values = _gauss_sample(400_000, 1.0, 0.024, 11)
        inflated = math.hypot(0.024, metrics.KDE_BANDWIDTH)
        assert metrics.fwhm_pct(values) == pytest.approx(100 * 2.3548 * inflated, rel=0.01)
        assert metrics.fwtm_pct(values) == pytest.approx(100 * 4.2919 * inflated, rel=0.01)

    def test_nan_cases(self) -> None:
        assert math.isnan(metrics.fwhm_pct(np.ones(metrics.MIN_METRIC_EVENTS - 1)))
        # A peak so wide that it does not stand out of its "continuum".
        assert math.isnan(metrics.fwtm_pct(_gauss_sample(5000, 1.0, 0.2, 2)))

    def test_tail_widens_fwtm_more_than_fwhm(self) -> None:
        rng = np.random.default_rng(4)
        core = rng.normal(1.0, 0.024, 20_000)
        tail = 1.0 - rng.exponential(0.05, 4_000) + rng.normal(0, 0.024, 4_000)
        both = np.concatenate([core, tail])
        fwtm_ratio = metrics.fwtm_pct(both) / metrics.fwtm_pct(core)
        fwhm_ratio = metrics.fwhm_pct(both) / metrics.fwhm_pct(core)
        assert fwtm_ratio > 1.04 and fwtm_ratio > fwhm_ratio

    def test_continuum_is_subtracted(self) -> None:
        # A flat continuum at 30 % of the peak over the whole range: a gross
        # tenth-maximum would not exist, the net widths are those of the peak.
        rng = np.random.default_rng(5)
        peak = rng.normal(1.0, 0.024, 20_000)
        continuum = rng.uniform(0.6, 1.3, 60_000)
        both = np.concatenate([peak, continuum])
        assert metrics.fwtm_pct(both) == pytest.approx(metrics.fwtm_pct(peak), rel=0.05)
        assert metrics.fwhm_pct(both) == pytest.approx(metrics.fwhm_pct(peak), rel=0.05)
