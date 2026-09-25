"""Tests of the default depth fit (plan 10.2 and the unit behaviour of dcalib.depth)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from dcalib import depth
from dcalib.options import DepthOptions
from tests import synthetic_depth as sd

OPTS = DepthOptions()


def _statuses(curve: sd.Curve, n: int, seeds: range, **kwargs: object) -> Counter[str]:
    return Counter(
        depth.fit_anode(*sd.simulate(n, curve, seed, **kwargs), OPTS).status  # type: ignore[arg-type]
        for seed in seeds
    )


# ---------------------------------------------------------------------------
# Plan 10.2
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSyntheticInjection:
    def test_flat_chooses_degree_zero(self) -> None:
        degrees = Counter()
        statuses = Counter()
        for seed in range(100):
            fit = depth.fit_anode(*sd.simulate(2000, sd.flat, seed), OPTS)
            statuses[fit.status] += 1
            degrees[fit.curve.degree if fit.curve is not None else -1] += 1
        assert degrees[0] >= 95
        assert statuses["ok"] <= 5  # false acceptance of the gain gate

    @pytest.mark.parametrize("curve", [sd.linear_drop, sd.concave], ids=["linear", "concave"])
    def test_two_percent_curves_accepted(self, curve: sd.Curve) -> None:
        assert _statuses(curve, 2000, range(100))["ok"] >= 90
        assert _statuses(curve, 4000, range(100, 150))["ok"] >= 48

    def test_steep_curve_accepted_with_large_gain(self) -> None:
        for seed in range(20):
            fit = depth.fit_anode(*sd.simulate(4000, sd.steep, seed), OPTS)
            assert fit.status == "ok" and fit.curve is not None and fit.curve.degree == 2
            assert fit.cv_gain > 0.2

    @pytest.mark.parametrize(
        ("curve", "truth"),
        [
            (sd.concave, (0.996, 0.02, -0.032)),
            (sd.steep, (1.01, 0.03, -0.10)),
            (sd.linear_drop, (1.0, -0.02, 0.0)),
        ],
        ids=["concave", "steep", "linear"],
    )
    def test_coefficients_recovered(self, curve: sd.Curve, truth: tuple[float, ...]) -> None:
        for seed in range(5):
            fit = depth.fit_anode(*sd.simulate(20_000, curve, 50 + seed), OPTS)
            assert fit.curve is not None
            coefficients = np.array(fit.curve.coefficients)
            errors = np.array(fit.curve.errors)
            deg = fit.curve.degree
            # c0 also absorbs the constant low-energy-tail bias of the core fit (~0.1 %).
            tolerance = 3 * errors + np.array([0.002, 0.0, 0.0])
            assert np.all(
                np.abs(coefficients[: deg + 1] - np.array(truth)[: deg + 1]) <= tolerance[: deg + 1]
            )
            if deg < 2:  # the linear case: the omitted term must be negligible
                assert abs(truth[2]) < 1e-12

    def test_source_offset_raises_flag(self) -> None:
        def tilted(r: np.ndarray) -> np.ndarray:
            # Cs 1.25 % above Ge at the 90 % r quantile, 1.25 % below at the 10 % one.
            return sd.concave(r) + 0.0125 * (r - 0.585) / 0.452

        flagged = [
            "source_inconsistent"
            in depth.fit_anode(*sd.simulate(20_000, sd.concave, s, cs_curve=tilted), OPTS).flags
            for s in range(10)
        ]
        assert all(flagged)
        clean = [
            "source_inconsistent"
            in depth.fit_anode(*sd.simulate(20_000, sd.concave, s), OPTS).flags
            for s in range(10)
        ]
        assert not any(clean)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def _slices(r: list[float], mu: list[float], err: float = 0.001) -> depth.Slices:
    n = len(r)
    return depth.Slices(
        r_lo=np.array(r) - 0.01,
        r_hi=np.array(r) + 0.01,
        r_med=np.array(r),
        mu=np.array(mu),
        mu_err=np.full(n, err),
        sigma=np.full(n, 0.02),
        n=np.full(n, 400),
    )


class TestCurves:
    def test_fit_curve_exact_polynomial(self) -> None:
        r = [0.1, 0.3, 0.5, 0.7, 0.9]
        mu = [1.0 + 0.02 * x - 0.03 * x * x for x in r]
        curve = depth.fit_curve(_slices(r, mu), 2)
        np.testing.assert_allclose(curve.coefficients, (1.0, 0.02, -0.03), atol=1e-10)
        assert curve.chi2 == pytest.approx(0.0, abs=1e-12) and curve.ndf == 2
        assert curve(np.array([0.5]))[0] == pytest.approx(mu[2])
        assert curve.sigma(np.array([0.5]))[0] < 0.001
        line = depth.fit_curve(_slices(r, mu), 1)
        assert line.degree == 1 and line.coefficients[2] == 0.0 and line.errors[2] == 0.0
        with pytest.raises(ValueError):
            depth.fit_curve(_slices(r[:2], mu[:2]), 2)

    def test_concave_only_bound(self) -> None:
        r = [0.1, 0.3, 0.5, 0.7, 0.9]
        convex = [1.0 - 0.02 * x + 0.03 * x * x for x in r]
        free = depth.fit_curve(_slices(r, convex), 2)
        bounded = depth.fit_curve(_slices(r, convex), 2, concave_only=True)
        assert free.coefficients[2] > 0 and bounded.coefficients[2] == 0.0
        linear = depth.fit_curve(_slices(r, convex), 1)
        np.testing.assert_allclose(bounded.coefficients, linear.coefficients)

    def test_choose_curve(self) -> None:
        r = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
        flat = depth.choose_curve(_slices(r, [1.0, 1.0005, 0.9995, 1.0, 1.0004, 0.9996]), OPTS)
        assert flat.degree == 0
        sloped = depth.choose_curve(_slices(r, [1.0 - 0.02 * x for x in r]), OPTS)
        assert sloped.degree == 1
        # Symmetric curvature: no linear term, but degree 2 beats degree 0.
        bowl = depth.choose_curve(_slices(r, [1.0 - 0.05 * (x - 0.6) ** 2 for x in r]), OPTS)
        assert bowl.degree == 2
        capped = depth.choose_curve(
            _slices(r, [1.0 - 0.05 * (x - 0.6) ** 2 for x in r]), DepthOptions(max_degree=1)
        )
        assert capped.degree <= 1

    def test_overdispersed_points_need_more_evidence(self) -> None:
        # Scatter far beyond the errors plus a small slope: the drop is scaled
        # by chi2/ndf of the top fit and no longer significant.
        rng = np.random.default_rng(0)
        r = list(np.linspace(0.1, 1.1, 10))
        mu = [1.0 - 0.004 * x + rng.normal(0, 0.01) for x in r]
        assert depth.choose_curve(_slices(r, mu), OPTS).degree == 0


class TestSelectionAndSlices:
    def test_select_windows_and_sources(self) -> None:
        x = np.array([0.74, 0.75, 1.0, 1.12, 1.13, 1.0, 1.0, np.nan])
        r = np.array([0.5, 0.5, -0.01, 1.3, 0.5, 1.31, 0.5, 0.5])
        source = np.array([0, 0, 0, 1, 1, 0, 1, 0], dtype=np.int8)
        assert depth.select(x, r, source, OPTS).tolist() == [
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
        ]
        ge = DepthOptions(sources="ge")
        assert depth.select(x, r, source, ge).tolist()[3] is False
        curve = depth.Curve(1, (1.1, 0.0, 0.0), np.zeros((3, 3)), 0.0, 1)
        # x / 1.1: 0.75 -> 0.68 (out), 1.12 -> 1.02 (in)
        corrected = depth.select(x, r, source, OPTS, curve)
        assert corrected[1] == False and corrected[3] == True  # noqa: E712

    def test_slice_count_rule(self) -> None:
        assert depth.n_slices_for(100, OPTS) == 4
        assert depth.n_slices_for(2400, OPTS) == 6
        assert depth.n_slices_for(100_000, OPTS) == 12

    def test_equal_count_slices(self) -> None:
        x, r, s = sd.simulate(6000, sd.linear_drop, 1)
        sel = depth.select(x, r, s, OPTS)
        slices = depth.fit_slices(x[sel], r[sel], OPTS, 0.025)
        assert len(slices) == depth.n_slices_for(int(sel.sum()), OPTS)
        assert slices.n.max() - slices.n.min() <= 1
        assert np.all(np.diff(slices.r_med) > 0) and np.all(slices.r_lo <= slices.r_med)
        assert slices.spread == pytest.approx(slices.mu.max() - slices.mu.min())
        assert depth.fit_slices(np.empty(0), np.empty(0), OPTS, None).n_failed == 0


class TestFitAnode:
    def test_too_few_events(self) -> None:
        fit = depth.fit_anode(*sd.simulate(500, sd.steep, 0), OPTS)
        assert fit.status == "too_few_events" and fit.curve is None
        assert fit.n_selected < OPTS.min_pairs
        np.testing.assert_array_equal(fit.correction(np.array([0.2, 0.9])), [1.0, 1.0])

    def test_fit_failed_on_noise(self) -> None:
        rng = np.random.default_rng(3)
        n = 3000
        fit = depth.fit_anode(rng.uniform(0.75, 1.12, n), rng.uniform(0, 1.2, n), np.zeros(n), OPTS)
        assert fit.status in ("fit_failed", "no_depth_dependence")

    def test_accepted_anode_fields(self) -> None:
        x, r, s = sd.simulate(8000, sd.steep, 5)
        fit = depth.fit_anode(x, r, s, OPTS, extra_flags=["partial_cathode_coverage"])
        assert fit.accepted and fit.curve is not None
        assert "partial_cathode_coverage" in fit.flags
        assert set(fit.slices) == {"both", "ge", "cs"} and set(fit.source_curves) == {"ge", "cs"}
        assert fit.peak_spread > 0.05 and 0 < fit.ge_cs_max_diff < 0.01
        p01, p50, p99 = fit.ca_quantiles
        assert 0.0 <= p01 < p50 < p99 <= 1.3
        assert fit.cv is not None and set(fit.cv.source_loss) == {0, 1}
        np.testing.assert_allclose(fit.correction(np.array([0.5])), fit.curve(np.array([0.5])))

    def test_single_source_has_no_consistency(self) -> None:
        x, r, s = sd.simulate(8000, sd.steep, 6)
        fit = depth.fit_anode(x, r, s, DepthOptions(sources="cs"))
        assert fit.status == "ok" and "ge" not in fit.slices and math.isnan(fit.ge_cs_max_diff)
        assert fit.cv is not None and set(fit.cv.source_loss) == {1}

    def test_flags(self) -> None:
        # Convex curve, reaching 1.14 at r = 1.3: convex_curve and extrapolation_risk.
        def convex(r: np.ndarray) -> np.ndarray:
            return 1.0 - 0.1 * r + 0.15 * r * r

        fit = depth.fit_anode(*sd.simulate(20_000, convex, 1), OPTS)
        assert fit.curve is not None and fit.curve.degree == 2
        assert {"convex_curve", "extrapolation_risk"} <= set(fit.flags)
        concave_fit = depth.fit_anode(
            *sd.simulate(20_000, convex, 1), DepthOptions(concave_only=True)
        )
        assert "convex_curve" not in concave_fit.flags
        # Events only at r in [0.4, 0.6]: narrow coverage.
        narrow = sd.LineShape(r_range=(0.4, 0.6))
        fit = depth.fit_anode(*sd.simulate(4000, sd.flat, 2, shape=narrow), OPTS)
        assert "narrow_ca_coverage" in fit.flags

    def test_no_gain_when_correction_does_not_help(self) -> None:
        # A real slope but an impossible gain threshold.
        fit = depth.fit_anode(*sd.simulate(8000, sd.steep, 7), DepthOptions(min_gain=0.9))
        assert fit.status == "no_gain" and fit.cv is not None and fit.cv.gain < 0.9
        np.testing.assert_array_equal(fit.correction(np.array([0.9])), [1.0])

    def test_width_ratio(self) -> None:
        raw = depth.Alignment(depth_var=0.02**2, sigma=0.03, n_slices=6)
        corrected = depth.Alignment(depth_var=-1e-6, sigma=0.024, n_slices=6)
        assert depth.width_ratio(raw, corrected) == pytest.approx(0.024 / math.hypot(0.024, 0.02))
        assert math.isnan(depth.width_ratio(depth.Alignment(math.nan, 0.02, 1), corrected))
