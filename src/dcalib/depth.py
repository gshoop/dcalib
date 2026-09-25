"""The default depth fit of one anode (plan 5.3, decisions D6-D8).

Inputs are one anode's events with a calibrated anode and cathode, as
normalised energy ``x = A_keV / E0``, depth proxy ``r = C_keV / A_keV`` (both
uncorrected) and source id, in the anode's row order (source, then CTS). The
fit models the photopeak position as a polynomial ``g(r)`` in E/E0 units; the
correction is ``x_c = x / g(r)`` and the ``.dcc`` stores ``p_k = 511 c_k``
(plan 5.4).

Steps (all thresholds are :class:`~dcalib.options.DepthOptions` fields):

1. **Selection**: ``x in [x_lo, x_hi]``, ``r in [r_lo, r_hi]``, chosen sources.
   Fewer than ``min_pairs`` events: ``too_few_events``.
2. **Slices**: ``clamp(N // per_slice, min_slices, max_slices)`` equal-count
   slices in r, each with its ``r_lo``, ``r_hi`` and median r.
3. **Slice photopeak fits** with :func:`dcalib.metrics.fit_photopeak`, the width
   seeded from the anode's pooled fit. Failed slices are dropped
   (``slice_fit_failed``); fewer than 3 good slices: ``fit_failed``.
4. **Curve**: weighted least squares of the slice means against the slice
   median r for degrees 0..``max_degree``. The chosen degree is the lowest one
   that no higher degree improves significantly: for each higher degree ``d'``
   the chi-square drop must have a chi-square(d' - d) p-value of at least
   ``p_degree``. When the highest-degree fit itself is poor (its chi-square
   p-value is below ``p_degree``), the drops are first divided by its
   ``chi2/ndf``, so scatter beyond the slice errors does not count as curvature. With
   ``concave_only`` the quadratic term is bounded to ``c2 <= 0`` (at the bound
   the quadratic model equals the linear one). A chosen quadratic with ``c2 > 0``
   raises ``convex_curve``.
5. **Re-gate**: the selection is redone on the corrected energy ``x / g(r)`` and
   steps 2-4 repeated, ``gate_iterations`` times. The legacy gate used the
   uncorrected energy, which removes exactly the most depth-affected events.
6. A constant curve (degree 0): ``no_depth_dependence``; nothing to correct.
7. **Gain gate** (D7): two-fold cross-fitting on the even and odd rows of the
   anode. Each half is fitted (steps 1-5, at the degree chosen on all events)
   and its curve corrects the other half. Per source, the photopeak positions
   of r slices of the raw and of these out-of-fold corrected energies are
   measured on the same slices; the spread of the positions (minus their fit
   noise) is the depth broadening, and the predicted width ratio is
   ``sqrt((sigma^2 + V_corr) / (sigma^2 + V_raw))`` with the slices' core
   width ``sigma``. ``cv_gain`` is ``1 - ratio`` averaged over sources. The
   correction is accepted when ``cv_gain >= min_gain`` and no source's
   relative loss (``ratio - 1``) exceeds ``max_source_loss``; otherwise
   ``no_gain``. The accepted curve is the fit on all events. (The plan first
   compared Gaussian-core FWHMs of the spectra; widths measured on a few
   hundred events per source are ten times noisier than positions, and the
   core fit does not see depth broadening; see ``docs/ALGORITHM.md``.)
8. **Source consistency** (D6, both sources only): the Ge-only and Cs-only
   curves at the chosen degree (each needs ``consistency_min_slices *
   per_slice`` selected events) are compared over the anode's 10-90 % r range;
   ``source_inconsistent`` when ``|g_ge - g_cs|`` exceeds ``max(
   consistency_min_diff, consistency_nsigma * sigma_diff)`` anywhere there.
9. **Extrapolation** (D8): ``extrapolation_risk`` when g leaves ``[extrap_g_lo,
   extrap_g_hi]`` on ``[0, extrap_r_max]``; ``narrow_ca_coverage`` when the 1 % r
   quantile of the selected events is above ``coverage_r_lo`` or the 99 %
   quantile below ``coverage_r_hi``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2 as chi2_dist

from dcalib.metrics import MIN_METRIC_EVENTS, PeakFit, fit_photopeak
from dcalib.options import (
    FLAG_CONVEX_CURVE,
    FLAG_EXTRAPOLATION_RISK,
    FLAG_NARROW_CA_COVERAGE,
    FLAG_SLICE_FIT_FAILED,
    FLAG_SOURCE_INCONSISTENT,
    SOURCE_CS,
    SOURCE_GE,
    SOURCES_BOTH,
    STATUS_FIT_FAILED,
    STATUS_NO_DEPTH_DEPENDENCE,
    STATUS_NO_GAIN,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    DepthOptions,
    order_flags,
)

__all__ = [
    "CURVE_BOTH",
    "CURVE_CS",
    "CURVE_GE",
    "CURVE_NAMES",
    "Curve",
    "DepthFit",
    "Slices",
    "choose_curve",
    "cross_validate",
    "fit_anode",
    "fit_curve",
    "fit_depth_curve",
    "fit_slices",
    "select",
]

FloatArray = npt.NDArray[np.float64]

CURVE_BOTH = "both"
CURVE_GE = "ge"
CURVE_CS = "cs"
CURVE_NAMES: tuple[str, ...] = (CURVE_BOTH, CURVE_GE, CURVE_CS)
"""Names of the curves stored per anode: the pooled fit and the per-source fits."""

_CURVE_SOURCE = {CURVE_GE: SOURCE_GE, CURVE_CS: SOURCE_CS}
_GRID_POINTS = 101


# ---------------------------------------------------------------------------
# Slices and curves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slices:
    """The successful slice photopeak fits of one curve (parallel arrays).

    Attributes:
        r_lo: Smallest r in the slice.
        r_hi: Largest r in the slice.
        r_med: Median r of the slice (the abscissa of the curve fit).
        mu: Gaussian-core photopeak position in x.
        mu_err: Its standard error.
        sigma: Gaussian-core width in x.
        n: Events in the slice.
        n_failed: Slices whose fit failed (not in the arrays).
    """

    r_lo: FloatArray
    r_hi: FloatArray
    r_med: FloatArray
    mu: FloatArray
    mu_err: FloatArray
    sigma: FloatArray
    n: npt.NDArray[np.int64]
    n_failed: int = 0

    def __len__(self) -> int:
        return len(self.mu)

    @classmethod
    def empty(cls, n_failed: int = 0) -> Slices:
        e = np.empty(0, dtype=np.float64)
        return cls(e, e, e, e, e, e, np.empty(0, dtype=np.int64), n_failed)

    @property
    def spread(self) -> float:
        """Largest minus smallest slice position (NaN without slices)."""
        return float(self.mu.max() - self.mu.min()) if len(self) else math.nan


@dataclass(frozen=True)
class Curve:
    """A fitted depth curve ``g(r) = c0 + c1 r + c2 r^2`` in E/E0 units.

    Attributes:
        degree: Polynomial degree (0, 1 or 2).
        coefficients: ``(c0, c1, c2)``; terms above ``degree`` are 0.
        covariance: 3x3 covariance of the coefficients (zero rows above ``degree``).
        chi2: Weighted residual sum of squares of the slice points.
        ndf: Slice points minus fitted coefficients.
    """

    degree: int
    coefficients: tuple[float, float, float]
    covariance: FloatArray
    chi2: float
    ndf: int

    @property
    def chi2ndf(self) -> float:
        return self.chi2 / self.ndf if self.ndf > 0 else math.nan

    @property
    def errors(self) -> tuple[float, float, float]:
        d = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        return float(d[0]), float(d[1]), float(d[2])

    def __call__(self, r: npt.ArrayLike) -> FloatArray:
        rr = np.asarray(r, dtype=np.float64)
        c0, c1, c2 = self.coefficients
        return np.asarray(c0 + rr * (c1 + rr * c2), dtype=np.float64)

    def sigma(self, r: npt.ArrayLike) -> FloatArray:
        """Standard error of g(r) from the coefficient covariance."""
        rr = np.asarray(r, dtype=np.float64)
        basis = np.stack([np.ones_like(rr), rr, rr * rr], axis=-1)
        var = np.einsum("...i,ij,...j->...", basis, self.covariance, basis)
        return np.asarray(np.sqrt(np.maximum(var, 0.0)), dtype=np.float64)

    @classmethod
    def constant(cls, value: float = 1.0) -> Curve:
        return cls(0, (float(value), 0.0, 0.0), np.zeros((3, 3)), 0.0, 0)


def select(
    x: FloatArray,
    r: FloatArray,
    source: npt.NDArray[np.integer],
    options: DepthOptions,
    curve: Curve | None = None,
) -> npt.NDArray[np.bool_]:
    """Step 1 (and the re-gate): events in the energy and r windows of the chosen sources.

    With a ``curve`` the energy window applies to the corrected ``x / g(r)``.
    """
    in_r = (r >= options.r_lo) & (r <= options.r_hi)
    energy = x
    if curve is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            energy = x / curve(r)
    in_x = (energy >= options.x_lo) & (energy <= options.x_hi)
    in_source = np.isin(source, options.source_ids)
    return np.asarray(in_r & in_x & in_source & np.isfinite(x) & np.isfinite(r))


def _peak(values: FloatArray, options: DepthOptions, seed_sigma: float | None) -> PeakFit:
    return fit_photopeak(
        values,
        hist_lo=options.hist_lo,
        hist_hi=options.hist_hi,
        hist_bin=options.hist_bin,
        window_lo_sigma=options.window_lo_sigma,
        window_hi_sigma=options.window_hi_sigma,
        window_iterations=options.window_iterations,
        seed_sigma=seed_sigma,
    )


def n_slices_for(n_events: int, options: DepthOptions) -> int:
    """``clamp(N // per_slice, min_slices, max_slices)``."""
    return int(min(max(n_events // options.per_slice, options.min_slices), options.max_slices))


def fit_slices(
    x: FloatArray, r: FloatArray, options: DepthOptions, seed_sigma: float | None
) -> Slices:
    """Steps 2-3: equal-count r slices of the given (selected) events and their peak fits."""
    n_total = len(x)
    if n_total == 0:
        return Slices.empty()
    order = np.argsort(r, kind="stable")
    parts = np.array_split(order, n_slices_for(n_total, options))
    rows: list[tuple[float, float, float, float, float, float, int]] = []
    failed = 0
    for idx in parts:
        if len(idx) == 0:
            continue
        fit = _peak(x[idx], options, seed_sigma)
        if not fit.ok:
            failed += 1
            continue
        rs = r[idx]
        rows.append(
            (
                float(rs.min()),
                float(rs.max()),
                float(np.median(rs)),
                fit.mu,
                fit.mu_err,
                fit.sigma,
                len(idx),
            )
        )
    if not rows:
        return Slices.empty(failed)
    cols = list(zip(*rows))
    return Slices(
        r_lo=np.array(cols[0]),
        r_hi=np.array(cols[1]),
        r_med=np.array(cols[2]),
        mu=np.array(cols[3]),
        mu_err=np.array(cols[4]),
        sigma=np.array(cols[5]),
        n=np.array(cols[6], dtype=np.int64),
        n_failed=failed,
    )


def fit_curve(slices: Slices, degree: int, concave_only: bool = False) -> Curve:
    """Weighted least-squares polynomial of the slice positions against median r.

    Args:
        slices: The slice points (at least ``degree + 1``).
        degree: 0, 1 or 2.
        concave_only: Bound ``c2 <= 0``; a convex optimum is replaced by the
            linear fit (the bounded optimum).

    Raises:
        ValueError: With fewer points than coefficients.
    """
    n_coef = degree + 1
    if len(slices) < n_coef:
        raise ValueError(f"a degree-{degree} curve needs {n_coef} points, got {len(slices)}")
    w = 1.0 / slices.mu_err**2
    basis = np.vander(slices.r_med, n_coef, increasing=True)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(basis * sw[:, None], slices.mu * sw, rcond=None)
    if concave_only and degree == 2 and coef[2] > 0:
        linear = fit_curve(slices, 1)
        return Curve(2, linear.coefficients, linear.covariance, linear.chi2, len(slices) - 3)
    resid = slices.mu - basis @ coef
    chi2 = float(np.sum(w * resid**2))
    cov_small = np.linalg.pinv(basis.T @ (basis * w[:, None]))
    cov = np.zeros((3, 3))
    cov[:n_coef, :n_coef] = cov_small
    coefficients = [float(c) for c in coef] + [0.0] * (3 - n_coef)
    return Curve(
        degree, (coefficients[0], coefficients[1], coefficients[2]), cov, chi2, len(slices) - n_coef
    )


def choose_curve(slices: Slices, options: DepthOptions) -> Curve:
    """Step 4: fit degrees 0..max_degree and keep the lowest adequate one."""
    top = min(options.max_degree, len(slices) - 1)
    curves = [fit_curve(slices, d, options.concave_only) for d in range(top + 1)]
    best = curves[top]
    overdispersed = best.ndf > 0 and chi2_dist.sf(best.chi2, best.ndf) < options.p_degree
    scale = best.chi2ndf if overdispersed else 1.0
    for d in range(top):
        adequate = True
        for higher in range(d + 1, top + 1):
            drop = max(curves[d].chi2 - curves[higher].chi2, 0.0) / scale
            if chi2_dist.sf(drop, higher - d) < options.p_degree:
                adequate = False
                break
        if adequate:
            return curves[d]
    return best


@dataclass(frozen=True)
class _CurveFit:
    status: str
    curve: Curve | None
    slices: Slices
    n_selected: int
    selected: npt.NDArray[np.bool_]
    first_selected: npt.NDArray[np.bool_]


def fit_depth_curve(
    x: FloatArray,
    r: FloatArray,
    source: npt.NDArray[np.integer],
    options: DepthOptions,
    *,
    check_min_pairs: bool = True,
    degree: int | None = None,
) -> _CurveFit:
    """Steps 1-5 on a set of events.

    With ``degree`` the curve has that degree (no degree test); the
    cross-validation uses it to refit the degree chosen on all events.

    Returns:
        The status (``ok`` meaning "a curve was fitted"; ``too_few_events`` or
        ``fit_failed`` otherwise), the curve, its slices, the final selection
        and the first (uncorrected) selection.
    """
    selected = select(x, r, source, options)
    first = selected
    curve: Curve | None = None
    slices = Slices.empty()
    for iteration in range(options.gate_iterations + 1):
        if iteration > 0:
            assert curve is not None
            selected = select(x, r, source, options, curve)
        n_sel = int(selected.sum())
        if n_sel < (options.min_pairs if check_min_pairs else options.min_pairs // 2):
            return _CurveFit(STATUS_TOO_FEW_EVENTS, curve, slices, n_sel, selected, first)
        xs, rs = x[selected], r[selected]
        pooled = _peak(xs, options, None)
        slices = fit_slices(xs, rs, options, pooled.sigma if pooled.ok else None)
        if len(slices) < 3:
            return _CurveFit(STATUS_FIT_FAILED, None, slices, n_sel, selected, first)
        if degree is None:
            curve = choose_curve(slices, options)
        else:
            curve = fit_curve(slices, min(degree, len(slices) - 1), options.concave_only)
    return _CurveFit(STATUS_OK, curve, slices, int(selected.sum()), selected, first)


# ---------------------------------------------------------------------------
# Gain gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossValidation:
    """Step 7's outcome.

    Attributes:
        gain: ``cv_gain``, the mean over sources of ``1 - ratio``, with ``ratio``
            from :func:`width_ratio` of the raw and the out-of-fold corrected
            energies (NaN when nothing could be evaluated).
        source_loss: Per source, ``ratio - 1``.
        accepted: ``gain >= min_gain`` and every source loss ``<= max_source_loss``.
    """

    gain: float
    source_loss: dict[int, float]
    accepted: bool


@dataclass(frozen=True)
class Alignment:
    """How well a set of events' photopeak positions line up across r.

    Attributes:
        depth_var: Sample variance of the slice positions minus their mean
            squared fit error (the part due to depth), times ``(k-1)/k`` for
            ``k`` slices; noise can make it slightly negative.
        sigma: Median Gaussian-core width of the slices.
        n_slices: Slices with a successful fit.
    """

    depth_var: float
    sigma: float
    n_slices: int


def alignment(
    x: FloatArray, r: FloatArray, options: DepthOptions, seed_sigma: float | None
) -> Alignment:
    """Slice ``x`` in r (as in step 2) and measure the spread of the slice positions."""
    slices = fit_slices(x, r, options, seed_sigma)
    k = len(slices)
    if k < 2:
        return Alignment(math.nan, math.nan, k)
    spread = float(np.var(slices.mu, ddof=1) - np.mean(slices.mu_err**2))
    return Alignment(spread * (k - 1) / k, float(np.median(slices.sigma)), k)


def width_ratio(raw: Alignment, corrected: Alignment) -> float:
    """Predicted corrected/raw photopeak width from the alignment of both.

    ``sqrt((sigma^2 + V_corr) / (sigma^2 + V_raw))``, with the slice core width
    ``sigma`` of the corrected events and the depth variances clipped at 0:
    the photopeak is the core broadened by the spread of positions over r.
    """
    sigma = corrected.sigma
    values = (sigma, raw.depth_var, corrected.depth_var)
    if not all(math.isfinite(v) for v in values):
        return math.nan
    base = sigma * sigma
    return math.sqrt((base + max(corrected.depth_var, 0.0)) / (base + max(raw.depth_var, 0.0)))


def cross_validate(
    x: FloatArray,
    r: FloatArray,
    source: npt.NDArray[np.integer],
    options: DepthOptions,
    degree: int,
) -> CrossValidation:
    """Step 7: two-fold cross-fitted resolution gain (even/odd rows).

    Each half is fitted with the ``degree`` chosen on all events (the
    cross-validation estimates the gain of that model on unseen events; it
    does not repeat the model selection on half the statistics), and every
    event is corrected with the curve of the *other* half. Per source, the
    events are sliced in r and the photopeak positions of the raw and of the
    out-of-fold corrected energies are measured on the same slices
    (:func:`alignment`); :func:`width_ratio` turns their spreads into the
    predicted corrected/raw photopeak width. Each fold's curve is judged only
    on events it never saw, and each source's comparison uses all its events.
    """
    n = len(x)
    rows = np.arange(n)
    in_r = (r >= options.r_lo) & (r <= options.r_hi) & np.isfinite(x) & np.isfinite(r)
    corrected = np.full(n, np.nan)
    for parity in (0, 1):
        train = rows % 2 == parity
        test = ~train
        fitted = fit_depth_curve(
            x[train], r[train], source[train], options, check_min_pairs=False, degree=degree
        )
        if fitted.status == STATUS_OK and fitted.curve is not None:
            corrected[test] = x[test] / fitted.curve(r[test])
        else:
            corrected[test] = x[test]  # no curve from this half: nothing is corrected
    loss: dict[int, float] = {}
    for s in options.source_ids:
        mask = in_r & (source == s)
        if int(mask.sum()) < MIN_METRIC_EVENTS:
            continue
        xs, rs = x[mask], r[mask]
        pooled = _peak(xs, options, None)
        seed = pooled.sigma if pooled.ok else None
        ratio = width_ratio(
            alignment(xs, rs, options, seed), alignment(corrected[mask], rs, options, seed)
        )
        if math.isfinite(ratio):
            loss[s] = ratio - 1.0
    gain = -float(np.mean(list(loss.values()))) if loss else math.nan
    accepted = (
        math.isfinite(gain)
        and gain >= options.min_gain
        and all(v <= options.max_source_loss for v in loss.values())
    )
    return CrossValidation(gain, loss, accepted)


# ---------------------------------------------------------------------------
# Whole anode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthFit:
    """Everything the default method measures for one anode.

    Attributes:
        status: One of the ``STATUS_*`` of :mod:`dcalib.options` handled here
            (``ok``, ``no_gain``, ``no_depth_dependence``, ``too_few_events``,
            ``fit_failed``).
        flags: Warning flags in canonical order.
        curve: The pooled curve on all events (None for ``too_few_events`` and
            ``fit_failed``).
        slices: Slice points per curve name (``both``, and ``ge``/``cs`` when
            fitted).
        source_curves: The per-source curves at the chosen degree.
        n_selected: Events in the final selection.
        ca_quantiles: 1, 50 and 99 % r quantiles of the first selection.
        peak_spread: Largest minus smallest pooled slice position.
        ge_cs_max_diff: Largest ``|g_ge - g_cs|`` over the 10-90 % r range.
        cv: The cross-validation outcome (None when not run).
    """

    status: str
    flags: tuple[str, ...] = ()
    curve: Curve | None = None
    slices: dict[str, Slices] = field(default_factory=dict)
    source_curves: dict[str, Curve] = field(default_factory=dict)
    n_selected: int = 0
    ca_quantiles: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    peak_spread: float = math.nan
    ge_cs_max_diff: float = math.nan
    cv: CrossValidation | None = None

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_OK

    @property
    def cv_gain(self) -> float:
        return self.cv.gain if self.cv is not None else math.nan

    def correction(self, r: npt.ArrayLike) -> FloatArray:
        """g(r) of an accepted anode, and 1 otherwise (nothing is corrected)."""
        rr = np.asarray(r, dtype=np.float64)
        if not self.accepted or self.curve is None:
            return np.ones_like(rr)
        return self.curve(rr)


def _quantiles(r: FloatArray) -> tuple[float, float, float]:
    if len(r) < 10:
        return (math.nan, math.nan, math.nan)
    q = np.quantile(r, [0.01, 0.5, 0.99])
    return float(q[0]), float(q[1]), float(q[2])


def _source_consistency(
    x: FloatArray,
    r: FloatArray,
    source: npt.NDArray[np.integer],
    selected: npt.NDArray[np.bool_],
    degree: int,
    options: DepthOptions,
    seed_sigma: float | None,
) -> tuple[dict[str, Slices], dict[str, Curve], float, bool]:
    """Step 8: per-source curves at ``degree``; returns (slices, curves, max diff, flag)."""
    slices: dict[str, Slices] = {}
    curves: dict[str, Curve] = {}
    need = options.consistency_min_slices * options.per_slice
    for name, s in _CURVE_SOURCE.items():
        mask = selected & (source == s)
        if int(mask.sum()) < need:
            continue
        sl = fit_slices(x[mask], r[mask], options, seed_sigma)
        if len(sl) < max(degree + 1, 3):
            continue
        slices[name] = sl
        curves[name] = fit_curve(sl, degree, options.concave_only)
    if len(curves) < 2:
        return slices, curves, math.nan, False
    lo, hi = np.quantile(r[selected], [0.1, 0.9])
    grid = np.linspace(lo, hi, _GRID_POINTS)
    diff = curves[CURVE_GE](grid) - curves[CURVE_CS](grid)
    sigma = np.sqrt(curves[CURVE_GE].sigma(grid) ** 2 + curves[CURVE_CS].sigma(grid) ** 2)
    limit = np.maximum(options.consistency_min_diff, options.consistency_nsigma * sigma)
    return slices, curves, float(np.max(np.abs(diff))), bool(np.any(np.abs(diff) > limit))


def fit_anode(
    x: npt.ArrayLike,
    r: npt.ArrayLike,
    source: npt.ArrayLike,
    options: DepthOptions | None = None,
    *,
    extra_flags: Sequence[str] = (),
) -> DepthFit:
    """Run the whole default method (steps 1-9) on one anode's events.

    Args:
        x: ``A_keV / E0`` per event (calibrated anode and cathode only).
        r: ``C_keV / A_keV`` per event.
        source: Source id per event.
        options: The options (default: ``DepthOptions()``).
        extra_flags: Flags decided by the caller (e.g. ``partial_cathode_coverage``).

    Returns:
        The anode's :class:`DepthFit`.
    """
    opts = options if options is not None else DepthOptions()
    xs = np.asarray(x, dtype=np.float64)
    rs = np.asarray(r, dtype=np.float64)
    ss = np.asarray(source, dtype=np.int8)
    flags = set(extra_flags)

    fitted = fit_depth_curve(xs, rs, ss, opts)
    quantiles = _quantiles(rs[fitted.first_selected])
    if math.isfinite(quantiles[0]) and (
        quantiles[0] > opts.coverage_r_lo or quantiles[2] < opts.coverage_r_hi
    ):
        flags.add(FLAG_NARROW_CA_COVERAGE)
    if fitted.slices.n_failed:
        flags.add(FLAG_SLICE_FIT_FAILED)
    if fitted.status != STATUS_OK or fitted.curve is None:
        return DepthFit(
            fitted.status,
            order_flags(flags),
            slices={CURVE_BOTH: fitted.slices} if len(fitted.slices) else {},
            n_selected=fitted.n_selected,
            ca_quantiles=quantiles,
            peak_spread=fitted.slices.spread,
        )

    curve = fitted.curve
    if curve.degree == 2 and curve.coefficients[2] > 0:
        flags.add(FLAG_CONVEX_CURVE)
    grid = np.linspace(0.0, opts.extrap_r_max, _GRID_POINTS)
    g = curve(grid)
    if curve.degree > 0 and (g.min() < opts.extrap_g_lo or g.max() > opts.extrap_g_hi):
        flags.add(FLAG_EXTRAPOLATION_RISK)

    slices = {CURVE_BOTH: fitted.slices}
    source_curves: dict[str, Curve] = {}
    max_diff = math.nan
    if opts.sources == SOURCES_BOTH:
        pooled = _peak(xs[fitted.selected], opts, None)
        per_slices, source_curves, max_diff, inconsistent = _source_consistency(
            xs, rs, ss, fitted.selected, curve.degree, opts, pooled.sigma if pooled.ok else None
        )
        slices.update(per_slices)
        if inconsistent:
            flags.add(FLAG_SOURCE_INCONSISTENT)

    if curve.degree == 0:
        status = STATUS_NO_DEPTH_DEPENDENCE
        cv = None
    else:
        cv = cross_validate(xs, rs, ss, opts, curve.degree)
        status = STATUS_OK if cv.accepted else STATUS_NO_GAIN
    return DepthFit(
        status,
        order_flags(flags),
        curve=curve,
        slices=slices,
        source_curves=source_curves,
        n_selected=fitted.n_selected,
        ca_quantiles=quantiles,
        peak_spread=fitted.slices.spread,
        ge_cs_max_diff=max_diff,
        cv=cv,
    )
