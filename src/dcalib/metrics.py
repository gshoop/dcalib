"""Photopeak fits and resolution metrics (plan 5.3 step 3 and 5.5).

All energies are normalised, ``x = A_keV / E0``, so a photopeak sits near 1.

:func:`fit_photopeak` is the one peak fitter of the package (slice positions,
pooled widths and the fitted curves drawn in the GUI). It histograms the
values on ``[hist_lo, hist_hi)`` in ``hist_bin`` bins, seeds the mean at the
mode of a lightly smoothed histogram and the width from a given seed (the
anode's pooled fit) or the half maximum, then fits a Gaussian plus a linear
background over the asymmetric window ``[mu - window_lo_sigma * sigma, mu +
window_hi_sigma * sigma]``. The asymmetry keeps the low-energy (hole-trapping)
tail from pulling the centroid. The window is re-centred on the fitted mean and
the fit repeated, up to ``window_iterations`` times, until the window stops
moving; its width stays at the seed (see ``adapt_width``).

The fit is a binned Poisson maximum-likelihood fit: it minimises the
Baker-Cousins deviance, as a least-squares problem on the signed root-deviance
residuals (the sum of their squares is -2 ln of the likelihood ratio). The
optimiser is a Levenberg-Marquardt loop compiled with numba (the same method as
``scipy.optimize.least_squares(method="lm")``, which the tests use as the
reference), because the depth fit runs about a hundred peak fits per anode and
scipy's per-call overhead would dominate the run time. Parameter errors come
from the inverse of ``J^T J`` at the optimum (the Fisher information of the
Poisson likelihood); they are not scaled by the goodness of fit. On a pure
Gaussian the asymmetric window biases the mean upwards by about a third of its
error (0.06 % of E0 at 800 events); the bias is the same for equal-count slices,
so it only shifts the curve's constant term.

:func:`fwhm_pct` and :func:`fwtm_pct` are non-parametric: the full width at
half and at a tenth of the net photopeak (above the continuum level) of the
spectrum smoothed with a Gaussian kernel (a kernel density estimate on a fine
grid), in % of the peak position. The continuum is subtracted because at
511 keV the Compton continuum of CZT often stays above a tenth of the peak. The
level is flat: exact for a flat continuum, while a continuum that ends under
the low half of the peak narrows the widths (by about 10 % for one at 30 % of
the peak); the metrics compare spectra of the same anode and across the fleet,
not absolute resolutions. The Gaussian-core width of :func:`fit_photopeak` is *not* used as the
resolution metric: with its free linear background in a narrow asymmetric
window it absorbs the low-side shoulder that depth smearing creates, so it
barely changes when a large depth effect is corrected (phase 3 study,
``docs/ALGORITHM.md``). The core fit is used for what it measures well, the
photopeak position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

__all__ = [
    "FWHM_PER_SIGMA",
    "CONTINUUM_OFFSETS",
    "KDE_BANDWIDTH",
    "KDE_RANGE",
    "PEAK_SEARCH",
    "MIN_METRIC_EVENTS",
    "PeakFit",
    "WidthCrossings",
    "fit_photopeak",
    "fwhm_pct",
    "fwtm_pct",
    "histogram",
    "kde_spectrum",
    "poisson_deviance",
    "smooth",
    "width_crossings",
    "width_pct",
]

FloatArray = npt.NDArray[np.float64]

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
"""FWHM of a Gaussian in units of its sigma (2.3548...)."""

MIN_METRIC_EVENTS = 200
"""Fewer values than this give NaN resolution metrics (the fits are too noisy)."""

MIN_WINDOW_COUNTS = 30
"""A fit window with fewer counts than this is not fitted."""

MIN_WINDOW_BINS = 6
"""The fit window always spans at least this many bins."""

DEFAULT_SIGMA = 0.024
"""Width seed in x when neither a seed nor the half maximum gives one (FWHM ~5.7 %)."""

KDE_RANGE = (0.60, 1.30)
"""Range in x of the smoothed spectra of the resolution metrics (wide, for the tails)."""

KDE_BIN = 0.0005
"""Grid step in x of the smoothed spectra."""

KDE_BANDWIDTH = 0.008
"""Gaussian kernel sigma in x of the smoothed spectra. A third of a typical peak
sigma: it widens a 5.7 % FWHM by about 5 % of itself, but a narrower kernel
resolves the fine structure at the top of real peaks and makes the half-maximum
crossings jump (phase 4, docs/ALGORITHM.md)."""

MAX_CONTINUUM_FRACTION = 0.6
"""Widths are NaN when the continuum level is above this fraction of the peak maximum."""

PEAK_SEARCH = (0.85, 1.15)
"""Range in x where the photopeak maximum is searched."""

CONTINUUM_OFFSETS = (0.18, 0.10)
"""The continuum level is the median of the smoothed spectrum from 0.18 to 0.10 below
the peak (in x); the widths are measured on the peak above it."""

_SERIES_U = 1e-4
_LOG_CLIP = 700.0
_FTOL = 1e-8  # relative change of the deviance (MINPACK's default is 1.49e-8)
_XTOL = 1e-10  # step size relative to the parameters
_STALL_TOL = 1e-6
_LM_MAX_ITER = 300
_SMOOTH_KERNEL = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


def histogram(
    values: npt.ArrayLike, lo: float, hi: float, width: float
) -> tuple[FloatArray, FloatArray]:
    """Histogram ``values`` on ``[lo, hi)`` in bins of ``width``.

    Returns:
        ``(bin centres, counts)`` as float arrays.
    """
    n_bins = max(1, int(round((hi - lo) / width)))
    edges = lo + width * np.arange(n_bins + 1, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    idx = np.floor((v - lo) / width).astype(np.int64)
    keep = (idx >= 0) & (idx < n_bins) & (v >= lo) & (v < hi)
    counts = np.bincount(idx[keep], minlength=n_bins).astype(np.float64)
    return 0.5 * (edges[:-1] + edges[1:]), counts


def smooth(counts: npt.ArrayLike) -> FloatArray:
    """Light smoothing: a 5-bin triangular kernel (1, 2, 3, 2, 1) / 9, same length."""
    c = np.asarray(counts, dtype=np.float64)
    if len(c) < len(_SMOOTH_KERNEL):
        return c.copy()
    return np.asarray(np.convolve(c, _SMOOTH_KERNEL, mode="same"), dtype=np.float64)


# ---------------------------------------------------------------------------
# Poisson deviance
# ---------------------------------------------------------------------------


def _deviance_and_slope(n: FloatArray, mu: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Deviance terms ``D = 2 (mu - n + n ln(n/mu))`` and ``g = |n - mu| / sqrt(mu D)``.

    ``g`` gives the derivative of the signed root deviance
    ``r = sign(n - mu) sqrt(D)``: ``dr/dmu = -g / sqrt(mu)``. Near ``n = mu``
    both use their Taylor series in ``u = n/mu - 1``.
    """
    u = (n - mu) / mu
    small = np.abs(u) < _SERIES_U
    h_series = u * u * (0.5 - u / 6.0 + u * u / 12.0)
    n_log = n * np.log1p(np.where(n > 0.0, u, 0.0))
    dev = np.where(small, 2.0 * mu * h_series, 2.0 * (mu - n + n_log))
    dev = np.maximum(dev, 0.0)
    g_general = np.abs(n - mu) / np.sqrt(mu * dev)
    g_series = 1.0 / np.sqrt(1.0 - u / 3.0 + u * u / 6.0)
    g = np.where(small, g_series, g_general)
    g = np.where(np.isfinite(g), g, 0.0)
    return dev, g


def poisson_deviance(counts: npt.ArrayLike, expected: npt.ArrayLike) -> FloatArray:
    """Per-bin Baker-Cousins deviance; the sum is the likelihood-ratio chi-square."""
    n = np.asarray(counts, dtype=np.float64)
    mu = np.asarray(expected, dtype=np.float64)
    with np.errstate(all="ignore"):
        dev, _ = _deviance_and_slope(n, mu)
    return dev


# ---------------------------------------------------------------------------
# Peak fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeakFit:
    """Result of :func:`fit_photopeak`.

    Attributes:
        ok: The fit converged to a plausible peak inside its window.
        mu: Gaussian-core mean (NaN when not ok).
        mu_err: Its standard error.
        sigma: Gaussian-core sigma.
        sigma_err: Its standard error.
        amplitude: Peak height in counts per bin.
        background: Linear background at the window edges (counts per bin).
        window: The final fit window ``(lo, hi)``.
        n: Number of finite values given.
        chi2ndf: Deviance per degree of freedom over the window.
        message: Why the fit is not ok (empty when ok).
    """

    ok: bool
    mu: float = math.nan
    mu_err: float = math.nan
    sigma: float = math.nan
    sigma_err: float = math.nan
    amplitude: float = math.nan
    background: tuple[float, float] = (math.nan, math.nan)
    window: tuple[float, float] = (math.nan, math.nan)
    n: int = 0
    chi2ndf: float = math.nan
    message: str = ""

    @property
    def fwhm_pct(self) -> float:
        """Gaussian-core FWHM in % of the mean (NaN when not ok)."""
        if not self.ok or not self.mu > 0:
            return math.nan
        return 100.0 * FWHM_PER_SIGMA * self.sigma / self.mu

    def curve(self, x: npt.ArrayLike) -> FloatArray:
        """The fitted model (counts per bin) at ``x`` (background extended linearly)."""
        xs = np.asarray(x, dtype=np.float64)
        if not self.ok:
            return np.full(xs.shape, np.nan)
        lo, hi = self.window
        b_lo, b_hi = self.background
        bkg = b_lo + (b_hi - b_lo) * (xs - lo) / (hi - lo)
        gauss = self.amplitude * np.exp(-0.5 * ((xs - self.mu) / self.sigma) ** 2)
        return np.asarray(gauss + bkg, dtype=np.float64)


def _fail(message: str, n: int) -> PeakFit:
    return PeakFit(False, n=n, message=message)


def _seed_sigma(centres: FloatArray, smoothed: FloatArray, i_max: int) -> float:
    """Sigma from the half-maximum crossings of the smoothed histogram."""
    half = 0.5 * smoothed[i_max]
    left = i_max
    while left > 0 and smoothed[left] > half:
        left -= 1
    right = i_max
    while right < len(smoothed) - 1 and smoothed[right] > half:
        right += 1
    width = centres[right] - centres[left]
    return width / FWHM_PER_SIGMA if width > 0 else DEFAULT_SIGMA


@numba.njit(cache=True, nogil=True)
def _model_and_residuals(
    theta: FloatArray, z0: FloatArray, t: FloatArray, counts: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Expected counts, signed root-deviance residuals and their Jacobian at ``theta``."""
    n_bins = z0.shape[0]
    log_amp, m, log_s, b0, b1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    s = math.exp(min(max(log_s, -_LOG_CLIP), _LOG_CLIP))
    expected = np.empty(n_bins)
    res = np.empty(n_bins)
    jac = np.empty((n_bins, 5))
    cost = 0.0
    for i in range(n_bins):
        z = (z0[i] - m) / s
        gauss = math.exp(min(max(log_amp - 0.5 * z * z, -_LOG_CLIP), _LOG_CLIP))
        e = gauss + b0 + (b1 - b0) * t[i]
        if e < 1e-12:
            e = 1e-12
        n = counts[i]
        u = (n - e) / e
        if abs(u) < _SERIES_U:
            dev = 2.0 * e * u * u * (0.5 - u / 6.0 + u * u / 12.0)
            g = 1.0 / math.sqrt(1.0 - u / 3.0 + u * u / 6.0)
        else:
            n_log = n * math.log1p(u) if n > 0.0 else 0.0
            dev = 2.0 * (e - n + n_log)
            if dev < 0.0:
                dev = 0.0
            g = abs(n - e) / math.sqrt(e * dev) if dev > 0.0 else 0.0
        root = math.sqrt(dev)
        res[i] = root if n >= e else -root
        d = -g / math.sqrt(e)
        jac[i, 0] = d * gauss
        jac[i, 1] = d * gauss * z / s
        jac[i, 2] = d * gauss * z * z
        jac[i, 3] = d * (1.0 - t[i])
        jac[i, 4] = d * t[i]
        expected[i] = e
        cost += dev
    return expected, res, jac, cost


@numba.njit(cache=True, nogil=True)
def _levenberg_marquardt(
    theta0: FloatArray, z0: FloatArray, t: FloatArray, counts: FloatArray
) -> tuple[bool, FloatArray, FloatArray, float]:
    """Minimise the Poisson deviance of Gaussian + linear background (backgrounds >= 0).

    Levenberg-Marquardt with Marquardt's diagonal scaling. The two background
    edge values are bounded below by 0: a step that would make one negative is
    projected onto 0, and a background at 0 whose gradient points outwards is
    held fixed for the step (an active set), so the iteration does not zig-zag
    along the bound.

    Returns:
        ``(converged, theta, J^T J at the optimum, deviance)``.
    """
    theta = theta0.copy()
    _, res, jac, cost = _model_and_residuals(theta, z0, t, counts)
    lam = 1e-3
    converged = False
    change = np.inf
    for _ in range(_LM_MAX_ITER):
        jtj = jac.T @ jac
        grad = jac.T @ res
        # Active set: a background at 0 whose gradient points below 0 stays fixed.
        rhs = -grad
        fixed = np.zeros(5, dtype=np.bool_)
        for k in (3, 4):
            if theta[k] <= 0.0 and grad[k] > 0.0:
                fixed[k] = True
                rhs[k] = 0.0
        accepted = False
        step_size = 0.0
        new_cost = cost
        for _ in range(30):
            system = jtj.copy()
            for k in range(5):
                system[k, k] += lam * max(jtj[k, k], 1e-12)
            for k in range(5):
                if fixed[k]:
                    system[k, :] = 0.0
                    system[:, k] = 0.0
                    system[k, k] = 1.0
            step = np.linalg.solve(system, rhs)
            trial = theta + step
            for k in (3, 4):
                if trial[k] < 0.0:
                    trial[k] = 0.0
            _, trial_res, trial_jac, trial_cost = _model_and_residuals(trial, z0, t, counts)
            if trial_cost <= cost:
                step_size = np.max(np.abs(trial - theta))
                theta = trial
                res = trial_res
                jac = trial_jac
                new_cost = trial_cost
                lam = max(lam * 0.3, 1e-12)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            converged = True  # no downhill step left: at the minimum (to precision)
            break
        change = cost - new_cost
        cost = new_cost
        if change <= _FTOL * (1.0 + cost) or step_size <= _XTOL * (1.0 + np.max(np.abs(theta))):
            converged = True
            break
    else:
        # Out of iterations while still creeping downhill: Gauss-Newton converges
        # only linearly when the residual curvature matters (many empty bins). A
        # last relative change this small moves the parameters far less than their
        # statistical errors, so the fit is taken as converged.
        converged = change <= _STALL_TOL * (1.0 + cost)
    return converged, theta, jac.T @ jac, cost


def _fit_window(
    x: FloatArray, counts: FloatArray, mu0: float, sigma0: float
) -> tuple[FloatArray, FloatArray, float] | None:
    """One Poisson-ML fit of Gaussian + linear background on the window bins.

    Parameters (standardised so the tolerances are scale free): ``ln A``,
    ``(mu - mu0) / sigma0``, ``ln(sigma / sigma0)``, and the background at the
    window's first and last bin centres (>= 0).

    Returns:
        ``(parameters [A, mu, sigma, b_lo, b_hi], covariance, deviance)`` or None.
    """
    lo, hi = float(x[0]), float(x[-1])
    z0 = (x - mu0) / sigma0
    t = (x - lo) / (hi - lo)
    amp0 = max(float(counts.max()), 1.0)
    edge = max(0.25 * (float(counts[0]) + float(counts[-1])), 0.0)
    theta0 = np.array([math.log(amp0), 0.0, 0.0, edge, edge], dtype=np.float64)
    try:
        converged, theta, jtj, deviance = _levenberg_marquardt(
            theta0, np.ascontiguousarray(z0), np.ascontiguousarray(t), counts.astype(np.float64)
        )
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return None
    if not converged or not np.all(np.isfinite(theta)):
        return None
    log_amp, m, log_s, b0, b1 = (float(v) for v in theta)
    if abs(log_amp) >= _LOG_CLIP or abs(log_s) >= _LOG_CLIP:
        return None
    amp, s = math.exp(log_amp), math.exp(log_s)
    params = np.array([amp, mu0 + sigma0 * m, sigma0 * s, b0, b1], dtype=np.float64)
    # Covariance of (ln A, m, ln s, b0, b1) -> (A, mu, sigma, b0, b1).
    try:
        cov_theta = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        cov_theta = np.linalg.pinv(jtj)
    scale = np.diag([amp, sigma0, sigma0 * s, 1.0, 1.0])
    with np.errstate(all="ignore"):
        cov = scale @ cov_theta @ scale
    return params, cov, float(deviance)


def fit_photopeak(
    values: npt.ArrayLike,
    *,
    hist_lo: float = 0.80,
    hist_hi: float = 1.15,
    hist_bin: float = 0.0025,
    window_lo_sigma: float = 2.0,
    window_hi_sigma: float = 3.0,
    window_iterations: int = 3,
    seed_mu: float | None = None,
    seed_sigma: float | None = None,
    adapt_width: bool = False,
    mode_range: tuple[float, float] | None = None,
) -> PeakFit:
    """Fit the photopeak of ``values`` (see the module docstring).

    Args:
        values: Normalised energies ``x`` (non-finite values are ignored).
        hist_lo: Histogram lower edge.
        hist_hi: Histogram upper edge.
        hist_bin: Histogram bin width.
        window_lo_sigma: Window extent below the mean, in sigmas.
        window_hi_sigma: Window extent above the mean, in sigmas.
        window_iterations: Most fits (window re-centrings).
        seed_mu: Mean seed (default: the mode of the smoothed histogram).
        mode_range: Search the mode only in this range of x (default: the
            whole histogram).
        seed_sigma: Sigma that sets the window width (default: from the
            smoothed half maximum).
        adapt_width: Also resize the window to each fit's sigma. Off by
            default: with a free background the width can collapse over the
            iterations on a few hundred events, and a slice's own width is
            noisier than the pooled one it is seeded with.

    Returns:
        The fit; ``ok`` is False with a ``message`` when no plausible peak is found.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    centres, counts = histogram(v, hist_lo, hist_hi, hist_bin)
    if counts.sum() < MIN_WINDOW_COUNTS:
        return _fail("too few counts in the histogram", n)
    smoothed = smooth(counts)
    candidates = np.arange(len(centres))
    if mode_range is not None:
        inside = candidates[(centres >= mode_range[0]) & (centres <= mode_range[1])]
        candidates = inside if len(inside) else candidates
    i_max = int(candidates[np.argmax(smoothed[candidates])])
    mu = float(centres[i_max]) if seed_mu is None or not math.isfinite(seed_mu) else seed_mu
    if seed_sigma is not None and math.isfinite(seed_sigma) and seed_sigma > 0:
        sigma = float(seed_sigma)
    else:
        sigma = _seed_sigma(centres, smoothed, i_max)
    sigma = max(sigma, hist_bin)

    result: tuple[FloatArray, FloatArray, float] | None = None
    window = (math.nan, math.nan)
    n_bins = len(centres)
    previous: tuple[int, int] | None = None
    for _ in range(max(1, window_iterations)):
        i_lo = int(np.floor((mu - window_lo_sigma * sigma - hist_lo) / hist_bin))
        i_hi = int(np.ceil((mu + window_hi_sigma * sigma - hist_lo) / hist_bin))
        i_lo, i_hi = max(i_lo, 0), min(i_hi, n_bins - 1)
        if i_hi - i_lo + 1 < MIN_WINDOW_BINS:
            centre = int(np.clip(round((mu - hist_lo) / hist_bin), 0, n_bins - 1))
            i_lo = max(0, min(centre - MIN_WINDOW_BINS // 2, n_bins - MIN_WINDOW_BINS))
            i_hi = i_lo + MIN_WINDOW_BINS - 1
        if previous == (i_lo, i_hi):
            break
        previous = (i_lo, i_hi)
        x_win, c_win = centres[i_lo : i_hi + 1], counts[i_lo : i_hi + 1]
        if c_win.sum() < MIN_WINDOW_COUNTS:
            return _fail("too few counts in the fit window", n)
        fitted = _fit_window(x_win, c_win, mu, sigma)
        if fitted is None:
            return _fail("the optimiser failed", n)
        result = fitted
        window = (float(x_win[0]), float(x_win[-1]))
        new_mu, new_sigma = float(fitted[0][1]), float(fitted[0][2])
        if not (hist_lo < new_mu < hist_hi) or not (0.5 * hist_bin < new_sigma < 0.25):
            return _fail("implausible peak (mean outside the histogram or sigma out of range)", n)
        mu = new_mu
        if adapt_width:
            sigma = new_sigma

    assert result is not None
    params, cov, deviance = result
    amp, mu, sigma, b0, b1 = (float(p) for p in params)
    if not (window[0] <= mu <= window[1]):
        return _fail("the mean left the fit window", n)
    variances = np.diag(cov)
    mu_err = math.sqrt(variances[1]) if variances[1] > 0 else math.nan
    sigma_err = math.sqrt(variances[2]) if variances[2] > 0 else math.nan
    if not math.isfinite(mu_err):
        return _fail("no error estimate", n)
    n_win = int(round((window[1] - window[0]) / hist_bin)) + 1
    ndf = max(n_win - 5, 1)
    return PeakFit(
        ok=True,
        mu=mu,
        mu_err=mu_err,
        sigma=sigma,
        sigma_err=sigma_err,
        amplitude=amp,
        background=(b0, b1),
        window=window,
        n=n,
        chi2ndf=deviance / ndf,
    )


# ---------------------------------------------------------------------------
# Resolution metrics
# ---------------------------------------------------------------------------


def kde_spectrum(
    values: npt.ArrayLike,
    bandwidth: float = KDE_BANDWIDTH,
    lo: float = KDE_RANGE[0],
    hi: float = KDE_RANGE[1],
    bin_width: float = KDE_BIN,
) -> tuple[FloatArray, FloatArray]:
    """Gaussian-kernel-smoothed spectrum of ``values`` on a fine grid.

    The values are histogrammed in ``bin_width`` bins on ``[lo, hi)`` and the
    histogram convolved with a Gaussian of sigma ``bandwidth`` (truncated at
    4 sigma): a kernel density estimate, up to the normalisation.

    Returns:
        ``(bin centres, smoothed counts)``.
    """
    centres, counts = histogram(values, lo, hi, bin_width)
    half = int(math.ceil(4.0 * bandwidth / bin_width))
    offsets = np.arange(-half, half + 1) * bin_width
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    kernel /= kernel.sum()
    return centres, np.asarray(np.convolve(counts, kernel, mode="same"), dtype=np.float64)


@dataclass(frozen=True)
class WidthCrossings:
    """Where :func:`width_pct` measured a width (for drawing it).

    Attributes:
        left: Low crossing (x).
        right: High crossing (x).
        level: The level crossed (smoothed counts per grid step).
        peak: Refined peak position (x).
        continuum: The continuum level subtracted.
    """

    left: float
    right: float
    level: float
    peak: float
    continuum: float

    @property
    def width_pct(self) -> float:
        return 100.0 * (self.right - self.left) / self.peak


def width_crossings(
    values: npt.ArrayLike, fraction: float, bandwidth: float = KDE_BANDWIDTH
) -> WidthCrossings | None:
    """The crossings behind :func:`width_pct` (None where it gives NaN)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < MIN_METRIC_EVENTS:
        return None
    centres, density = kde_spectrum(v, bandwidth)
    search = np.flatnonzero((centres >= PEAK_SEARCH[0]) & (centres <= PEAK_SEARCH[1]))
    i = int(search[np.argmax(density[search])])
    top = density[i]
    if i == 0 or i == len(density) - 1:
        return None
    peak_x = centres[i]
    far, near = CONTINUUM_OFFSETS
    window = (centres >= peak_x - far) & (centres <= peak_x - near)
    continuum = float(np.median(density[window])) if window.any() else 0.0
    if continuum > MAX_CONTINUUM_FRACTION * top:
        return None  # no peak standing out of the continuum
    level = continuum + fraction * (top - continuum)
    left = i
    while left > 0 and density[left] > level:
        left -= 1
    right = i
    while right < len(density) - 1 and density[right] > level:
        right += 1
    if density[right] > level or centres[left] < peak_x - near:
        return None

    def cross(a: int, b: int) -> float:
        # density[a] <= level < density[b]
        da, db = density[a], density[b]
        return float(centres[a] + (level - da) * (centres[b] - centres[a]) / (db - da))

    denom = density[i - 1] - 2.0 * top + density[i + 1]
    shift = 0.5 * (density[i - 1] - density[i + 1]) / denom if denom < 0 else 0.0
    peak = float(peak_x + shift * (centres[1] - centres[0]))
    return WidthCrossings(
        cross(left, left + 1), cross(right, right - 1), float(level), peak, continuum
    )


def width_pct(values: npt.ArrayLike, fraction: float, bandwidth: float = KDE_BANDWIDTH) -> float:
    """Full width of the net photopeak at ``fraction`` of its height, in % of the peak.

    On :func:`kde_spectrum`: the maximum is searched in :data:`PEAK_SEARCH`
    (the continuum of a weak 511 keV peak can be higher further down); the
    continuum level ``B`` is the median of the spectrum
    :data:`CONTINUUM_OFFSETS` below the peak; the crossings of ``B + fraction
    * (max - B)`` are found by walking outwards from the maximum and
    interpolating linearly between grid points (:func:`width_crossings`). The
    peak position is the maximum's grid point refined by a parabola through
    its neighbours.

    Returns:
        The width in %, or NaN below ``MIN_METRIC_EVENTS`` values, when the
        continuum is above ``MAX_CONTINUUM_FRACTION`` of the maximum, or when
        the low crossing is not above the continuum window (the tail merges
        with the continuum).
    """
    crossings = width_crossings(values, fraction, bandwidth)
    return crossings.width_pct if crossings is not None else math.nan


def fwhm_pct(values: npt.ArrayLike, bandwidth: float = KDE_BANDWIDTH) -> float:
    """Full width at half maximum of the net photopeak (see :func:`width_pct`), in %."""
    return width_pct(values, 0.5, bandwidth)


def fwtm_pct(values: npt.ArrayLike, bandwidth: float = KDE_BANDWIDTH) -> float:
    """Full width at a tenth of the net photopeak (see :func:`width_pct`), in %."""
    return width_pct(values, 0.1, bandwidth)
