"""Synthetic anode events with a known depth curve (plan section 10.2).

Each event has a depth proxy ``r`` and a source; its photopeak energy is
``g_source(r)`` in E/E0 units, smeared by a realistic line shape: a Gaussian
core, a low-energy (hole-trapping) tail and a flat background, with a source
dependent share of the events.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Curve = Callable[[FloatArray], FloatArray]


def flat(r: FloatArray) -> FloatArray:
    return np.ones_like(r)


def linear_drop(r: FloatArray) -> FloatArray:
    """A 2 % drop from r = 0 to r = 1."""
    return 1.0 - 0.02 * r


def concave(r: FloatArray) -> FloatArray:
    """A concave pol2 with a 1.7 % spread over r in [0, 1] (the fleet median)."""
    return 0.996 + 0.02 * r - 0.032 * r * r


def steep(r: FloatArray) -> FloatArray:
    """An 8 % spread (the fleet p90)."""
    return 1.01 + 0.03 * r - 0.10 * r * r


@dataclass(frozen=True)
class LineShape:
    sigma: float = 0.024  # Gaussian core, FWHM 5.7 %
    tail_frac: float = 0.15
    tail_scale: float = 0.03
    bkg_frac: float = 0.10
    bkg_range: tuple[float, float] = (0.6, 1.3)
    r_range: tuple[float, float] = (0.02, 1.15)


def simulate(
    n: int,
    curve: Curve,
    seed: int,
    *,
    cs_frac: float = 0.5,
    cs_curve: Curve | None = None,
    shape: LineShape = LineShape(),
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int8]]:
    """Return ``(x, r, source)`` for ``n`` events (rows interleaved like CTS order)."""
    rng = np.random.default_rng(seed)
    source = (rng.random(n) < cs_frac).astype(np.int8)
    r = rng.uniform(*shape.r_range, n)
    peak = curve(r)
    if cs_curve is not None:
        peak = np.where(source == 1, cs_curve(r), peak)
    # Cs-137 has a slightly narrower relative resolution than Ge-68 (511 keV)
    sigma = np.where(source == 1, 0.9 * shape.sigma, shape.sigma)
    x = peak + rng.normal(0.0, 1.0, n) * sigma
    u = rng.random(n)
    tail = u < shape.tail_frac
    x[tail] -= rng.exponential(shape.tail_scale, int(tail.sum()))
    bkg = (u >= shape.tail_frac) & (u < shape.tail_frac + shape.bkg_frac)
    x[bkg] = rng.uniform(*shape.bkg_range, int(bkg.sum()))
    r[bkg] = rng.uniform(*shape.r_range, int(bkg.sum()))
    order = np.argsort(source, kind="stable")  # rows by source, like BoardEvents
    return x[order], r[order], source[order]
