"""Exact replica of the legacy C++ Dcalib (plan 4.2, 5.6 and decision D4).

The replica runs on the same :class:`~dcalib.events.BoardEvents` as the
default method, but reproduces ``~/DataProcessing/Dcalib/main.cpp`` step by
step so that its ``.dcc`` can be cross-checked against the ROOT binary run on
``.chd`` files dumped from the same events (``scripts/dump_chd.py``). For one
anode (one ``.chd`` file), with the rows in CTS order:

1. **Import** (``importCHD``). An anode without a calibration skips the whole
   file. Every (anode, cathode) pair is converted to keV; the pair is kept if
   ``AN_ELOW <= A <= AN_EHIGH`` and the cathode has a calibration and
   ``0 <= C/A <= 1`` (all inclusive). Fewer than ``MIN_DATA`` kept pairs skip
   the file; that count is taken *before* the next step.
2. **EOF quirk** (on by default). ``while (!eof)`` runs once more after the last
   line pair: every ``>>`` fails and leaves its variable unchanged, and the
   loop body runs again on the last pair. Because ``aEn`` and ``cEn`` were
   converted to keV *in place*, that pass converts the anode energy a second
   time (``A' = A * slope + intercept``) and, if the last real pair reached
   the cathode conversion, the cathode energy too. The resulting pair almost
   never passes the gates. Then ``pop_back()`` removes the last kept pair, so
   in practice the last *real* kept pair is lost; only when the doubly
   converted pair passes the gates is it the one removed. With the quirk off
   there is no extra pass and no ``pop_back``.
3. **Histogram**: ``TH2D`` with x = C/A (``CA_BINS`` bins on ``[MIN_CABIN,
   MAX_CABIN)``) and y = anode keV (``AN_BINS`` on ``[MIN_ANBIN, MAX_ANBIN)``).
   Bins follow ``TAxis::FindBin`` exactly: ``1 + int(n * (v - lo) / (hi -
   lo))``, underflow 0, overflow ``n + 1``.
4. **Peaks**: per x bin, the y bin with the largest content (first maximum
   wins); empty columns are skipped. Bin centres follow ``TAxis::GetBinCenter``.
   Peaks lower than ``MIN_MIN2MAX_PEAKRATIO`` times the tallest are dropped.
5. **Fit**: ``pol2`` to the (C/A, keV) peak points, unweighted, with
   ``p2 in [-1e5, 0]`` (concave only). ROOT minimises this with Minuit
   (option ``"SB"``); the replica solves the same bounded linear least-squares
   problem exactly with :func:`scipy.optimize.lsq_linear`. At an interior
   optimum this is the ordinary pol2 fit, at the bound the pol1 fit with
   ``p2 = 0``. With fewer than 3 peaks the C++ result is undefined; the
   replica records ``fit_failed`` and writes no line.
6. **Export**: ``node board rena channel p0 p1 p2 `` (:mod:`dcalib.io.dcc`).

Two constant sets exist: the 511 keV build (Ge-68 events, the default) and the
662 keV set commented out in ``main.cpp`` (Cs-137 events, ±20 % window).
The replica never touches the sidecar results file.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.optimize import lsq_linear

from dcalib.calib import BoardLUT, Calibrations
from dcalib.channels import AnodeKey
from dcalib.events import DEFAULT_CTS_WINDOW, BoardEvents, build_board_events, list_boards
from dcalib.options import (
    SOURCE_CS,
    SOURCE_GE,
    STATUS_FIT_FAILED,
    STATUS_NO_ANODE_CALIBRATION,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LEGACY_511",
    "LEGACY_662",
    "LEGACY_STATUSES",
    "LegacyConstants",
    "LegacyResult",
    "PeakPoints",
    "find_peaks",
    "fit_concave_pol2",
    "import_pairs",
    "legacy_all",
    "legacy_anode",
    "legacy_board",
    "legacy_constants",
    "root_bin_center",
    "root_find_bin",
]

LEGACY_STATUSES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_NO_ANODE_CALIBRATION,
    STATUS_TOO_FEW_EVENTS,
    STATUS_FIT_FAILED,
)
"""Statuses of the replica: written, anode skipped, fewer than MIN_DATA pairs, < 3 peaks."""


@dataclass(frozen=True)
class LegacyConstants:
    """The ``#define`` constants of ``main.cpp`` (one build).

    The energy window is computed like the C++ macros (``511*.9`` etc., in
    double precision), so the gates agree bit for bit.
    """

    energy: int
    source: int
    an_elow: float
    an_ehigh: float
    min_anbin: float
    max_anbin: float
    c2a_low: float = 0.0
    c2a_high: float = 1.0
    ca_bins: int = 50
    an_bins: int = 50
    min_cabin: float = 0.0
    max_cabin: float = 1.2
    min_data: int = 20
    min_peak_ratio: float = 0.25
    p2_min: float = -100000.0
    p2_max: float = 0.0


LEGACY_511 = LegacyConstants(
    energy=511,
    source=SOURCE_GE,
    an_elow=511 * 0.9,
    an_ehigh=511 * 1.1,
    min_anbin=300.0,
    max_anbin=600.0,
)
"""The compiled 511 keV build (``main.cpp`` lines 26-44), run on the Ge-68 events."""

LEGACY_662 = LegacyConstants(
    energy=662,
    source=SOURCE_CS,
    an_elow=662 * 0.8,
    an_ehigh=662 * 1.2,
    min_anbin=400.0,
    max_anbin=750.0,
)
"""The commented-out 662 keV set (``main.cpp`` lines 46-57), run on the Cs-137 events."""


def legacy_constants(energy: int) -> LegacyConstants:
    """Return the constant set of a photopeak energy (511 or 662 keV).

    Raises:
        ValueError: For any other energy.
    """
    for constants in (LEGACY_511, LEGACY_662):
        if constants.energy == energy:
            return constants
    raise ValueError(f"legacy energy must be 511 or 662, got {energy}")


# ---------------------------------------------------------------------------
# ROOT axis semantics
# ---------------------------------------------------------------------------


def root_find_bin(values: npt.ArrayLike, nbins: int, lo: float, hi: float) -> npt.NDArray[np.int64]:
    """``TAxis::FindBin`` for a fixed-width axis, vectorised.

    Returns 0 below ``lo``, ``nbins + 1`` at or above ``hi`` (and for NaN), and
    ``1 + int(nbins * (v - lo) / (hi - lo))`` otherwise, evaluated in the same
    order as ROOT so edge values land in the same bin.
    """
    v = np.asarray(values, dtype=np.float64)
    bins = np.full(v.shape, nbins + 1, dtype=np.int64)
    below = v < lo
    inside = ~below & (v < hi)
    bins[below] = 0
    bins[inside] = 1 + ((float(nbins) * (v[inside] - lo)) / (hi - lo)).astype(np.int64)
    return bins


def root_bin_center(
    bins: npt.ArrayLike, nbins: int, lo: float, hi: float
) -> npt.NDArray[np.float64]:
    """``TAxis::GetBinCenter`` for a fixed-width axis: ``lo + (bin-1)*w + 0.5*w``."""
    width = (hi - lo) / float(nbins)
    b = np.asarray(bins, dtype=np.float64)
    return np.asarray(lo + (b - 1.0) * width + 0.5 * width, dtype=np.float64)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def import_pairs(
    anode_pha: npt.ArrayLike,
    cathode_pha: npt.ArrayLike,
    anode_cal: tuple[float, float],
    cathode_slope: npt.ArrayLike,
    cathode_intercept: npt.ArrayLike,
    constants: LegacyConstants = LEGACY_511,
    *,
    eof_quirk: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Step 1-2: convert and gate one anode's pairs like ``importCHD``.

    Args:
        anode_pha: Raw anode PHA of each pair, in file (CTS) order.
        cathode_pha: Raw cathode PHA of each pair.
        anode_cal: The anode's ``(slope, intercept)``.
        cathode_slope: Each pair's cathode slope (NaN when uncalibrated).
        cathode_intercept: Each pair's cathode intercept (NaN when uncalibrated).
        constants: The build's constants.
        eof_quirk: Reproduce the end-of-file re-read and ``pop_back``.

    Returns:
        ``(anode_kev, cathode_kev, n_before_pop)``: the kept pairs after the
        ``pop_back`` (when ``eof_quirk``), and the number of kept pairs the
        ``MIN_DATA`` check sees (before the pop).
    """
    a_pha = np.asarray(anode_pha, dtype=np.float64)
    c_pha = np.asarray(cathode_pha, dtype=np.float64)
    c_slope = np.asarray(cathode_slope, dtype=np.float64)
    c_intercept = np.asarray(cathode_intercept, dtype=np.float64)
    a_slope, a_intercept = float(anode_cal[0]), float(anode_cal[1])
    lo, hi = constants.an_elow, constants.an_ehigh

    a_kev = a_pha * a_slope + a_intercept
    a_pass = (a_kev >= lo) & (a_kev <= hi)
    c_cal = ~np.isnan(c_slope)
    c_kev = c_pha * c_slope + c_intercept
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c_kev / a_kev
    keep = a_pass & c_cal & (ratio >= constants.c2a_low) & (ratio <= constants.c2a_high)
    anode_kev = a_kev[keep]
    cathode_kev = c_kev[keep]

    if eof_quirk and len(a_pha):
        last = len(a_pha) - 1
        # aEn still holds the converted energy of the last pair: converted again.
        a2 = float(a_kev[last]) * a_slope + a_intercept
        if lo <= a2 <= hi and c_cal[last]:
            # cEn was converted only if the last pair got past the anode gate.
            c_state = float(c_kev[last]) if a_pass[last] else float(c_pha[last])
            c2 = c_state * float(c_slope[last]) + float(c_intercept[last])
            r2 = c2 / a2
            if constants.c2a_low <= r2 <= constants.c2a_high:
                anode_kev = np.append(anode_kev, a2)
                cathode_kev = np.append(cathode_kev, c2)
    n_before_pop = len(anode_kev)
    if eof_quirk and n_before_pop >= 1:
        anode_kev = anode_kev[:-1]
        cathode_kev = cathode_kev[:-1]
    return anode_kev, cathode_kev, n_before_pop


@dataclass(frozen=True)
class PeakPoints:
    """The peak points of step 4 (after the 25 % gate), one per kept C/A column."""

    ca: npt.NDArray[np.float64]
    energy: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.ca)


def histogram2d(
    anode_kev: npt.ArrayLike, cathode_kev: npt.ArrayLike, constants: LegacyConstants = LEGACY_511
) -> npt.NDArray[np.int64]:
    """Step 3: the ``TH2D`` contents, indexed ``[x bin, y bin]`` including under/overflow."""
    a = np.asarray(anode_kev, dtype=np.float64)
    ca = np.asarray(cathode_kev, dtype=np.float64) / a
    bx = root_find_bin(ca, constants.ca_bins, constants.min_cabin, constants.max_cabin)
    by = root_find_bin(a, constants.an_bins, constants.min_anbin, constants.max_anbin)
    counts = np.zeros((constants.ca_bins + 2, constants.an_bins + 2), dtype=np.int64)
    np.add.at(counts, (bx, by), 1)
    return counts


def find_peaks(
    anode_kev: npt.ArrayLike, cathode_kev: npt.ArrayLike, constants: LegacyConstants = LEGACY_511
) -> PeakPoints:
    """Steps 3-4: fill the histogram, take each column's first maximum, gate at 25 %."""
    counts = histogram2d(anode_kev, cathode_kev, constants)
    inner = counts[1 : constants.ca_bins + 1, 1 : constants.an_bins + 1]
    y_best = np.argmax(inner, axis=1)  # first maximum, as the strict '>' scan
    peak = inner[np.arange(constants.ca_bins), y_best]
    used = peak > 0
    x_bins = np.flatnonzero(used) + 1
    y_bins = y_best[used] + 1
    values = peak[used]
    if len(values):
        keep = ~(values < constants.min_peak_ratio * float(values.max()))
        x_bins, y_bins, values = x_bins[keep], y_bins[keep], values[keep]
    return PeakPoints(
        ca=root_bin_center(x_bins, constants.ca_bins, constants.min_cabin, constants.max_cabin),
        energy=root_bin_center(y_bins, constants.an_bins, constants.min_anbin, constants.max_anbin),
        counts=values.astype(np.int64),
    )


def fit_concave_pol2(
    x: npt.ArrayLike, y: npt.ArrayLike, constants: LegacyConstants = LEGACY_511
) -> tuple[float, float, float]:
    """Step 5: unweighted least-squares ``p0 + p1 x + p2 x^2`` with ``p2`` bounded.

    Args:
        x: C/A of the peak points.
        y: Anode energy (keV) of the peak points.
        constants: Supplies the bounds of ``p2`` (``[-1e5, 0]``).

    Returns:
        ``(p0, p1, p2)``.

    Raises:
        ValueError: With fewer than 3 points.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if len(xs) < 3:
        raise ValueError(f"a pol2 fit needs at least 3 points, got {len(xs)}")
    design = np.column_stack([np.ones_like(xs), xs, xs * xs])
    result = lsq_linear(
        design,
        ys,
        bounds=([-np.inf, -np.inf, constants.p2_min], [np.inf, np.inf, constants.p2_max]),
        method="bvls",
    )
    p0, p1, p2 = (float(v) for v in result.x)
    return p0, p1, p2


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyResult:
    """The replica's outcome for one anode (one ``.chd`` file).

    Attributes:
        key: The anode.
        status: One of ``LEGACY_STATUSES``.
        n_events: 1A1C events of the source (lines pairs of the ``.chd`` file).
        n_pairs: Kept pairs after the import (after ``pop_back``).
        n_peaks: Peak points after the 25 % gate.
        coefficients: ``(p0, p1, p2)`` in keV when the status is ``ok``.
    """

    key: AnodeKey
    status: str
    n_events: int
    n_pairs: int = 0
    n_peaks: int = 0
    coefficients: tuple[float, float, float] | None = None


def legacy_anode(
    key: AnodeKey,
    anode_pha: npt.ArrayLike,
    cathode_rena: npt.ArrayLike,
    cathode_channel: npt.ArrayLike,
    cathode_pha: npt.ArrayLike,
    lut: BoardLUT,
    constants: LegacyConstants = LEGACY_511,
    *,
    eof_quirk: bool = True,
) -> LegacyResult:
    """Run the replica on one anode's events (in CTS order)."""
    a_pha = np.asarray(anode_pha)
    n_events = len(a_pha)
    if not lut.is_calibrated(key.rena, key.channel):
        return LegacyResult(key, STATUS_NO_ANODE_CALIBRATION, n_events)
    c_r = np.asarray(cathode_rena, dtype=np.intp)
    c_c = np.asarray(cathode_channel, dtype=np.intp)
    anode_kev, cathode_kev, n_checked = import_pairs(
        a_pha,
        cathode_pha,
        (float(lut.slope[key.rena, key.channel]), float(lut.intercept[key.rena, key.channel])),
        lut.slope[c_r, c_c],
        lut.intercept[c_r, c_c],
        constants,
        eof_quirk=eof_quirk,
    )
    if n_checked < constants.min_data:
        return LegacyResult(key, STATUS_TOO_FEW_EVENTS, n_events, len(anode_kev))
    peaks = find_peaks(anode_kev, cathode_kev, constants)
    if len(peaks) < 3:
        return LegacyResult(key, STATUS_FIT_FAILED, n_events, len(anode_kev), len(peaks))
    coefficients = fit_concave_pol2(peaks.ca, peaks.energy, constants)
    if not all(math.isfinite(p) for p in coefficients):
        return LegacyResult(key, STATUS_FIT_FAILED, n_events, len(anode_kev), len(peaks))
    return LegacyResult(key, STATUS_OK, n_events, len(anode_kev), len(peaks), coefficients)


def legacy_board(
    events: BoardEvents,
    lut: BoardLUT,
    constants: LegacyConstants = LEGACY_511,
    *,
    eof_quirk: bool = True,
) -> list[LegacyResult]:
    """Run the replica on every anode of a board that has events of the source.

    Returns:
        One result per anode with at least one 1A1C event of
        ``constants.source`` (an anode without events has no ``.chd`` file),
        sorted by key.
    """
    source_events = events.take(events.source == constants.source)
    results = []
    for (rena, channel), rows in source_events.anode_rows().items():
        key = AnodeKey(events.node, events.board, rena, channel)
        results.append(
            legacy_anode(
                key,
                source_events.anode_pha[rows],
                source_events.cathode_rena[rows],
                source_events.cathode_channel[rows],
                source_events.cathode_pha[rows],
                lut,
                constants,
                eof_quirk=eof_quirk,
            )
        )
    return results


def legacy_all(
    cache_path: Path,
    calibrations: Calibrations,
    constants: LegacyConstants = LEGACY_511,
    *,
    eof_quirk: bool = True,
    cts_window: int = DEFAULT_CTS_WINDOW,
    boards: list[tuple[int, int]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[LegacyResult]:
    """Run the replica on every board of the cache (one process, board by board).

    Args:
        cache_path: The adc2kev calibration cache (read-only).
        calibrations: The calibrations to convert with.
        constants: The build's constants (and so the source).
        eof_quirk: Reproduce the end-of-file quirk.
        cts_window: Event-building window.
        boards: Restrict to these boards (default: every board in the cache).
        progress: Called with ``(boards done, boards total)`` after each board.

    Returns:
        Every anode's result, sorted by key.
    """
    todo = list(boards) if boards is not None else list(list_boards(cache_path))
    results: list[LegacyResult] = []
    for done, (node, board) in enumerate(todo, start=1):
        events = build_board_events(cache_path, node, board, cts_window)
        results.extend(
            legacy_board(
                events, calibrations.board_lut(node, board), constants, eof_quirk=eof_quirk
            )
        )
        if progress is not None:
            progress(done, len(todo))
    return sorted(results, key=lambda r: r.key)
