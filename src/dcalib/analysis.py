"""Per-anode, per-board and whole-system analysis driver (plan sections 5 and 7).

- :class:`AnodeResult`: every ``depth_summary.csv`` field of one anode (plan
  6.4). Its dataclass fields *are* the CSV columns, in CSV order and with the
  CSV names; :data:`RESULT_COLUMNS` derives the column list from the fields,
  so the CSV writer, the sidecar table and the GUI inspector share one source
  of truth.
- :class:`SliceRow`: one slice point of one curve (the sidecar's ``slices``
  table), from which the GUI draws the curves without refitting.
- :func:`analyze_anode`: the default method on one anode's events.
- :func:`analyze_board`: every anode of one board that has 1A1C events.
- :func:`analyze_all`: every board of the cache, one ``spawn`` pool task per
  board, largest first.

Unavailable values (a failed anode's coefficients, a metric on too few events)
are ``None`` in :class:`AnodeResult`, never NaN: an empty CSV cell, NaN in a
float column of the sidecar table and -1 in its count columns.

Energies, statuses and counts (plan 5.2): ``n_events`` counts the anode's 1A1C
events of both sources; an uncalibrated anode is ``no_anode_calibration``;
events whose cathode has no calibration are dropped and counted
(``n_uncal_cathode``), and an anode left without events is
``no_calibrated_cathode``. Everything else goes to :func:`dcalib.depth.fit_anode`.

Resolution metrics (plan 5.5): per source, on the anode's events with a
calibrated cathode and ``r`` in ``[r_lo, r_hi]``, the FWHM and FWTM of the
smoothed spectrum (:mod:`dcalib.metrics`) before (``x``) and after (``x /
g(r)`` for an ``ok`` anode, ``x`` otherwise). ``peak_511``/``peak_662`` are the
photopeak positions (Gaussian core, E/E0) of the *after* spectra: 1 for a
corrected anode up to the fit's small tail bias, and the ``.kev`` scale of an
omitted one (open item O3). They are extra columns beyond plan 6.4.

Process pool and BLAS threads follow uvcorr: ``spawn`` workers (safe from the
GUI's threads), single-threaded BLAS via ``threadpoolctl``, Ctrl-C ignored in
the workers (the parent sets a shared stop event), ``workers=1`` in-process.
The parent loads the calibrations once and sends each task its board's LUT.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import signal
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from dcalib.calib import BoardLUT, Calibrations, event_energies
from dcalib.channels import AnodeKey, electrode_label
from dcalib.depth import CURVE_NAMES, DepthFit, Slices, fit_anode
from dcalib.events import BoardEvents, build_board_events, list_boards
from dcalib.metrics import MIN_METRIC_EVENTS, fit_photopeak, fwhm_pct, fwtm_pct
from dcalib.options import (
    DCC_ENERGY_KEV,
    FLAG_PARTIAL_CATHODE_COVERAGE,
    FLAG_SEPARATOR,
    REVIEW_STATES,
    SOURCE_CS,
    SOURCE_GE,
    STATUS_NO_ANODE_CALIBRATION,
    STATUS_NO_CALIBRATED_CATHODE,
    STATUS_OK,
    STATUSES,
    DepthOptions,
    order_flags,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CSV_COLUMNS",
    "KIND_COUNT",
    "KIND_FLAGS",
    "KIND_FLOAT",
    "KIND_INT",
    "KIND_STR",
    "OPTIONS_BATCH",
    "OPTIONS_OVERRIDE",
    "OPTIONS_SOURCES",
    "RESULT_COLUMNS",
    "SLICE_COLUMNS",
    "AnalysisCancelled",
    "AnalysisError",
    "AnodeResult",
    "BoardAnalysis",
    "SliceRow",
    "WorkerCrashedError",
    "analyze_all",
    "analyze_anode",
    "analyze_board",
    "default_workers",
    "merge_results",
    "schedule_boards",
]

OPTIONS_BATCH = "batch"
"""``options_source`` of a result computed by a batch run with the run's options."""

OPTIONS_OVERRIDE = "override"
"""``options_source`` of a per-anode re-fit with its own options (a GUI override)."""

OPTIONS_SOURCES: tuple[str, ...] = (OPTIONS_BATCH, OPTIONS_OVERRIDE)

MAX_DEFAULT_WORKERS = 8

ProgressCallback = Callable[[int, int], None]
"""``progress_cb(done_hits, total_hits)``: the hits of the boards finished so far and of all."""

StopFlag = Callable[[], bool] | threading.Event

FloatArray = npt.NDArray[np.float64]


class AnalysisCancelled(Exception):
    """The analysis was stopped through its ``stop_flag``; no results are returned."""


class AnalysisError(Exception):
    """The analysis failed (in a worker process or in-process); the cause is chained."""

    def __init__(self, message: str, node: int | None = None, board: int | None = None) -> None:
        super().__init__(message)
        self.node = node
        self.board = board

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (str(self), self.node, self.board))


WORKER_CRASHED_MESSAGE = (
    "a worker process terminated abruptly (killed, e.g. out of memory, or crashed); "
    "rerun with --workers 1 to locate the board"
)


class WorkerCrashedError(AnalysisError):
    """A worker process died without reporting an error (the board is unknown)."""

    def __init__(
        self,
        message: str = WORKER_CRASHED_MESSAGE,
        node: int | None = None,
        board: int | None = None,
    ) -> None:
        super().__init__(message, node, board)


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------

KIND_INT = "int"
"""Always-present integer (address, event counts)."""

KIND_COUNT = "count"
"""Integer or None (``degree``, ``n_slices``)."""

KIND_STR = "str"
"""Text (``electrode``, ``status``, ``review``, ``options_source``)."""

KIND_FLAGS = "flags"
"""Tuple of flags; ``;``-joined in the CSV and the sidecar."""

KIND_FLOAT = "float"
"""Finite float or None."""


def _col(kind: str, **kwargs: Any) -> Any:
    return field(metadata={"kind": kind}, **kwargs)


def _opt(kind: str) -> Any:
    return _col(kind, default=None)


@dataclass(frozen=True, kw_only=True)
class AnodeResult:
    """Every ``depth_summary.csv`` field of one anode (plan 6.4), in column order.

    Construction normalises the values: numpy scalars become Python scalars,
    non-finite floats become None and ``flags`` a tuple in canonical order.

    Attributes:
        node, board, rena, channel: The anode.
        electrode: ``ElectrodeMap`` label, ``A01``-``A39``.
        status: One of :data:`dcalib.options.STATUSES`.
        flags: Warning flags, canonical order.
        review: ``""`` or ``"rejected"`` (plan D10).
        options_source: ``"batch"`` or ``"override"``.
        n_events: 1A1C events of both sources.
        n_ge, n_cs: 1A1C events per source.
        n_uncal_cathode: Events dropped because their cathode has no calibration.
        n_selected: Events in the final selection of the depth fit.
        ca_p01, ca_p50, ca_p99: r quantiles of the first selection.
        degree: Degree of the pooled curve.
        p0, p1, p2: The curve in ``.dcc`` units, ``511 * c_k``.
        err_p0, err_p1, err_p2: Their standard errors.
        chi2ndf: Reduced chi-square of the curve fit.
        n_slices: Slice points of the pooled curve.
        peak_spread: Largest minus smallest slice position (E/E0).
        ge_cs_max_diff: Largest Ge/Cs curve difference over the 10-90 % r range.
        cv_gain: Cross-fitted relative width gain.
        fwhm_511_before ... fwtm_662_after: Widths in %.
        peak_511, peak_662: Photopeak position (E/E0) of the exported spectrum.
    """

    node: int = _col(KIND_INT)
    board: int = _col(KIND_INT)
    rena: int = _col(KIND_INT)
    channel: int = _col(KIND_INT)
    electrode: str = _col(KIND_STR, default="")
    status: str = _col(KIND_STR)
    flags: tuple[str, ...] = _col(KIND_FLAGS, default=())
    review: str = _col(KIND_STR, default="")
    options_source: str = _col(KIND_STR, default=OPTIONS_BATCH)
    n_events: int = _col(KIND_INT, default=0)
    n_ge: int = _col(KIND_INT, default=0)
    n_cs: int = _col(KIND_INT, default=0)
    n_uncal_cathode: int = _col(KIND_INT, default=0)
    n_selected: int = _col(KIND_INT, default=0)
    ca_p01: float | None = _opt(KIND_FLOAT)
    ca_p50: float | None = _opt(KIND_FLOAT)
    ca_p99: float | None = _opt(KIND_FLOAT)
    degree: int | None = _opt(KIND_COUNT)
    p0: float | None = _opt(KIND_FLOAT)
    p1: float | None = _opt(KIND_FLOAT)
    p2: float | None = _opt(KIND_FLOAT)
    err_p0: float | None = _opt(KIND_FLOAT)
    err_p1: float | None = _opt(KIND_FLOAT)
    err_p2: float | None = _opt(KIND_FLOAT)
    chi2ndf: float | None = _opt(KIND_FLOAT)
    n_slices: int | None = _opt(KIND_COUNT)
    peak_spread: float | None = _opt(KIND_FLOAT)
    ge_cs_max_diff: float | None = _opt(KIND_FLOAT)
    cv_gain: float | None = _opt(KIND_FLOAT)
    fwhm_511_before: float | None = _opt(KIND_FLOAT)
    fwhm_511_after: float | None = _opt(KIND_FLOAT)
    fwhm_662_before: float | None = _opt(KIND_FLOAT)
    fwhm_662_after: float | None = _opt(KIND_FLOAT)
    fwtm_511_before: float | None = _opt(KIND_FLOAT)
    fwtm_511_after: float | None = _opt(KIND_FLOAT)
    fwtm_662_before: float | None = _opt(KIND_FLOAT)
    fwtm_662_after: float | None = _opt(KIND_FLOAT)
    peak_511: float | None = _opt(KIND_FLOAT)
    peak_662: float | None = _opt(KIND_FLOAT)

    def __post_init__(self) -> None:
        for column in RESULT_COLUMNS:
            name, kind = column.name, column.kind
            object.__setattr__(self, name, _normalise(name, kind, getattr(self, name)))
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.review not in REVIEW_STATES:
            raise ValueError(f"review must be one of {REVIEW_STATES}, got {self.review!r}")
        if self.options_source not in OPTIONS_SOURCES:
            raise ValueError(
                f"options_source must be one of {OPTIONS_SOURCES}, got {self.options_source!r}"
            )

    @property
    def key(self) -> AnodeKey:
        return AnodeKey(self.node, self.board, self.rena, self.channel)

    @property
    def ok(self) -> bool:
        """True when the correction was accepted (``status == "ok"``)."""
        return self.status == STATUS_OK

    @property
    def exported(self) -> bool:
        """True when the anode is written to the ``.dcc``: ``ok`` and not rejected."""
        return self.ok and not self.review

    @property
    def coefficients(self) -> tuple[float, float, float] | None:
        """``(p0, p1, p2)`` in ``.dcc`` units, or None without a curve."""
        if self.p0 is None:
            return None
        return (self.p0, self.p1 or 0.0, self.p2 or 0.0)

    def g(self, r: npt.ArrayLike) -> FloatArray:
        """The curve in E/E0 units, ``(p0 + p1 r + p2 r^2) / 511`` (NaN without one)."""
        rr = np.asarray(r, dtype=np.float64)
        c = self.coefficients
        if c is None:
            return np.full(rr.shape, np.nan)
        return np.asarray((c[0] + rr * (c[1] + rr * c[2])) / DCC_ENERGY_KEV, dtype=np.float64)

    @property
    def flags_text(self) -> str:
        return FLAG_SEPARATOR.join(self.flags)

    def to_dict(self) -> dict[str, Any]:
        return {column.name: getattr(self, column.name) for column in RESULT_COLUMNS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> AnodeResult:
        """Build a result from ``{column: value}`` (``flags`` may be the ``;``-joined text)."""
        unknown = sorted(set(data) - set(CSV_COLUMNS))
        if unknown:
            if strict:
                raise ValueError(f"Unknown AnodeResult column(s): {unknown}")
            logger.warning("Ignoring unknown AnodeResult column(s): %s", unknown)
        values = {name: data[name] for name in CSV_COLUMNS if name in data}
        flags = values.get("flags")
        if isinstance(flags, str):
            values["flags"] = tuple(part for part in flags.split(FLAG_SEPARATOR) if part)
        return cls(**values)


class ResultColumn(NamedTuple):
    name: str
    kind: str


RESULT_COLUMNS: tuple[ResultColumn, ...] = tuple(
    ResultColumn(f.name, f.metadata["kind"]) for f in fields(AnodeResult)
)
"""The ``depth_summary.csv`` columns in order, derived from :class:`AnodeResult`."""

CSV_COLUMNS: tuple[str, ...] = tuple(column.name for column in RESULT_COLUMNS)


def _normalise(name: str, kind: str, value: Any) -> Any:
    if kind == KIND_INT:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"AnodeResult.{name} must be an int, got {value!r}")
        return int(value)
    if kind == KIND_COUNT:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"AnodeResult.{name} must be an int or None, got {value!r}")
        if value < 0:
            raise ValueError(f"AnodeResult.{name} must be >= 0, got {value}")
        return int(value)
    if kind == KIND_FLOAT:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"AnodeResult.{name} must be a float or None, got {value!r}")
        number = float(value)
        return number if math.isfinite(number) else None
    if kind == KIND_STR:
        if not isinstance(value, str):
            raise TypeError(f"AnodeResult.{name} must be a str, got {value!r}")
        return value
    if kind == KIND_FLAGS:
        if isinstance(value, str):
            raise TypeError(f"AnodeResult.{name} must be a sequence of flags, got {value!r}")
        return order_flags(value)
    raise AssertionError(f"unknown column kind {kind!r}")


class SliceRow(NamedTuple):
    """One slice point of one curve of one anode (the sidecar ``slices`` table)."""

    node: int
    board: int
    rena: int
    channel: int
    curve: str
    slice: int
    r_lo: float
    r_hi: float
    r_med: float
    mu: float
    mu_err: float
    sigma: float
    n: int

    @property
    def key(self) -> AnodeKey:
        return AnodeKey(self.node, self.board, self.rena, self.channel)


SLICE_COLUMNS: tuple[str, ...] = SliceRow._fields


def slice_rows(key: AnodeKey, slices: Mapping[str, Slices]) -> list[SliceRow]:
    """Flatten a fit's slices (per curve name) into :class:`SliceRow` rows."""
    rows: list[SliceRow] = []
    for curve in CURVE_NAMES:
        sl = slices.get(curve)
        if sl is None:
            continue
        for i in range(len(sl)):
            rows.append(
                SliceRow(
                    *key,
                    curve,
                    i,
                    float(sl.r_lo[i]),
                    float(sl.r_hi[i]),
                    float(sl.r_med[i]),
                    float(sl.mu[i]),
                    float(sl.mu_err[i]),
                    float(sl.sigma[i]),
                    int(sl.n[i]),
                )
            )
    return rows


def slices_from_rows(rows: Iterable[SliceRow]) -> dict[str, Slices]:
    """Inverse of :func:`slice_rows` for one anode: ``{curve name: Slices}``."""
    grouped: dict[str, list[SliceRow]] = {}
    for row in rows:
        grouped.setdefault(row.curve, []).append(row)
    out: dict[str, Slices] = {}
    for curve, items in grouped.items():
        items.sort(key=lambda r: r.slice)
        out[curve] = Slices(
            r_lo=np.array([r.r_lo for r in items]),
            r_hi=np.array([r.r_hi for r in items]),
            r_med=np.array([r.r_med for r in items]),
            mu=np.array([r.mu for r in items]),
            mu_err=np.array([r.mu_err for r in items]),
            sigma=np.array([r.sigma for r in items]),
            n=np.array([r.n for r in items], dtype=np.int64),
        )
    return out


# ---------------------------------------------------------------------------
# Per-anode analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnodeInput:
    """One anode's events, ready for the depth fit (see :func:`anode_inputs`).

    ``x``, ``r`` and ``source`` hold the events with a calibrated cathode, in
    the anode's row order; the counts cover all of its 1A1C events.
    """

    key: AnodeKey
    anode_calibrated: bool
    n_events: int
    n_ge: int
    n_cs: int
    n_uncal_cathode: int
    x: FloatArray
    r: FloatArray
    source: npt.NDArray[np.int8]


def anode_inputs(events: BoardEvents, lut: BoardLUT) -> Iterator[AnodeInput]:
    """Split a board's events by anode and convert them (plan 5.2), sorted by key."""
    energies = event_energies(events, lut)
    cal_cathode = energies.cathode_calibrated
    for (rena, channel), rows in events.anode_rows().items():
        src = events.source[rows]
        keep = rows[cal_cathode[rows]]
        yield AnodeInput(
            key=AnodeKey(events.node, events.board, rena, channel),
            anode_calibrated=lut.is_calibrated(rena, channel),
            n_events=len(rows),
            n_ge=int(np.count_nonzero(src == SOURCE_GE)),
            n_cs=int(np.count_nonzero(src == SOURCE_CS)),
            n_uncal_cathode=int(len(rows) - len(keep)),
            x=energies.x[keep],
            r=energies.r[keep],
            source=events.source[keep],
        )


def _label(board: int, rena: int, channel: int) -> str:
    try:
        return electrode_label(board, rena, channel)
    except KeyError:
        return ""


def _source_metrics(
    x: FloatArray,
    r: FloatArray,
    source: npt.NDArray[np.int8],
    fit: DepthFit | None,
    options: DepthOptions,
) -> dict[str, float | None]:
    """FWHM/FWTM before and after, and the exported peak position, per source."""
    out: dict[str, float | None] = {}
    in_r = (r >= options.r_lo) & (r <= options.r_hi) & np.isfinite(x) & np.isfinite(r)
    for source_id, energy in ((SOURCE_GE, 511), (SOURCE_CS, 662)):
        mask = in_r & (source == source_id)
        before = x[mask]
        after = before / fit.correction(r[mask]) if fit is not None else before
        out[f"fwhm_{energy}_before"] = fwhm_pct(before)
        out[f"fwtm_{energy}_before"] = fwtm_pct(before)
        out[f"fwhm_{energy}_after"] = fwhm_pct(after)
        out[f"fwtm_{energy}_after"] = fwtm_pct(after)
        peak = fit_photopeak(after) if len(after) >= MIN_METRIC_EVENTS else None
        out[f"peak_{energy}"] = peak.mu if peak is not None and peak.ok else None
    return out


def analyze_anode(
    anode: AnodeInput,
    options: DepthOptions | None = None,
    *,
    options_source: str = OPTIONS_BATCH,
) -> tuple[AnodeResult, list[SliceRow]]:
    """Run the default method on one anode and build its CSV row and slice rows."""
    opts = options if options is not None else DepthOptions()
    key = anode.key
    base: dict[str, Any] = {
        "node": key.node,
        "board": key.board,
        "rena": key.rena,
        "channel": key.channel,
        "electrode": _label(key.board, key.rena, key.channel),
        "options_source": options_source,
        "n_events": anode.n_events,
        "n_ge": anode.n_ge,
        "n_cs": anode.n_cs,
        "n_uncal_cathode": anode.n_uncal_cathode,
    }
    flags: list[str] = []
    if anode.n_events and anode.n_uncal_cathode / anode.n_events > opts.partial_cathode_frac:
        flags.append(FLAG_PARTIAL_CATHODE_COVERAGE)
    if not anode.anode_calibrated:
        return AnodeResult(**base, status=STATUS_NO_ANODE_CALIBRATION, flags=tuple(flags)), []
    if len(anode.x) == 0:
        return AnodeResult(**base, status=STATUS_NO_CALIBRATED_CATHODE, flags=tuple(flags)), []

    fit = fit_anode(anode.x, anode.r, anode.source, opts, extra_flags=flags)
    values = dict(base)
    values.update(
        status=fit.status,
        flags=fit.flags,
        n_selected=fit.n_selected,
        ca_p01=fit.ca_quantiles[0],
        ca_p50=fit.ca_quantiles[1],
        ca_p99=fit.ca_quantiles[2],
        peak_spread=fit.peak_spread,
        ge_cs_max_diff=fit.ge_cs_max_diff,
        cv_gain=fit.cv_gain,
    )
    pooled = fit.slices.get("both")
    if pooled is not None:
        values["n_slices"] = len(pooled)
    if fit.curve is not None:
        curve = fit.curve
        values.update(
            degree=curve.degree,
            p0=DCC_ENERGY_KEV * curve.coefficients[0],
            p1=DCC_ENERGY_KEV * curve.coefficients[1],
            p2=DCC_ENERGY_KEV * curve.coefficients[2],
            err_p0=DCC_ENERGY_KEV * curve.errors[0],
            err_p1=DCC_ENERGY_KEV * curve.errors[1],
            err_p2=DCC_ENERGY_KEV * curve.errors[2],
            chi2ndf=curve.chi2ndf,
        )
    values.update(_source_metrics(anode.x, anode.r, anode.source, fit, opts))
    return AnodeResult(**values), slice_rows(key, fit.slices)


# ---------------------------------------------------------------------------
# Per-board analysis
# ---------------------------------------------------------------------------


@dataclass
class BoardAnalysis:
    """The results and slice rows of some anodes (a board, a system)."""

    results: list[AnodeResult] = field(default_factory=list)
    slices: list[SliceRow] = field(default_factory=list)

    def extend(self, other: BoardAnalysis) -> None:
        self.results.extend(other.results)
        self.slices.extend(other.slices)

    def sort(self) -> None:
        self.results.sort(key=lambda r: r.key)
        self.slices.sort(key=lambda s: (s.key, CURVE_NAMES.index(s.curve), s.slice))


def _stop_callable(stop_flag: StopFlag | None) -> Callable[[], bool]:
    if stop_flag is None:
        return lambda: False
    if isinstance(stop_flag, threading.Event):
        return stop_flag.is_set
    if callable(stop_flag):
        return stop_flag
    raise TypeError(f"stop_flag must be a callable or threading.Event, got {type(stop_flag)}")


def analyze_board(
    cache_path: str | Path | None,
    node: int,
    board: int,
    lut: BoardLUT,
    options: DepthOptions | None = None,
    *,
    events: BoardEvents | None = None,
    anodes: Iterable[tuple[int, int]] | None = None,
    options_source: str = OPTIONS_BATCH,
    stop_flag: StopFlag | None = None,
) -> BoardAnalysis:
    """Analyse the anodes of one board that have 1A1C events.

    Args:
        cache_path: The adc2kev cache (read-only); unused when ``events`` is given.
        node: Node number.
        board: Board number.
        lut: The board's calibration.
        options: Depth-fit options (default ``DepthOptions()``); their
            ``cts_window`` builds the events.
        events: The board's events, already built (the GUI's board cache).
        anodes: Restrict to these ``(rena, channel)`` anodes.
        options_source: ``options_source`` of the rows.
        stop_flag: Checked before every anode.

    Returns:
        The results (sorted by key) and their slice rows.

    Raises:
        AnalysisCancelled: If ``stop_flag`` requested a stop.
    """
    opts = options if options is not None else DepthOptions()
    should_stop = _stop_callable(stop_flag)
    if events is None:
        if cache_path is None:
            raise ValueError("analyze_board needs a cache path or the board's events")
        events = build_board_events(Path(cache_path), node, board, opts.cts_window)
    wanted = None if anodes is None else {(int(r), int(c)) for r, c in anodes}
    out = BoardAnalysis()
    for anode in anode_inputs(events, lut):
        if wanted is not None and (anode.key.rena, anode.key.channel) not in wanted:
            continue
        if should_stop():
            raise AnalysisCancelled(f"Analysis of node {node} board {board} cancelled")
        result, slices = analyze_anode(anode, opts, options_source=options_source)
        out.results.append(result)
        out.slices.extend(slices)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Whole-system analysis
# ---------------------------------------------------------------------------


def default_workers() -> int:
    """Default worker processes: ``min(8, usable CPUs)``."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_DEFAULT_WORKERS, cpus))


_worker_stop: Any = None
_worker_blas_limit: Any = None


def _init_worker(stop_event: Any) -> None:
    """Pool initializer: single-threaded BLAS, Ctrl-C ignored, the shared stop event."""
    global _worker_stop, _worker_blas_limit
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    _worker_stop = stop_event
    from threadpoolctl import threadpool_limits

    _worker_blas_limit = threadpool_limits(limits=1)


def _board_task(
    cache_path: str, node: int, board: int, lut: BoardLUT, options: DepthOptions
) -> BoardAnalysis:
    stop = _worker_stop.is_set if _worker_stop is not None else None
    return analyze_board(cache_path, node, board, lut, options, stop_flag=stop)


@contextmanager
def _sigint_blocked() -> Iterator[None]:
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def schedule_boards(counts: Mapping[tuple[int, int], int]) -> list[tuple[int, int]]:
    """Largest boards first (then by (node, board)): keeps the tail of a pooled run short."""
    return sorted(counts, key=lambda nb: (-counts[nb], nb))


class _Progress:
    def __init__(self, counts: Mapping[tuple[int, int], int], callback: ProgressCallback | None):
        self._counts = counts
        self._callback = callback
        self._total = int(sum(counts.values()))
        self._done = 0

    def start(self) -> None:
        if self._callback is not None:
            self._callback(0, self._total)

    def board_done(self, node: int, board: int) -> None:
        self._done += int(self._counts[(node, board)])
        if self._callback is not None:
            self._callback(self._done, self._total)


def _board_error(node: int, board: int, exc: BaseException) -> AnalysisError:
    return AnalysisError(
        f"Analysis of node {node} board {board} failed: {type(exc).__name__}: {exc}", node, board
    )


def analyze_all(
    cache_path: str | Path,
    calibrations: Calibrations,
    options: DepthOptions | None = None,
    workers: int | None = None,
    progress_cb: ProgressCallback | None = None,
    stop_flag: StopFlag | None = None,
    boards: Iterable[tuple[int, int]] | None = None,
) -> BoardAnalysis:
    """Analyse every board of the cache (see the module docstring for the pool).

    Args:
        cache_path: The adc2kev calibration cache (read-only).
        calibrations: The calibrations (loaded once, in the caller).
        options: Depth-fit options (default ``DepthOptions()``).
        workers: Worker processes (default :func:`default_workers`); at most
            one per board; 1 runs in-process.
        progress_cb: ``progress_cb(done_hits, total_hits)``, once with 0, then
            after every board.
        stop_flag: Polled about every 0.1 s (between anodes in-process).
        boards: Restrict to these boards (default: every board of the cache).

    Returns:
        Every anode's result and slice rows, sorted.

    Raises:
        AnalysisCancelled: If ``stop_flag`` requested a stop.
        AnalysisError: If a board failed (``node``/``board`` identify it).
        WorkerCrashedError: If a worker process died.
    """
    opts = options if options is not None else DepthOptions()
    should_stop = _stop_callable(stop_flag)
    n_workers = default_workers() if workers is None else int(workers)
    if n_workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    counts = list_boards(Path(cache_path))
    if boards is not None:
        wanted = set(boards)
        counts = {nb: n for nb, n in counts.items() if nb in wanted}
    order = schedule_boards(counts)
    progress = _Progress(counts, progress_cb)
    progress.start()
    out = BoardAnalysis()
    if not order:
        return out
    luts = {nb: calibrations.board_lut(*nb) for nb in order}
    n_workers = min(n_workers, len(order))
    if n_workers == 1:
        for node, board in order:
            if should_stop():
                raise AnalysisCancelled("Analysis cancelled")
            try:
                out.extend(
                    analyze_board(
                        cache_path, node, board, luts[(node, board)], opts, stop_flag=should_stop
                    )
                )
            except AnalysisCancelled:
                raise
            except Exception as exc:
                raise _board_error(node, board, exc) from exc
            progress.board_done(node, board)
    else:
        _analyze_in_pool(str(cache_path), order, luts, opts, n_workers, progress, should_stop, out)
    out.sort()
    return out


_POLL_SECONDS = 0.1


def _analyze_in_pool(
    cache_path: str,
    order: list[tuple[int, int]],
    luts: Mapping[tuple[int, int], BoardLUT],
    options: DepthOptions,
    n_workers: int,
    progress: _Progress,
    should_stop: Callable[[], bool],
    out: BoardAnalysis,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    futures: dict[Future[BoardAnalysis], tuple[int, int]] = {}
    completed = False
    with _sigint_blocked():
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(stop_event,),
        )
    try:
        with _sigint_blocked():
            for node, board in order:
                future = executor.submit(
                    _board_task, cache_path, node, board, luts[(node, board)], options
                )
                futures[future] = (node, board)
        pending: set[Future[BoardAnalysis]] = set(futures)
        while pending:
            if should_stop():
                raise AnalysisCancelled("Analysis cancelled")
            done, pending = wait(pending, timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                node, board = futures[future]
                try:
                    board_out = future.result()
                except BrokenProcessPool as exc:
                    if should_stop():
                        raise AnalysisCancelled("Analysis cancelled") from exc
                    raise WorkerCrashedError() from exc
                except Exception as exc:
                    if should_stop():
                        raise AnalysisCancelled("Analysis cancelled") from exc
                    raise _board_error(node, board, exc) from exc
                out.extend(board_out)
                progress.board_done(node, board)
        completed = True
    except BrokenProcessPool as exc:
        if should_stop():
            raise AnalysisCancelled("Analysis cancelled") from exc
        raise WorkerCrashedError() from exc
    finally:
        if not completed:
            stop_event.set()
            for future in futures:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=not completed)


def merge_results(
    batch: Iterable[AnodeResult],
    overrides: Iterable[AnodeResult],
    review: Mapping[AnodeKey, str] | None = None,
) -> list[AnodeResult]:
    """Apply per-anode overrides and review states to batch results (sorted by key).

    An override replaces its anode's batch row (``options_source="override"``);
    ``review`` sets the ``review`` column (anodes not in it get ``""``).
    """
    merged = {result.key: result for result in batch}
    for result in overrides:
        if result.options_source != OPTIONS_OVERRIDE:
            result = replace(result, options_source=OPTIONS_OVERRIDE)
        merged[result.key] = result
    states = review or {}
    return [
        (
            merged[key]
            if merged[key].review == states.get(key, "")
            else replace(merged[key], review=states.get(key, ""))
        )
        for key in sorted(merged)
    ]
