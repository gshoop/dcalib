"""Data/session layer of the GUI (plan section 9).

The widgets never touch the cache, the sidecar or the analysis modules
directly: they go through :class:`DepthSession` (the pattern of uvcorr's and
specview's ``session.py``). The session owns

- the opened cache (:class:`OpenedCache`): its path, identity, boards, the
  calibrations (the cache's or a ``.kev`` file's) and the sidecar results file;
- the stored results (``/results/current``: batch rows, slices, overrides)
  and the review, merged into one :class:`~dcalib.analysis.AnodeResult` per
  anode with their slice points, and whether they are stale;
- the depth-fit options of the control band and the current selection;
- a small LRU of loaded boards: their 1A1C events and energies
  (:data:`DEFAULT_BOARD_CACHE_SIZE` boards, ~20-120 MB each).

Everything here is Qt-free apart from :class:`ChannelView` (a map record whose
module uses ``QColor``, a value type that needs no ``QApplication``), so the
session is unit tested without widgets.

Threads. Opening a cache (:func:`open_cache`), Fit All
(:meth:`DepthSession.run_batch`) and loading an anode's data
(:meth:`DepthSession.anode_data`, which may read and cluster a board) block
and run in the worker threads of :mod:`dcalib.gui.threads`; everything that
changes the session's state (:meth:`DepthSession.install`,
:meth:`DepthSession.apply_batch`) runs on the GUI thread. Only one Fit All
runs at a time (:class:`SessionBusyError`). The board LRU and every sidecar
access hold one lock, since in-process HDF5 read and write handles on one
file conflict. The cache itself is only ever read.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from adc2kev.cache import CalibrationCache

from dcalib.analysis import (
    AnodeResult,
    BoardAnalysis,
    SliceRow,
    analyze_all,
    default_workers,
    merge_results,
    slices_from_rows,
)
from dcalib.calib import (
    CalibrationError,
    Calibrations,
    EventEnergies,
    event_energies,
    load_calibrations,
)
from dcalib.channels import AnodeKey, anode_channels, cathode_channels, electrode_label
from dcalib.depth import Curve, Slices, fit_curve
from dcalib.events import BoardEvents, build_board_events, list_boards
from dcalib.gui._system_map_model import ChannelView
from dcalib.gui.map_colors import CATHODE_CALIBRATED, CATHODE_UNCALIBRATED
from dcalib.options import DCC_ENERGY_KEV, DepthOptions
from dcalib.results import (
    CacheIdentity,
    ResultsError,
    ResultsFile,
    StoredResults,
    Validity,
    cache_identity,
    default_results_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BOARD_CACHE_SIZE",
    "AnodeData",
    "BatchOutcome",
    "DepthSession",
    "OpenedCache",
    "SessionBusyError",
    "SessionError",
    "anode_title",
    "anode_view",
    "open_cache",
]

DEFAULT_BOARD_CACHE_SIZE = 4

ProgressCallback = Callable[[int, int], None]
StopFlag = Callable[[], bool] | threading.Event

# Metrics of an AnodeResult that the map's views carry (tooltips, colour modes).
_VIEW_METRICS = (
    "n_events",
    "n_selected",
    "degree",
    "cv_gain",
    "peak_spread",
    "fwhm_511_before",
    "fwhm_511_after",
    "fwhm_662_after",
    "ge_cs_max_diff",
)


class SessionError(Exception):
    """An operation of the session failed; the message is meant for the user."""


class SessionBusyError(SessionError):
    """Another operation (a Fit All) is running."""


# ---------------------------------------------------------------------------
# Opening a cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenedCache:
    """Everything read when a cache is opened (built in a worker thread).

    Attributes:
        cache_path: The adc2kev calibration cache.
        results_path: The sidecar results file (it may not exist yet).
        kev_path: The ``.kev`` file replacing the cache's calibrations, if any.
        calibrations: The calibrations in use.
        identity: The cache identity (for the staleness check).
        board_hits: Hits per board with events, sorted by (node, board).
        stored: The stored results, or None.
        validity: Whether ``stored`` belongs to this cache and calibration.
    """

    cache_path: Path
    results_path: Path
    kev_path: Path | None
    calibrations: Calibrations
    identity: CacheIdentity
    board_hits: dict[tuple[int, int], int]
    stored: StoredResults | None
    validity: Validity | None


def open_cache(
    cache_path: str | Path,
    results_path: str | Path | None = None,
    kev_path: str | Path | None = None,
) -> OpenedCache:
    """Read what the GUI needs from a cache and its sidecar (blocking, ~2 s).

    Raises:
        SessionError: If the cache or the ``.kev`` file cannot be used.
    """
    path = Path(cache_path)
    if not path.is_file():
        raise SessionError(f"Cache file not found: {path}")
    if not h5py.is_hdf5(path):
        raise SessionError(f"{path} is not an HDF5 file")
    valid, message = CalibrationCache(path).is_cache_structurally_valid()
    if not valid:
        raise SessionError(f"{path} is not a usable adc2kev calibration cache: {message}")
    kev = Path(kev_path) if kev_path is not None else None
    try:
        calibrations = load_calibrations(path, kev)
    except CalibrationError as exc:
        raise SessionError(str(exc)) from exc
    identity = cache_identity(path)
    sidecar = ResultsFile(results_path if results_path is not None else default_results_path(path))
    try:
        stored = sidecar.load()
    except ResultsError as exc:
        raise SessionError(str(exc)) from exc
    validity = stored.validity(identity, calibrations.fingerprint) if stored is not None else None
    return OpenedCache(
        cache_path=path,
        results_path=sidecar.path,
        kev_path=kev,
        calibrations=calibrations,
        identity=identity,
        board_hits=list_boards(path),
        stored=stored,
        validity=validity,
    )


# ---------------------------------------------------------------------------
# Views and titles
# ---------------------------------------------------------------------------


def anode_view(result: AnodeResult) -> ChannelView:
    """The map record of one anode's (merged) result."""
    return ChannelView(
        status=result.status,
        flags=result.flags,
        options_source=result.options_source,
        review=result.review,
        metrics={name: getattr(result, name) for name in _VIEW_METRICS},
    )


def anode_title(key: AnodeKey | tuple[int, int, int, int]) -> str:
    """``"n1 b17 A09 (r0 ch9)"``: the anode's address and electrode label."""
    node, board, rena, channel = (int(v) for v in key)
    try:
        label = electrode_label(board, rena, channel)
    except KeyError:
        label = "?"
    return f"n{node} b{board} {label} (r{rena} ch{channel})"


# ---------------------------------------------------------------------------
# Anode data (the Depth tab)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnodeData:
    """One anode's events and fit, as the Depth tab draws them.

    Attributes:
        key: The anode.
        x: ``A_keV / E0`` of the events with a calibrated anode and cathode.
        r: ``C_keV / A_keV`` of the same events.
        source: Their source (0 Ge-68, 1 Cs-137).
        anode_kev: Their anode energy in keV.
        cathode: Their cathode as ``rena * 36 + channel``.
        n_events: 1A1C events of the anode (both sources, any cathode).
        n_uncal_cathode: Events dropped for an uncalibrated cathode.
        result: The anode's merged result, if any.
        slices: Its stored slice points per curve name.
        curve: The pooled curve (from the ``.dcc`` coefficients), if any.
        source_curves: The Ge and Cs curves at the pooled degree, fitted from
            the stored per-source slices (no events are refitted).
        options: The options the result was fitted with (batch or override).
    """

    key: AnodeKey
    x: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    source: npt.NDArray[np.int8]
    anode_kev: npt.NDArray[np.float64]
    cathode: npt.NDArray[np.int64]
    n_events: int
    n_uncal_cathode: int
    result: AnodeResult | None
    slices: dict[str, Slices] = field(default_factory=dict)
    curve: Curve | None = None
    source_curves: dict[str, Curve] = field(default_factory=dict)
    options: DepthOptions | None = None

    @property
    def corrected(self) -> bool:
        """Whether the anode is written to the ``.dcc`` (ok and not rejected)."""
        return self.result is not None and self.result.exported

    def g(self, r: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The pooled curve at ``r`` (NaN without one)."""
        rr = np.asarray(r, dtype=np.float64)
        return self.curve(rr) if self.curve is not None else np.full(rr.shape, np.nan)

    def corrected_x(self) -> npt.NDArray[np.float64]:
        """``x / g(r)`` with the anode's curve (``x`` itself without one)."""
        if self.curve is None:
            return self.x
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(self.x / self.curve(self.r), dtype=np.float64)

    def cathodes(self) -> list[tuple[int, int]]:
        """The ``(rena, channel)`` of the cathodes seen by the anode, sorted."""
        return [(int(c) // 36, int(c) % 36) for c in np.unique(self.cathode)]


def _curve_from_result(result: AnodeResult) -> Curve | None:
    coefficients = result.coefficients
    if coefficients is None or result.degree is None:
        return None
    c = tuple(p / DCC_ENERGY_KEV for p in coefficients)
    errors = [
        (e / DCC_ENERGY_KEV) ** 2 if e is not None else 0.0
        for e in (result.err_p0, result.err_p1, result.err_p2)
    ]
    return Curve(
        result.degree,
        (float(c[0]), float(c[1]), float(c[2])),
        np.diag(errors),
        0.0,
        0,
    )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchOutcome:
    """A finished Fit All: its results, stored in the sidecar."""

    options: DepthOptions
    analysis: BoardAnalysis
    stored: StoredResults | None
    n_dropped_overrides: int

    def describe(self) -> str:
        n_ok = sum(1 for r in self.analysis.results if r.ok)
        text = f"Fit All: {len(self.analysis.results):,} anodes, {n_ok:,} corrected"
        if self.n_dropped_overrides:
            text += f"; {self.n_dropped_overrides} override(s) dropped"
        return text


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class _Board:
    events: BoardEvents
    energies: EventEnergies


class DepthSession:
    """The GUI's state and its access to the cache and the sidecar."""

    def __init__(self, board_cache_size: int = DEFAULT_BOARD_CACHE_SIZE) -> None:
        self._opened: OpenedCache | None = None
        self._stored: StoredResults | None = None
        self._validity: Validity | None = None
        self._merged: dict[AnodeKey, AnodeResult] = {}
        self._slices: dict[AnodeKey, list[SliceRow]] = {}
        self._options = DepthOptions()
        self._selection: AnodeKey | None = None
        self._boards: OrderedDict[tuple[int, int], _Board] = OrderedDict()
        self._board_cache_size = max(1, int(board_cache_size))
        self._io_lock = threading.RLock()
        self._busy: str | None = None

    # -- state -------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._opened is not None

    @property
    def opened(self) -> OpenedCache | None:
        return self._opened

    @property
    def cache_path(self) -> Path | None:
        return self._opened.cache_path if self._opened is not None else None

    @property
    def results_file(self) -> ResultsFile | None:
        return ResultsFile(self._opened.results_path) if self._opened is not None else None

    @property
    def stored(self) -> StoredResults | None:
        return self._stored

    @property
    def has_results(self) -> bool:
        return self._stored is not None

    @property
    def validity(self) -> Validity | None:
        return self._validity

    @property
    def stale(self) -> bool:
        """Whether the stored results belong to another cache or calibration."""
        return self._validity is not None and self._validity.stale

    @property
    def batch_options(self) -> DepthOptions | None:
        return self._stored.options if self._stored is not None else None

    @property
    def options(self) -> DepthOptions:
        return self._options

    @options.setter
    def options(self, options: DepthOptions) -> None:
        self._options = options

    @property
    def selection(self) -> AnodeKey | None:
        return self._selection

    def select(self, key: AnodeKey | tuple[int, int, int, int] | None) -> AnodeKey | None:
        self._selection = None if key is None else AnodeKey(*(int(v) for v in key))
        return self._selection

    @property
    def busy(self) -> bool:
        return self._busy is not None

    @contextmanager
    def _exclusive(self, what: str) -> Iterator[None]:
        with self._io_lock:
            if self._busy is not None:
                raise SessionBusyError(f"{self._busy} is running; wait for it to finish")
            self._busy = what
        try:
            yield
        finally:
            with self._io_lock:
                self._busy = None

    def install(self, opened: OpenedCache) -> None:
        """Make ``opened`` the session's cache (GUI thread)."""
        self._opened = opened
        self._boards.clear()
        self._selection = None
        self._set_stored(opened.stored, opened.validity)
        if opened.stored is not None and not self.stale:
            self._options = opened.stored.options

    def close(self) -> None:
        self._opened = None
        self._boards.clear()
        self._selection = None
        self._set_stored(None, None)

    def _set_stored(self, stored: StoredResults | None, validity: Validity | None) -> None:
        self._stored = stored
        self._validity = validity
        if stored is None:
            self._merged, self._slices = {}, {}
            return
        if validity is not None and validity.stale:
            # Stale: show the batch rows as they are (the overrides do not apply).
            review = stored.review_states()
            self._merged = {r.key: r for r in _with_review(list(stored.results), review)}
            by_key: dict[AnodeKey, list[SliceRow]] = {}
            for row in stored.slices:
                by_key.setdefault(row.key, []).append(row)
            self._slices = by_key
            return
        self._merged = {r.key: r for r in stored.merged()}
        self._slices = stored.slices_by_anode()

    # -- navigation ----------------------------------------------------------

    def boards(self) -> list[tuple[int, int]]:
        """Boards with events, sorted."""
        return list(self._opened.board_hits) if self._opened is not None else []

    def nodes(self) -> list[int]:
        return sorted({node for node, _ in self.boards()})

    def boards_on_node(self, node: int) -> list[int]:
        return [board for n, board in self.boards() if n == node]

    def anodes_on_board(self, node: int, board: int) -> list[AnodeKey]:
        """The board's anodes with a result, else all 39 (before any Fit All)."""
        with_results = sorted(k for k in self._merged if (k.node, k.board) == (node, board))
        if with_results:
            return with_results
        return [AnodeKey(node, board, rena, ch) for rena, ch in sorted(anode_channels(board))]

    def all_anodes(self) -> list[AnodeKey]:
        return [key for node, board in self.boards() for key in self.anodes_on_board(node, board)]

    def step_anode(self, key: AnodeKey | None, delta: int) -> AnodeKey | None:
        """The anode ``delta`` places from ``key`` in (node, board, rena, channel) order."""
        keys = self.all_anodes()
        if not keys:
            return None
        if key is None or key not in keys:
            return keys[0] if delta >= 0 else keys[-1]
        index = keys.index(key) + delta
        return keys[max(0, min(index, len(keys) - 1))]

    # -- results -------------------------------------------------------------

    def result(self, key: AnodeKey | tuple[int, int, int, int]) -> AnodeResult | None:
        return self._merged.get(AnodeKey(*(int(v) for v in key)))

    def merged_results(self) -> list[AnodeResult]:
        return [self._merged[k] for k in sorted(self._merged)]

    def slices_for(self, key: AnodeKey) -> dict[str, Slices]:
        return slices_from_rows(self._slices.get(key, []))

    def options_for(self, key: AnodeKey) -> DepthOptions | None:
        """The options the anode's result was fitted with (its override's, else the batch's)."""
        if self._stored is None:
            return None
        override = self._stored.overrides.get(key)
        if override is not None and not self.stale:
            return override.options
        return self._stored.options

    def views(self) -> dict[AnodeKey, ChannelView]:
        """The map records: every anode with a result and every cathode of a board with events."""
        views: dict[AnodeKey, ChannelView] = {k: anode_view(r) for k, r in self._merged.items()}
        if self._opened is not None:
            calibrations = self._opened.calibrations
            for node, board in self.boards():
                for rena, channel in cathode_channels(board):
                    key = AnodeKey(node, board, rena, channel)
                    status = (
                        CATHODE_CALIBRATED
                        if calibrations.is_calibrated(key)
                        else CATHODE_UNCALIBRATED
                    )
                    views[key] = ChannelView(status=status)
        return views

    # -- boards and anode data -------------------------------------------------

    def cached_boards(self) -> list[tuple[int, int]]:
        with self._io_lock:
            return list(self._boards)

    def _board(self, node: int, board: int) -> _Board:
        opened = self._require_open()
        with self._io_lock:
            cached = self._boards.get((node, board))
            if cached is not None:
                self._boards.move_to_end((node, board))
                return cached
        events = build_board_events(opened.cache_path, node, board, self._options.cts_window)
        loaded = _Board(events, event_energies(events, opened.calibrations.board_lut(node, board)))
        with self._io_lock:
            self._boards[(node, board)] = loaded
            while len(self._boards) > self._board_cache_size:
                self._boards.popitem(last=False)
        return loaded

    def board_events(self, node: int, board: int) -> BoardEvents:
        """The board's 1A1C events (loaded and clustered on first use, then cached)."""
        return self._board(node, board).events

    def anode_data(self, key: AnodeKey | tuple[int, int, int, int]) -> AnodeData:
        """Everything the Depth tab draws for one anode (blocking: may load a board)."""
        key = AnodeKey(*(int(v) for v in key))
        loaded = self._board(key.node, key.board)
        events, energies = loaded.events, loaded.energies
        codes = events.anode_codes()
        rows = np.flatnonzero(codes == key.rena * 36 + key.channel)
        usable = rows[energies.anode_calibrated[rows] & energies.cathode_calibrated[rows]]
        result = self._merged.get(key)
        slices = self.slices_for(key)
        curve = _curve_from_result(result) if result is not None else None
        source_curves: dict[str, Curve] = {}
        if curve is not None:
            for name in ("ge", "cs"):
                sl = slices.get(name)
                if sl is not None and len(sl) >= max(curve.degree + 1, 3):
                    source_curves[name] = fit_curve(sl, curve.degree)
        return AnodeData(
            key=key,
            x=energies.x[usable],
            r=energies.r[usable],
            source=events.source[usable],
            anode_kev=energies.anode_kev[usable],
            cathode=(
                events.cathode_rena[usable].astype(np.int64) * 36 + events.cathode_channel[usable]
            ),
            n_events=len(rows),
            n_uncal_cathode=int(len(rows) - np.count_nonzero(energies.cathode_calibrated[rows])),
            result=result,
            slices=slices,
            curve=curve,
            source_curves=source_curves,
            options=self.options_for(key),
        )

    def _require_open(self) -> OpenedCache:
        if self._opened is None:
            raise SessionError("No cache is open")
        return self._opened

    # -- Fit All ---------------------------------------------------------------

    def run_batch(
        self,
        options: DepthOptions,
        workers: int | None = None,
        progress: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
    ) -> BatchOutcome:
        """Analyse every board and store the batch in the sidecar (worker thread).

        Raises:
            SessionBusyError: If another Fit All runs.
            AnalysisCancelled: If ``stop_flag`` requested a stop (nothing stored).
            SessionError: If the results cannot be stored.
        """
        opened = self._require_open()
        with self._exclusive("Fit All"):
            analysis = analyze_all(
                opened.cache_path,
                opened.calibrations,
                options,
                workers=workers if workers is not None else default_workers(),
                progress_cb=progress,
                stop_flag=stop_flag,
            )
            sidecar = ResultsFile(opened.results_path)
            with self._io_lock:
                try:
                    n_dropped = sidecar.save_batch(
                        analysis.results,
                        analysis.slices,
                        options,
                        opened.identity,
                        opened.calibrations,
                    )
                    stored = sidecar.load()
                except (ResultsError, OSError) as exc:
                    raise SessionError(f"The results could not be stored: {exc}") from exc
        return BatchOutcome(options, analysis, stored, n_dropped)

    def apply_batch(self, outcome: BatchOutcome) -> dict[AnodeKey, ChannelView]:
        """Install a finished Fit All (GUI thread); returns the new map views."""
        if self._opened is None:
            return {}
        validity = (
            outcome.stored.validity(self._opened.identity, self._opened.calibrations.fingerprint)
            if outcome.stored is not None
            else None
        )
        self._set_stored(outcome.stored, validity)
        self._options = outcome.options
        return self.views()

    # -- text ------------------------------------------------------------------

    def describe(self) -> str:
        """One line about the open cache and its results (status bar)."""
        if self._opened is None:
            return "No cache open"
        opened = self._opened
        n_boards = len(opened.board_hits)
        text = f"{opened.cache_path.name}: {n_boards} boards"
        if self._stored is None:
            return text + ", no results (run Process > Fit All)"
        n_ok = sum(1 for r in self._merged.values() if r.exported)
        text += f", {len(self._merged):,} anodes, {n_ok:,} corrected"
        if self.stale:
            text += " (stale results)"
        return text


def _with_review(results: list[AnodeResult], review: dict[AnodeKey, str]) -> list[AnodeResult]:
    return merge_results(results, [], review)
