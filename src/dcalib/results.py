"""The sidecar results file ``<cache stem>.depth.h5`` (plan 6.2, decision D12).

Layout::

    /metadata              attrs: dcalib_version, depth_results_version, created_at, the
                           cache identity and calibration provenance of the stored results
    /results/current       attrs: options_json, created_at, dcalib_version, cache identity,
                           calibration source/sha256/fingerprint
        table              one row per anode (the CSV columns, analysis.RESULT_COLUMNS)
        slices             the slice points of every curve (analysis.SLICE_COLUMNS)
        overrides/table    per-anode re-fits: the same columns plus options_json
        overrides/slices   their slice points
    /review/table          node, board, rena, channel, state, timestamp, note

Writes follow uvcorr's cache: a batch is written completely into
``/results/_new`` and then swapped in (``current`` -> ``_old``, ``_new`` ->
``current``, the commit point); an interrupted swap is recovered on the next
write and readers fall back to ``_old``. The overrides and the review tables
are replaced through ``table_new`` the same way. No handle stays open between
calls; an HDF5 file lock held by another process is retried for
``LOCK_RETRY_SECONDS`` and then reported as :class:`ResultsBusyError`.

Validity: the stored results are *stale* when the cache identity (size,
adc2kev ``created_at`` and source hashes; the path and mtime are recorded but
not compared, so a moved or copied cache stays valid) or the calibration
fingerprint no longer matches. ``dcalib process`` then replaces them and drops
the overrides; the review decisions are human choices keyed by channel and are
kept (the census lists them).

Unavailable values: NaN in the float columns and -1 in the count columns
(``degree``, ``n_slices``).
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import math
import os
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from adc2kev.cache import CalibrationCache

from dcalib import __version__
from dcalib.analysis import (
    KIND_COUNT,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_INT,
    KIND_STR,
    OPTIONS_BATCH,
    OPTIONS_OVERRIDE,
    RESULT_COLUMNS,
    SLICE_COLUMNS,
    AnodeResult,
    SliceRow,
    merge_results,
)
from dcalib.calib import Calibrations
from dcalib.channels import AnodeKey
from dcalib.io import cache_stem
from dcalib.options import (
    FLAG_SEPARATOR,
    FLAGS,
    REVIEW_REJECTED,
    STATUSES,
    DepthOptions,
    same_fit,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEPTH_RESULTS_VERSION",
    "LOCK_RETRY_SECONDS",
    "CacheIdentity",
    "ResultsBusyError",
    "ResultsError",
    "ResultsFile",
    "ReviewEntry",
    "StaleResultsError",
    "StoredOverride",
    "StoredResults",
    "Validity",
    "cache_identity",
    "default_results_path",
]

DEPTH_RESULTS_VERSION = "1.0.0"
LOCK_RETRY_SECONDS = 1.0
LOCK_RETRY_INTERVAL = 0.05
RESULTS_SUFFIX = ".depth.h5"

_RESULTS = "results"
_CURRENT = "current"
_NEW = "_new"
_OLD = "_old"
_TABLE = "table"
_TABLE_NEW = "table_new"
_SLICES = "slices"
_SLICES_NEW = "slices_new"
_OVERRIDES = "overrides"
_REVIEW = "review"
_METADATA = "metadata"
_OPTIONS_COLUMN = "options_json"
_MISSING_COUNT = -1


class ResultsError(Exception):
    """The results file cannot be read or written as requested."""


class ResultsBusyError(ResultsError):
    """Another process holds the results file's HDF5 lock."""


class StaleResultsError(ResultsError):
    """The stored results changed since the caller loaded them."""


def default_results_path(cache_path: str | Path) -> Path:
    """``<cache stem>.depth.h5`` next to the cache."""
    path = Path(cache_path)
    return path.with_name(cache_stem(path) + RESULTS_SUFFIX)


# ---------------------------------------------------------------------------
# Identity and validity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheIdentity:
    """What identifies the adc2kev cache the results were computed from."""

    path: str
    size: int
    mtime: float
    created_at: str
    ge68_hash: str
    cs137_hash: str

    def same_cache(self, other: CacheIdentity) -> bool:
        """Compare size, ``created_at`` and the source hashes (not the path or mtime)."""
        return (self.size, self.created_at, self.ge68_hash, self.cs137_hash) == (
            other.size,
            other.created_at,
            other.ge68_hash,
            other.cs137_hash,
        )

    def to_attrs(self) -> dict[str, Any]:
        return {f"cache_{f.name}": getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> CacheIdentity | None:
        try:
            return cls(
                path=_text(attrs["cache_path"]),
                size=int(attrs["cache_size"]),
                mtime=float(attrs["cache_mtime"]),
                created_at=_text(attrs["cache_created_at"]),
                ge68_hash=_text(attrs["cache_ge68_hash"]),
                cs137_hash=_text(attrs["cache_cs137_hash"]),
            )
        except KeyError:
            return None


def cache_identity(cache_path: str | Path) -> CacheIdentity:
    """Read the identity of an adc2kev cache (read-only)."""
    path = Path(cache_path)
    stat = path.stat()
    meta = CalibrationCache(path).get_metadata()
    return CacheIdentity(
        path=str(path.resolve()),
        size=int(stat.st_size),
        mtime=float(stat.st_mtime),
        created_at=meta.created_at if meta is not None else "",
        ge68_hash=meta.ge68_hash if meta is not None else "",
        cs137_hash=meta.cs137_hash if meta is not None else "",
    )


@dataclass(frozen=True)
class Validity:
    """Whether stored results belong to the current cache and calibration."""

    valid: bool
    reasons: tuple[str, ...] = ()

    @property
    def stale(self) -> bool:
        return not self.valid


# ---------------------------------------------------------------------------
# Stored content
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredOverride:
    """One per-anode re-fit: its row, slices and options."""

    result: AnodeResult
    slices: tuple[SliceRow, ...]
    options: DepthOptions
    unknown_options: tuple[str, ...] = ()

    def reproduced_by(self, batch_options: DepthOptions) -> bool:
        """True when a batch with ``batch_options`` fits this anode identically."""
        return not self.unknown_options and same_fit(self.options, batch_options)


@dataclass(frozen=True)
class ReviewEntry:
    """A per-anode review decision (plan D10)."""

    state: str
    timestamp: str
    note: str = ""


@dataclass(frozen=True)
class StoredResults:
    """Everything in ``/results/current`` plus the review table."""

    results: tuple[AnodeResult, ...]
    slices: tuple[SliceRow, ...]
    options: DepthOptions
    created_at: str
    dcalib_version: str
    identity: CacheIdentity | None
    calibration_source: str
    calibration_fingerprint: str
    overrides: Mapping[AnodeKey, StoredOverride] = field(default_factory=dict)
    review: Mapping[AnodeKey, ReviewEntry] = field(default_factory=dict)

    def validity(self, identity: CacheIdentity, fingerprint: str) -> Validity:
        reasons = []
        if self.identity is None or not self.identity.same_cache(identity):
            reasons.append("the cache is not the one the results were computed from")
        if self.calibration_fingerprint != fingerprint:
            reasons.append("the keV calibration changed")
        return Validity(not reasons, tuple(reasons))

    def review_states(self) -> dict[AnodeKey, str]:
        return {key: entry.state for key, entry in self.review.items()}

    def merged(self) -> list[AnodeResult]:
        """Batch rows with the overrides and the review applied (what the exports write)."""
        return merge_results(
            self.results, (o.result for o in self.overrides.values()), self.review_states()
        )

    def slices_by_anode(self) -> dict[AnodeKey, list[SliceRow]]:
        """Slice rows per anode: the batch slices, replaced by an override's own."""
        out: dict[AnodeKey, list[SliceRow]] = {}
        for row in self.slices:
            out.setdefault(row.key, []).append(row)
        for key, override in self.overrides.items():
            out[key] = list(override.slices)
        return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _object_array(values: list[Any]) -> npt.NDArray[np.object_]:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _results_dtype(with_options: bool) -> np.dtype[Any]:
    text = h5py.string_dtype()
    cols: list[tuple[str, Any]] = []
    for column in RESULT_COLUMNS:
        if column.kind in (KIND_INT, KIND_COUNT):
            cols.append((column.name, np.int64))
        elif column.kind == KIND_FLOAT:
            cols.append((column.name, np.float64))
        else:
            cols.append((column.name, text))
    if with_options:
        cols.append((_OPTIONS_COLUMN, text))
    return np.dtype(cols)


def _encode_results(
    rows: list[AnodeResult], options: list[DepthOptions] | None = None
) -> npt.NDArray[np.void]:
    table = np.zeros(len(rows), dtype=_results_dtype(options is not None))
    for column in RESULT_COLUMNS:
        values = [getattr(row, column.name) for row in rows]
        if column.kind == KIND_COUNT:
            table[column.name] = [_MISSING_COUNT if v is None else v for v in values]
        elif column.kind == KIND_FLOAT:
            table[column.name] = [math.nan if v is None else v for v in values]
        elif column.kind == KIND_FLAGS:
            table[column.name] = _object_array([row.flags_text for row in rows])
        elif column.kind == KIND_STR:
            table[column.name] = _object_array(values)
        else:
            table[column.name] = values
    if options is not None:
        table[_OPTIONS_COLUMN] = _object_array([o.to_json() for o in options])
    return table


def _decode_results(
    table: npt.NDArray[np.void], with_options: bool
) -> list[tuple[int, AnodeResult, DepthOptions | None]]:
    """Decode a results table; rows a newer dcalib wrote with unknown values are skipped."""
    names = set(table.dtype.names or ())
    columns: dict[str, list[Any]] = {}
    for column in RESULT_COLUMNS:
        if column.name not in names:
            continue
        data = table[column.name]
        if column.kind == KIND_COUNT:
            columns[column.name] = [None if v < 0 else v for v in data.astype(np.int64).tolist()]
        elif column.kind == KIND_INT:
            columns[column.name] = data.astype(np.int64).tolist()
        elif column.kind == KIND_FLOAT:
            columns[column.name] = data.astype(np.float64).tolist()
        else:
            columns[column.name] = [_text(v) for v in data]
    known_flags = set(FLAGS)
    decoded: list[tuple[int, AnodeResult, DepthOptions | None]] = []
    skipped = 0
    for i in range(len(table)):
        row = {name: values[i] for name, values in columns.items()}
        if row.get("status") not in STATUSES:
            skipped += 1
            continue
        if "flags" in row:
            row["flags"] = tuple(
                f for f in str(row["flags"]).split(FLAG_SEPARATOR) if f in known_flags
            )
        options = None
        if with_options:
            options = DepthOptions.from_json(_text(table[_OPTIONS_COLUMN][i]), strict=False)
        try:
            decoded.append((i, AnodeResult.from_dict(row), options))
        except (TypeError, ValueError):
            skipped += 1
    if skipped:
        logger.warning("Skipping %d stored result row(s) this dcalib cannot read", skipped)
    return decoded


def _slices_dtype() -> np.dtype[Any]:
    cols: list[tuple[str, Any]] = []
    for name in SLICE_COLUMNS:
        if name == "curve":
            cols.append((name, h5py.string_dtype()))
        elif name in ("node", "board", "rena", "channel", "slice", "n"):
            cols.append((name, np.int64))
        else:
            cols.append((name, np.float64))
    return np.dtype(cols)


def _encode_slices(rows: Iterable[SliceRow]) -> npt.NDArray[np.void]:
    items = list(rows)
    table = np.zeros(len(items), dtype=_slices_dtype())
    for j, name in enumerate(SLICE_COLUMNS):
        values = [row[j] for row in items]
        table[name] = _object_array(values) if name == "curve" else values
    return table


def _decode_slices(table: npt.NDArray[np.void]) -> list[SliceRow]:
    cols = [table[name] for name in SLICE_COLUMNS]
    out = []
    for i in range(len(table)):
        values = [c[i] for c in cols]
        out.append(
            SliceRow(
                int(values[0]),
                int(values[1]),
                int(values[2]),
                int(values[3]),
                _text(values[4]),
                int(values[5]),
                float(values[6]),
                float(values[7]),
                float(values[8]),
                float(values[9]),
                float(values[10]),
                float(values[11]),
                int(values[12]),
            )
        )
    return out


def _review_dtype() -> np.dtype[Any]:
    text = h5py.string_dtype()
    return np.dtype(
        [
            ("node", np.int64),
            ("board", np.int64),
            ("rena", np.int64),
            ("channel", np.int64),
            ("state", text),
            ("timestamp", text),
            ("note", text),
        ]
    )


# ---------------------------------------------------------------------------
# File access
# ---------------------------------------------------------------------------


def _is_lock_error(exc: OSError) -> bool:
    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
        return True
    message = str(exc)
    return "unable to lock file" in message or "temporarily unavailable" in message


@contextlib.contextmanager
def _open(path: Path, mode: str = "r") -> Iterator[h5py.File]:
    deadline = time.monotonic() + LOCK_RETRY_SECONDS
    while True:
        try:
            h5f = h5py.File(path, mode)
            break
        except OSError as exc:
            if not _is_lock_error(exc):
                raise
            if time.monotonic() >= deadline:
                raise ResultsBusyError(
                    f"The results file {path} is in use by another process (HDF5 file lock); "
                    "try again when it has finished"
                ) from exc
            time.sleep(LOCK_RETRY_INTERVAL)
    with h5f:
        yield h5f


def _delete_quietly(group: Any, name: str) -> None:
    try:
        if name in group:
            del group[name]
    except Exception as exc:  # post-commit cleanup: readers ignore leftovers
        logger.warning("Could not remove %s/%s (cleaned up later): %s", group.name, name, exc)


def _current(h5f: Any) -> Any:
    """The complete stored results group: ``current``, else an interrupted swap's ``_old``."""
    group = h5f.get(_RESULTS)
    if group is None:
        return None
    for name in (_CURRENT, _OLD):
        candidate = group.get(name)
        if candidate is not None and _TABLE in candidate:
            return candidate
    return None


def _recover(group: Any) -> None:
    _delete_quietly(group, _NEW)
    if _OLD in group:
        if _CURRENT in group:
            _delete_quietly(group, _OLD)
        else:
            group.move(_OLD, _CURRENT)


def _dataset(group: Any, name: str, new_name: str) -> Any:
    """``group[name]``, or an interrupted swap's ``group[new_name]``, or None."""
    if group is None:
        return None
    for candidate in (name, new_name):
        if candidate in group:
            return group[candidate]
    return None


def _replace_dataset(group: Any, name: str, new_name: str, data: npt.NDArray[Any] | None) -> None:
    """Replace ``group[name]`` through ``new_name`` (``data=None`` deletes it)."""
    if new_name in group:
        if name in group:
            del group[new_name]
        else:
            group.move(new_name, name)
    if data is None:
        if name in group:
            del group[name]
        return
    group.create_dataset(new_name, data=data)
    if name in group:
        del group[name]  # commit point: readers use new_name meanwhile
    try:
        group.move(new_name, name)
    except Exception as exc:
        logger.warning("Could not rename %s (it is used as is): %s", new_name, exc)


def _unknown_option_keys(text: str) -> tuple[str, ...]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, dict):
        return ()
    known = {f.name for f in fields(DepthOptions)}
    return tuple(sorted(str(k) for k in data if k not in known))


class ResultsFile:
    """The sidecar results file of one cache (see the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __repr__(self) -> str:
        return f"ResultsFile({str(self._path)!r})"

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def check_writable(self) -> None:
        """Raise OSError if the file (or, for a new file, its directory) is not writable."""
        target = self._path if self._path.exists() else self._path.parent
        if not target.exists():
            raise OSError(f"the directory of the results file does not exist: {target}")
        if not os.access(target, os.W_OK):
            raise PermissionError(f"cannot write the results file {self._path}")

    # -- batch -------------------------------------------------------------

    def save_batch(
        self,
        results: Iterable[AnodeResult],
        slices: Iterable[SliceRow],
        options: DepthOptions,
        identity: CacheIdentity,
        calibrations: Calibrations,
        *,
        keep_overrides: bool = True,
    ) -> int:
        """Store a batch run as ``/results/current`` (write, then swap).

        Overrides are carried over when ``keep_overrides``, except those that
        the new batch reproduces (:meth:`StoredOverride.reproduced_by`) and all
        of them when the stored results are stale (another cache or
        calibration). The review table is never touched.

        Returns:
            The number of stored overrides not carried over.

        Raises:
            ValueError: If a row is not a batch row or two rows share a key.
            ResultsBusyError: If another process holds the file's lock.
            OSError: If the file cannot be written.
        """
        rows = sorted(results, key=lambda r: r.key)
        for prev, row in zip(rows, rows[1:]):
            if prev.key == row.key:
                raise ValueError(f"Duplicate results for {row.key}")
        for row in rows:
            if row.options_source != OPTIONS_BATCH:
                raise ValueError(f"save_batch takes batch rows; {row.key} is an override")
        rows = [r if not r.review else _without_review(r) for r in rows]
        table = _encode_results(rows)
        slice_table = _encode_slices(slices)
        now = datetime.now().isoformat()
        attrs: dict[str, Any] = {
            "options_json": options.to_json(),
            "created_at": now,
            "dcalib_version": __version__,
            "depth_results_version": DEPTH_RESULTS_VERSION,
            "calibration_source": calibrations.source,
            "calibration_sha256": calibrations.source_sha256 or "",
            "calibration_fingerprint": calibrations.fingerprint,
            "n_anodes": np.int64(len(rows)),
            **identity.to_attrs(),
        }
        with _open(self._path, "a") as h5f:
            group = h5f.require_group(_RESULTS)
            _recover(group)
            current = group.get(_CURRENT)
            n_dropped = 0
            keep_table: npt.NDArray[np.void] | None = None
            keep_slices: npt.NDArray[np.void] | None = None
            if current is not None and _OVERRIDES in current:
                stored_fp = _text(current.attrs.get("calibration_fingerprint", ""))
                stored_id = CacheIdentity.from_attrs(current.attrs)
                stale = stored_fp != calibrations.fingerprint or (
                    stored_id is None or not stored_id.same_cache(identity)
                )
                keep_table, keep_slices, n_dropped = _overrides_to_keep(
                    current[_OVERRIDES], options, keep_overrides and not stale
                )
            new = group.create_group(_NEW)
            try:
                new.attrs.update(attrs)
                new.create_dataset(_TABLE, data=table)
                new.create_dataset(_SLICES, data=slice_table)
                if keep_table is not None and len(keep_table):
                    ov = new.create_group(_OVERRIDES)
                    ov.create_dataset(_TABLE, data=keep_table)
                    ov.create_dataset(
                        _SLICES,
                        data=keep_slices if keep_slices is not None else _encode_slices([]),
                    )
                if current is not None:
                    group.move(_CURRENT, _OLD)
                group.move(_NEW, _CURRENT)  # commit point
            except BaseException:
                with contextlib.suppress(Exception):
                    if _NEW in group:
                        del group[_NEW]
                    if _CURRENT not in group and _OLD in group:
                        group.move(_OLD, _CURRENT)
                raise
            _delete_quietly(group, _OLD)
            meta = h5f.require_group(_METADATA)
            meta.attrs.update({k: v for k, v in attrs.items() if k not in ("options_json",)})
        logger.info("Saved %d anode results to %s", len(rows), self._path)
        return n_dropped

    # -- reading -----------------------------------------------------------

    def created_at(self) -> str | None:
        """``created_at`` of the stored batch, or None (cheap)."""
        if not self.exists():
            return None
        with _open(self._path) as h5f:
            current = _current(h5f)
            return None if current is None else _text(current.attrs.get("created_at", ""))

    def load(self) -> StoredResults | None:
        """Load the stored results, overrides and review (None without batch results).

        Raises:
            ResultsError: If the stored results cannot be decoded.
            ResultsBusyError: If another process holds the file's lock.
        """
        if not self.exists():
            return None
        with _open(self._path) as h5f:
            current = _current(h5f)
            review = _read_review(h5f)
            if current is None:
                return None
            try:
                attrs = current.attrs
                rows = _decode_results(current[_TABLE][()], with_options=False)
                slices = _decode_slices(current[_SLICES][()]) if _SLICES in current else []
                overrides = _read_overrides(current.get(_OVERRIDES))
                return StoredResults(
                    results=tuple(r for _, r, _ in rows),
                    slices=tuple(slices),
                    options=DepthOptions.from_json(_text(attrs["options_json"]), strict=False),
                    created_at=_text(attrs.get("created_at", "")),
                    dcalib_version=_text(attrs.get("dcalib_version", "")),
                    identity=CacheIdentity.from_attrs(attrs),
                    calibration_source=_text(attrs.get("calibration_source", "")),
                    calibration_fingerprint=_text(attrs.get("calibration_fingerprint", "")),
                    overrides=overrides,
                    review=review,
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ResultsError(
                    f"The stored results in {self._path} are unreadable: {exc}"
                ) from exc

    def load_review(self) -> dict[AnodeKey, ReviewEntry]:
        """The review decisions (also without batch results)."""
        if not self.exists():
            return {}
        with _open(self._path) as h5f:
            return _read_review(h5f)

    # -- overrides ---------------------------------------------------------

    def replace_overrides(
        self,
        save: Iterable[tuple[AnodeResult, Iterable[SliceRow], DepthOptions]] = (),
        delete: Iterable[AnodeKey | tuple[int, int, int, int]] = (),
        *,
        expected_created_at: str | None = None,
    ) -> tuple[int, int]:
        """Store some per-anode re-fits and delete others, in one table swap.

        Returns:
            ``(saved, deleted)``.

        Raises:
            ResultsError: If there are no batch results to attach overrides to.
            StaleResultsError: If ``expected_created_at`` does not match.
            ResultsBusyError: If another process holds the file's lock.
        """
        items = list(save)
        wanted = {AnodeKey(*(int(v) for v in key)) for key in delete}
        with _open(self._path, "a") as h5f:
            group = h5f.get(_RESULTS)
            if group is not None:
                _recover(group)
            current = _current(h5f)
            found = None if current is None else _text(current.attrs.get("created_at", ""))
            if expected_created_at is not None and found != expected_created_at:
                raise StaleResultsError(
                    f"The stored results in {self._path.name} changed since they were loaded"
                )
            if current is None:
                if items:
                    raise ResultsError(
                        f"{self._path} stores no batch results to attach an override to; "
                        "run a batch analysis first"
                    )
                return 0, 0
            entries = _read_override_entries(current.get(_OVERRIDES))
            deleted = [key for key in entries if key in wanted]
            for key in deleted:
                del entries[key]
            for result, slices, options in items:
                if result.options_source != OPTIONS_OVERRIDE:
                    result = _as_override(result)
                if result.review:
                    result = _without_review(result)
                entries[result.key] = StoredOverride(result, tuple(slices), options)
            if not items and not deleted:
                return 0, 0
            _write_override_entries(current, entries)
        return len(items), len(deleted)

    def clear_overrides(self) -> int:
        """Delete every override; returns how many."""
        if not self.exists():
            return 0
        with _open(self._path, "a") as h5f:
            group = h5f.get(_RESULTS)
            if group is not None:
                _recover(group)
            current = _current(h5f)
            if current is None or _OVERRIDES not in current:
                return 0
            table = _dataset(current[_OVERRIDES], _TABLE, _TABLE_NEW)
            n = 0 if table is None else int(table.shape[0])
            del current[_OVERRIDES]
            return n

    # -- review ------------------------------------------------------------

    def set_review(
        self, key: AnodeKey | tuple[int, int, int, int], state: str, note: str = ""
    ) -> None:
        """Record a review decision (``state="rejected"``) or clear it (``state=""``)."""
        self.set_reviews({AnodeKey(*(int(v) for v in key)): (state, note)})

    def set_reviews(self, changes: Mapping[AnodeKey, tuple[str, str]]) -> None:
        """Record or clear several review decisions in one table swap."""
        for state, _ in changes.values():
            if state not in ("", REVIEW_REJECTED):
                raise ValueError(f"review state must be '' or {REVIEW_REJECTED!r}, got {state!r}")
        with _open(self._path, "a") as h5f:
            entries = _read_review(h5f)
            now = datetime.now().isoformat(timespec="seconds")
            for key, (state, note) in changes.items():
                if state:
                    entries[AnodeKey(*key)] = ReviewEntry(state, now, note)
                else:
                    entries.pop(AnodeKey(*key), None)
            _write_review(h5f, entries)

    def clear_review(self) -> int:
        """Delete every review decision; returns how many."""
        if not self.exists():
            return 0
        with _open(self._path, "a") as h5f:
            entries = _read_review(h5f)
            _write_review(h5f, {})
            return len(entries)


def _without_review(result: AnodeResult) -> AnodeResult:
    return replace(result, review="")


def _as_override(result: AnodeResult) -> AnodeResult:
    return replace(result, options_source=OPTIONS_OVERRIDE)


def _read_override_entries(group: Any) -> dict[AnodeKey, StoredOverride]:
    return dict(_read_overrides(group))


def _read_overrides(group: Any) -> dict[AnodeKey, StoredOverride]:
    table = _dataset(group, _TABLE, _TABLE_NEW)
    if table is None:
        return {}
    raw = table[()]
    slice_table = _dataset(group, _SLICES, _SLICES_NEW)
    slices = _decode_slices(slice_table[()]) if slice_table is not None else []
    by_key: dict[AnodeKey, list[SliceRow]] = {}
    for row in slices:
        by_key.setdefault(row.key, []).append(row)
    out: dict[AnodeKey, StoredOverride] = {}
    for index, result, options in _decode_results(raw, with_options=True):
        assert options is not None
        unknown = _unknown_option_keys(_text(raw[_OPTIONS_COLUMN][index]))
        out[result.key] = StoredOverride(
            result, tuple(by_key.get(result.key, [])), options, unknown
        )
    return dict(sorted(out.items()))


def _overrides_to_keep(
    group: Any, batch_options: DepthOptions, keep: bool
) -> tuple[npt.NDArray[np.void] | None, npt.NDArray[np.void] | None, int]:
    """The override table and slices to carry over to a new batch, and how many are dropped."""
    table = _dataset(group, _TABLE, _TABLE_NEW)
    if table is None:
        return None, None, 0
    raw = table[()]
    if not keep:
        return None, None, len(raw)
    stored = _read_overrides(group)
    drop = {key for key, o in stored.items() if o.reproduced_by(batch_options)}
    if not drop:
        slice_table = _dataset(group, _SLICES, _SLICES_NEW)
        return raw, None if slice_table is None else slice_table[()], 0
    kept = {key: o for key, o in stored.items() if key not in drop}
    rows = list(kept.values())
    return (
        _encode_results([o.result for o in rows], [o.options for o in rows]),
        _encode_slices([s for o in rows for s in o.slices]),
        len(drop),
    )


def _write_override_entries(current: Any, entries: Mapping[AnodeKey, StoredOverride]) -> None:
    if not entries:
        if _OVERRIDES in current:
            del current[_OVERRIDES]
        return
    keys = sorted(entries)
    table = _encode_results([entries[k].result for k in keys], [entries[k].options for k in keys])
    slices = _encode_slices([s for k in keys for s in entries[k].slices])
    group = current.require_group(_OVERRIDES)
    _replace_dataset(group, _SLICES, _SLICES_NEW, slices)
    _replace_dataset(group, _TABLE, _TABLE_NEW, table)


def _read_review(h5f: Any) -> dict[AnodeKey, ReviewEntry]:
    table = _dataset(h5f.get(_REVIEW), _TABLE, _TABLE_NEW)
    if table is None:
        return {}
    out: dict[AnodeKey, ReviewEntry] = {}
    for row in table[()]:
        key = AnodeKey(int(row["node"]), int(row["board"]), int(row["rena"]), int(row["channel"]))
        out[key] = ReviewEntry(_text(row["state"]), _text(row["timestamp"]), _text(row["note"]))
    return dict(sorted(out.items()))


def _write_review(h5f: Any, entries: Mapping[AnodeKey, ReviewEntry]) -> None:
    group = h5f.require_group(_REVIEW)
    if not entries:
        _replace_dataset(group, _TABLE, _TABLE_NEW, None)
        return
    keys = sorted(entries)
    table = np.zeros(len(keys), dtype=_review_dtype())
    table["node"] = [k.node for k in keys]
    table["board"] = [k.board for k in keys]
    table["rena"] = [k.rena for k in keys]
    table["channel"] = [k.channel for k in keys]
    table["state"] = _object_array([entries[k].state for k in keys])
    table["timestamp"] = _object_array([entries[k].timestamp for k in keys])
    table["note"] = _object_array([entries[k].note for k in keys])
    _replace_dataset(group, _TABLE, _TABLE_NEW, table)
