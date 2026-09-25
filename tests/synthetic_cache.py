"""Write small adc2kev-layout calibration caches for tests.

The caches pass ``CalibrationCache.is_cache_structurally_valid``: ``/metadata``
carries every required attribute and adc2kev's ``CACHE_VERSION``,
``/histograms`` exists (empty), the events are written with adc2kev's own
``append_event_datasets`` and the calibrations with
``CalibrationCache.save_board_calibrations``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from adc2kev.cache.hdf5_cache import CACHE_VERSION, CalibrationCache, append_event_datasets
from adc2kev.calibration.calibrator import CalibrationResult
from adc2kev.database import ChannelKey

Key = tuple[int, int, int, int]


@dataclass
class Hits:
    """One polarity's hits of a board, in file order."""

    rena: list[int] = field(default_factory=list)
    channel: list[int] = field(default_factory=list)
    pha: list[int] = field(default_factory=list)
    cts: list[int] = field(default_factory=list)
    source: list[int] = field(default_factory=list)

    def add(self, rena: int, channel: int, pha: int, cts: int, source: int = 0) -> Hits:
        self.rena.append(rena)
        self.channel.append(channel)
        self.pha.append(pha)
        self.cts.append(cts)
        self.source.append(source)
        return self

    def extend(
        self,
        rena: npt.ArrayLike,
        channel: npt.ArrayLike,
        pha: npt.ArrayLike,
        cts: npt.ArrayLike,
        source: npt.ArrayLike,
    ) -> Hits:
        n = len(np.atleast_1d(np.asarray(cts)))
        for name, values in (
            ("rena", rena),
            ("channel", channel),
            ("pha", pha),
            ("cts", cts),
            ("source", source),
        ):
            getattr(self, name).extend(np.broadcast_to(np.asarray(values), (n,)).tolist())
        return self

    def __len__(self) -> int:
        return len(self.cts)

    def columns(self) -> tuple[npt.NDArray[np.int64], ...]:
        """adc2kev column order: triggers, renas, channels, phas, file_ids, timestamps."""
        n = len(self)
        return (
            np.arange(n, dtype=np.int64),
            np.asarray(self.rena, dtype=np.int64),
            np.asarray(self.channel, dtype=np.int64),
            np.asarray(self.pha, dtype=np.int64),
            np.asarray(self.source, dtype=np.int64),
            np.asarray(self.cts, dtype=np.int64),
        )


@dataclass
class BoardData:
    """The anode and cathode hits of one board."""

    anode: Hits = field(default_factory=Hits)
    cathode: Hits = field(default_factory=Hits)


def write_cache(
    path: Path,
    boards: Mapping[tuple[int, int], BoardData],
    calibrations: Mapping[Key, tuple[float, float]] | None = None,
    failed: Iterable[Key] = (),
    created_at: str = "2026-09-11T14:17:17.962966",
) -> Path:
    """Write a structurally valid adc2kev calibration cache.

    Args:
        path: Destination ``*.cache.h5``.
        boards: Hits per (node, board).
        calibrations: ``(slope, intercept)`` per channel (stored as successful).
        failed: Channels stored with ``success=False`` (and a real slope).
        created_at: ``/metadata`` ``created_at`` attribute.

    Returns:
        ``path``.
    """
    with h5py.File(path, "w") as h5f:
        meta = h5f.create_group("metadata")
        meta.attrs.update(
            {
                "ge68_path": "/data/ge/data_20260911_124617.dat",
                "ge68_hash": "a" * 64,
                "ge68_size": 1000,
                "ge68_mtime": 1.0,
                "cs137_path": "/data/cs/data_20260911_125053.dat",
                "cs137_hash": "b" * 64,
                "cs137_size": 2000,
                "cs137_mtime": 2.0,
                "cache_version": CACHE_VERSION,
                "created_at": created_at,
                "histogram_bins": 500,
                "adc_range_min": 0,
                "adc_range_max": 4095,
                "total_events": sum(len(b.anode) + len(b.cathode) for b in boards.values()),
            }
        )
        h5f.create_group("histograms")
        h5f.create_group("coincidences")
        for (node, board), data in boards.items():
            group_path = f"/coincidences/node_{node}/board_{board}"
            h5f.require_group(group_path)
            for event_type, hits in (("anode", data.anode), ("cathode", data.cathode)):
                if len(hits):
                    append_event_datasets(h5f, group_path, event_type, hits.columns())

    results: dict[ChannelKey, CalibrationResult] = {}
    for key, (slope, intercept) in (calibrations or {}).items():
        results[ChannelKey(*key)] = _result(key, slope, intercept, success=True)
    for key in failed:
        results[ChannelKey(*key)] = _result(key, 1.0, 0.0, success=False)
    cache = CalibrationCache(path)
    for node, board in sorted({(k.node, k.board) for k in results}):
        cache.save_board_calibrations(node, board, results)
    return path


def _result(key: Key, slope: float, intercept: float, *, success: bool) -> CalibrationResult:
    return CalibrationResult(
        channel=ChannelKey(*key),
        slope=slope,
        intercept=intercept,
        peak_511_adc=0.0,
        peak_662_adc=0.0,
        resolution_511=0.0,
        resolution_662=0.0,
        r_squared=1.0,
        success=success,
        message="synthetic" if success else "synthetic failure",
    )


def write_kev(path: Path, calibrations: Mapping[Key, tuple[float, float]]) -> Path:
    """Write a ``.kev`` file (adc2kev's text layout) with the given calibrations."""
    lines = ["# ADC2KEV Calibration Results", "# node board rena channel slope intercept"]
    for key in sorted(calibrations):
        slope, intercept = calibrations[key]
        lines.append(" ".join(str(v) for v in key) + f" {slope:.6f} {intercept:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def add_pairs(
    board: BoardData,
    anode: tuple[int, int],
    cathode: tuple[int, int],
    pairs: Iterable[tuple[int, int]],
    *,
    source: int = 0,
    t0: int = 1000,
    spacing: int = 1000,
    dcts: int = 5,
) -> BoardData:
    """Append 1A1C events (``(anode PHA, cathode PHA)`` pairs) well separated in CTS."""
    t = t0
    for a_pha, c_pha in pairs:
        board.anode.add(anode[0], anode[1], int(a_pha), t, source)
        board.cathode.add(cathode[0], cathode[1], int(c_pha), t + dcts, source)
        t += spacing
    return board


def tie_pairs(scale: float) -> list[tuple[int, int]]:
    """Legacy test pattern (keV x ``scale``): three peak columns and a tie broken by the last pair.

    With an identity calibration the legacy fit sees 4 peaks; with slope
    ``1/scale`` != 1 the EOF quirk drops the last pair and it sees 3.
    """
    events: list[tuple[float, float]] = []
    for ca, energy, n in [(0.1, 505.0, 30), (0.5, 499.0, 30), (0.9, 487.0, 30)]:
        events += [(energy, ca * energy)] * n
    events += [(493.0, 0.3 * 493.0)] * 7 + [(511.0, 0.3 * 511.0)] * 8
    return [(round(a * scale), round(c * scale)) for a, c in events]
