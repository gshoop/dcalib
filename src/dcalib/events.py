"""One-anode/one-cathode events of a board, built from the adc2kev cache (plan 5.1, D5).

Per board and per source (the cache's ``file_id``: 0 = Ge-68, 1 = Cs-137):

1. :func:`read_board_hits` reads the board's ``anode_*`` and ``cathode_*``
   datasets (RENA, channel, PHA, CTS timestamp and file id) in one pass. The
   file is opened read-only and closed before returning.
2. The anode and cathode hits of one source are concatenated and stable-sorted
   by CTS (the cache stores hits in file order, which is not monotonic in CTS).
3. Greedy clustering: the first hit anchors a cluster, and a later hit starts
   a new cluster when ``cts > anchor + cts_window``; otherwise it joins the
   current one. The window is inclusive, as in extractData (``NUMCTS = 48``).
4. Clusters with exactly one anode hit and one cathode hit become events.
   Multi-anode and multi-cathode clusters are dropped and cathodes are never
   summed.

The result, :class:`BoardEvents`, holds parallel arrays with one row per event:
raw PHAs and channel addresses only, so a different calibration needs no
rebuild. Rows are ordered by source, then by cluster (CTS) order.

Difference from extractData: extractData anchors its windows on the time-sorted
hit stream of the *whole system* and only then groups each window's hits by
(node, board). The cache stores every board separately, so dcalib anchors the
windows per board (the plan's rule). The two agree whenever no other board's
hit anchors a window between a board's anode and cathode hits.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import h5py
import numba
import numpy as np
import numpy.typing as npt

from dcalib.channels import N_CHANNELS, N_RENAS
from dcalib.options import SOURCE_IDS

logger = logging.getLogger(__name__)

__all__ = [
    "CLUSTER_CLASSES",
    "DEFAULT_CTS_WINDOW",
    "BoardEvents",
    "BoardHits",
    "PolarityHits",
    "build_board_events",
    "cluster_one_anode_one_cathode",
    "greedy_cluster_ids",
    "list_boards",
    "read_board_hits",
]

DEFAULT_CTS_WINDOW = 48
"""extractData's ``NUMCTS``: 1 us at the 48 MHz coarse clock."""

CLUSTER_CLASSES = 4
"""Cluster census bins per polarity: 0, 1, 2 and 3-or-more hits."""

_COINCIDENCES = "coincidences"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _empty(dtype: type[np.generic]) -> npt.NDArray[Any]:
    return np.empty(0, dtype=dtype)


@dataclass
class PolarityHits:
    """One polarity's hits of a board, in the cache's (file) order."""

    rena: npt.NDArray[np.int8] = field(default_factory=lambda: _empty(np.int8))
    channel: npt.NDArray[np.int8] = field(default_factory=lambda: _empty(np.int8))
    pha: npt.NDArray[np.int16] = field(default_factory=lambda: _empty(np.int16))
    cts: npt.NDArray[np.int64] = field(default_factory=lambda: _empty(np.int64))
    source: npt.NDArray[np.int8] = field(default_factory=lambda: _empty(np.int8))

    def __len__(self) -> int:
        return len(self.cts)

    def take(self, index: npt.NDArray[Any]) -> PolarityHits:
        """Return the hits selected by a boolean mask or an index array."""
        return PolarityHits(*(getattr(self, f.name)[index] for f in fields(self)))


@dataclass
class BoardHits:
    """Every anode and cathode hit of one board, as stored in the cache."""

    node: int
    board: int
    anode: PolarityHits
    cathode: PolarityHits


def _read_polarity(group: h5py.Group, prefix: str) -> PolarityHits:
    """Read one polarity of a board group; a missing polarity has no hits."""
    if f"{prefix}phas" not in group:
        return PolarityHits()
    rena = np.asarray(group[f"{prefix}renas"][:], dtype=np.int8)
    channel = np.asarray(group[f"{prefix}channels"][:], dtype=np.int8)
    pha = np.asarray(group[f"{prefix}phas"][:], dtype=np.int16)
    n = len(pha)
    if f"{prefix}timestamps" in group:
        cts = np.asarray(group[f"{prefix}timestamps"][:], dtype=np.int64)
    else:  # caches older than per-event timestamps cannot be clustered in time
        cts = np.zeros(n, dtype=np.int64)
    if f"{prefix}file_ids" in group:
        # adc2kev's rule: non-zero means Cs-137 (never let a wide dtype wrap to 0)
        source = np.where(group[f"{prefix}file_ids"][:] != 0, 1, 0).astype(np.int8)
    else:
        source = np.zeros(n, dtype=np.int8)
    # The cache only stores valid channel keys; guard the LUT indexing anyway.
    valid = (rena >= 0) & (rena < N_RENAS) & (channel >= 0) & (channel < N_CHANNELS)
    hits = PolarityHits(rena, channel, pha, cts, source)
    return hits if valid.all() else hits.take(valid)


def read_board_hits(cache_path: Path, node: int, board: int) -> BoardHits:
    """Read one board's anode and cathode hits from the cache (read-only).

    Args:
        cache_path: adc2kev calibration or diagnostic cache.
        node: Node number.
        board: Board number.

    Returns:
        The hits; both polarities are empty when the board has no group.
    """
    with h5py.File(cache_path, "r") as h5f:
        path = f"/{_COINCIDENCES}/node_{node}/board_{board}"
        if path not in h5f:
            return BoardHits(node, board, PolarityHits(), PolarityHits())
        group = h5f[path]
        return BoardHits(
            node, board, _read_polarity(group, "anode_"), _read_polarity(group, "cathode_")
        )


def list_boards(cache_path: Path) -> dict[tuple[int, int], int]:
    """Return every board under ``/coincidences`` with its total hit count.

    The counts come from the dataset shapes (no data is read) and serve to
    schedule the largest boards first.

    Returns:
        ``{(node, board): anode + cathode hits}``, sorted by (node, board).
    """
    boards: dict[tuple[int, int], int] = {}
    with h5py.File(cache_path, "r") as h5f:
        if _COINCIDENCES not in h5f:
            return boards
        for node_name, node_group in h5f[_COINCIDENCES].items():
            node = int(node_name.split("_")[1])
            for board_name, group in node_group.items():
                n = sum(
                    group[f"{prefix}phas"].shape[0]
                    for prefix in ("anode_", "cathode_")
                    if f"{prefix}phas" in group
                )
                boards[(node, int(board_name.split("_")[1]))] = int(n)
    return dict(sorted(boards.items()))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@numba.njit(cache=True, nogil=True)
def _greedy_cluster_ids(cts: npt.NDArray[np.int64], window: int) -> npt.NDArray[np.int64]:
    n = cts.shape[0]
    ids = np.empty(n, dtype=np.int64)
    if n == 0:
        return ids
    cluster = 0
    anchor = cts[0]
    for i in range(n):
        if cts[i] > anchor + window:
            cluster += 1
            anchor = cts[i]
        ids[i] = cluster
    return ids


def greedy_cluster_ids(cts: npt.NDArray[np.int64], window: int) -> npt.NDArray[np.int64]:
    """Assign CTS-sorted hits to greedy clusters.

    The first hit anchors cluster 0; a hit joins the current cluster when
    ``cts <= anchor + window`` and otherwise anchors the next cluster. A hit
    within ``window`` of the previous hit but more than ``window`` after the
    anchor therefore starts a new cluster.

    Args:
        cts: Hit timestamps sorted ascending.
        window: Inclusive window in CTS ticks (>= 0).

    Returns:
        Cluster number of each hit (0, 1, ...; non-decreasing).
    """
    result: npt.NDArray[np.int64] = _greedy_cluster_ids(
        np.ascontiguousarray(cts, dtype=np.int64), int(window)
    )
    return result


def cluster_one_anode_one_cathode(
    anode_cts: npt.NDArray[np.int64],
    cathode_cts: npt.NDArray[np.int64],
    window: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """Cluster one source's hits and pick the 1-anode/1-cathode clusters.

    Args:
        anode_cts: CTS of the anode hits (any order).
        cathode_cts: CTS of the cathode hits (any order).
        window: Inclusive greedy window in CTS ticks.

    Returns:
        ``(anode_index, cathode_index, census)``: the indices into the input
        arrays of the anode and the cathode of each 1A1C cluster, in cluster
        order, and a ``(4, 4)`` census of all clusters by
        ``[min(n_anodes, 3), min(n_cathodes, 3)]``.
    """
    n_anode = len(anode_cts)
    cts = np.concatenate([np.asarray(anode_cts, np.int64), np.asarray(cathode_cts, np.int64)])
    census = np.zeros((CLUSTER_CLASSES, CLUSTER_CLASSES), dtype=np.int64)
    if len(cts) == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, census
    order = np.argsort(cts, kind="stable")
    ids = greedy_cluster_ids(cts[order], window)
    is_anode = order < n_anode
    n_clusters = int(ids[-1]) + 1
    n_a = np.bincount(ids[is_anode], minlength=n_clusters)
    n_c = np.bincount(ids[~is_anode], minlength=n_clusters)
    top = CLUSTER_CLASSES - 1
    classes = np.minimum(n_a, top) * CLUSTER_CLASSES + np.minimum(n_c, top)
    census[:] = np.bincount(classes, minlength=CLUSTER_CLASSES**2).reshape(census.shape)
    in_good = ((n_a == 1) & (n_c == 1))[ids]
    anode_index = order[in_good & is_anode].astype(np.intp)
    cathode_index = (order[in_good & ~is_anode] - n_anode).astype(np.intp)
    return anode_index, cathode_index, census


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class BoardEvents:
    """The 1-anode/1-cathode events of one board (parallel arrays, one row each).

    Rows are ordered by source, then by cluster (CTS) order. Energies are not
    stored: they follow from the PHAs and a calibration
    (:func:`dcalib.calib.event_energies`).

    Attributes:
        node: Node number.
        board: Board number.
        cts_window: Clustering window the events were built with.
        source: Source id (0 = Ge-68, 1 = Cs-137).
        anode_rena: Anode RENA.
        anode_channel: Anode channel.
        anode_pha: Anode raw PHA.
        cathode_rena: Cathode RENA.
        cathode_channel: Cathode channel.
        cathode_pha: Cathode raw PHA.
        cts: Anode CTS timestamp.
        dcts: Cathode CTS minus anode CTS.
        census: ``(2, 4, 4)`` cluster counts per source by
            ``[source, min(n_anodes, 3), min(n_cathodes, 3)]``.
    """

    node: int
    board: int
    cts_window: int
    source: npt.NDArray[np.int8]
    anode_rena: npt.NDArray[np.int8]
    anode_channel: npt.NDArray[np.int8]
    anode_pha: npt.NDArray[np.int16]
    cathode_rena: npt.NDArray[np.int8]
    cathode_channel: npt.NDArray[np.int8]
    cathode_pha: npt.NDArray[np.int16]
    cts: npt.NDArray[np.int64]
    dcts: npt.NDArray[np.int32]
    census: npt.NDArray[np.int64]

    ROW_FIELDS = (
        "source",
        "anode_rena",
        "anode_channel",
        "anode_pha",
        "cathode_rena",
        "cathode_channel",
        "cathode_pha",
        "cts",
        "dcts",
    )

    def __len__(self) -> int:
        return len(self.source)

    @property
    def nbytes(self) -> int:
        """Memory held by the row arrays (for the GUI's board LRU)."""
        return int(sum(getattr(self, name).nbytes for name in self.ROW_FIELDS))

    def take(self, index: npt.NDArray[Any]) -> BoardEvents:
        """Return the rows selected by a boolean mask or an index array (same census)."""
        rows = {name: getattr(self, name)[index] for name in self.ROW_FIELDS}
        return BoardEvents(self.node, self.board, self.cts_window, census=self.census, **rows)

    def anode_codes(self) -> npt.NDArray[np.int64]:
        """Return ``rena * 36 + channel`` of each row's anode."""
        return self.anode_rena.astype(np.int64) * N_CHANNELS + self.anode_channel

    def anode_rows(self) -> dict[tuple[int, int], npt.NDArray[np.intp]]:
        """Group the rows by anode.

        Returns:
            ``{(rena, channel): row indices}``, keys sorted, each index array
            ascending (so in source, then CTS order).
        """
        codes = self.anode_codes()
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        values, starts = np.unique(sorted_codes, return_index=True)
        ends = np.append(starts[1:], len(order))
        return {
            (int(v) // N_CHANNELS, int(v) % N_CHANNELS): order[s:e].astype(np.intp)
            for v, s, e in zip(values, starts, ends)
        }

    def iter_sources(self) -> Iterator[tuple[int, npt.NDArray[np.bool_]]]:
        """Yield ``(source id, row mask)`` for each source."""
        for source_id in SOURCE_IDS:
            yield source_id, self.source == source_id

    def one_anode_one_cathode_share(self, source: int) -> float:
        """Fraction of the source's clusters that are 1A1C (NaN without clusters)."""
        census = self.census[source]
        total = int(census.sum())
        return float(census[1, 1]) / total if total else float("nan")


def events_from_hits(hits: BoardHits, cts_window: int = DEFAULT_CTS_WINDOW) -> BoardEvents:
    """Build the 1A1C events of a board from its hits (steps 2-4 of the module doc)."""
    parts: list[dict[str, npt.NDArray[Any]]] = []
    census = np.zeros((len(SOURCE_IDS), CLUSTER_CLASSES, CLUSTER_CLASSES), dtype=np.int64)
    for source_id in SOURCE_IDS:
        anode = hits.anode.take(hits.anode.source == source_id)
        cathode = hits.cathode.take(hits.cathode.source == source_id)
        a_idx, c_idx, census[source_id] = cluster_one_anode_one_cathode(
            anode.cts, cathode.cts, cts_window
        )
        parts.append(
            {
                "source": np.full(len(a_idx), source_id, dtype=np.int8),
                "anode_rena": anode.rena[a_idx],
                "anode_channel": anode.channel[a_idx],
                "anode_pha": anode.pha[a_idx],
                "cathode_rena": cathode.rena[c_idx],
                "cathode_channel": cathode.channel[c_idx],
                "cathode_pha": cathode.pha[c_idx],
                "cts": anode.cts[a_idx],
                "dcts": (cathode.cts[c_idx] - anode.cts[a_idx]).astype(np.int32),
            }
        )
    rows = {name: np.concatenate([part[name] for part in parts]) for name in BoardEvents.ROW_FIELDS}
    return BoardEvents(hits.node, hits.board, int(cts_window), census=census, **rows)


def build_board_events(
    cache_path: Path, node: int, board: int, cts_window: int = DEFAULT_CTS_WINDOW
) -> BoardEvents:
    """Read a board from the cache and build its 1-anode/1-cathode events.

    Args:
        cache_path: adc2kev calibration cache (opened read-only).
        node: Node number.
        board: Board number.
        cts_window: Inclusive greedy clustering window in CTS ticks.

    Returns:
        The board's events (empty when the board is not in the cache).
    """
    return events_from_hits(read_board_hits(cache_path, node, board), cts_window)
