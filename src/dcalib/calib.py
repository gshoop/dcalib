"""keV calibrations: loading (cache or ``.kev``), per-board LUTs, energies and C/A.

The calibration source is the adc2kev cache's stored calibrations (the
default) or a ``.kev`` file (``--kev``). Either is reduced with
:func:`adc2kev.analysis.valid_linear_calibrations` (success and slope != 0)
to one linear ``keV = slope * pha + intercept`` per channel (plan 5.2).

The **fingerprint** is the SHA-256 of the sorted ``(node, board, rena,
channel, slope, intercept)`` table, with the floats in their exact hex form. It
is stored with the results and tells when they were made with a different
calibration (a changed adc2kev calibration makes them stale, plan 6.2).

:func:`event_energies` converts a board's events to keV, the normalised energy
``x = A_keV / E0`` (E0 = 511 keV for Ge-68, 662 keV for Cs-137) and the depth
proxy ``r = C_keV / A_keV``, both uncorrected, as the ``.dcc`` consumers
compute them.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from adc2kev.analysis import valid_linear_calibrations
from adc2kev.cache import CalibrationCache
from adc2kev.io import KEVExporter

from dcalib.channels import N_CHANNELS, N_RENAS, AnodeKey
from dcalib.events import BoardEvents
from dcalib.options import SOURCE_ENERGY_KEV, SOURCE_IDS

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATION_SOURCE_CACHE",
    "BoardLUT",
    "CalibrationError",
    "Calibrations",
    "EventEnergies",
    "calibration_fingerprint",
    "event_energies",
    "file_sha256",
    "load_calibrations",
    "source_e0",
]

CALIBRATION_SOURCE_CACHE = "cache"
"""``Calibrations.source`` of calibrations read from the adc2kev cache."""

LinearTable = Mapping[AnodeKey, tuple[float, float]]
"""Valid ``(slope, intercept)`` per channel (anodes and cathodes alike)."""


class CalibrationError(Exception):
    """The calibrations cannot be loaded (missing file, bad format, none valid)."""


@dataclass(frozen=True)
class BoardLUT:
    """(RENA, channel)-indexed linear calibration of one board.

    ``slope`` and ``intercept`` are ``(2, 36)`` float64 tables; both are NaN
    for a channel without a valid calibration, so :meth:`energy` gives NaN for
    its events.
    """

    node: int
    board: int
    slope: npt.NDArray[np.float64]
    intercept: npt.NDArray[np.float64]

    @property
    def calibrated(self) -> npt.NDArray[np.bool_]:
        """``(2, 36)`` table, True where the channel has a valid calibration."""
        return np.asarray(~np.isnan(self.slope))

    def is_calibrated(self, rena: int, channel: int) -> bool:
        """Return True if ``(rena, channel)`` has a valid calibration."""
        return bool(not np.isnan(self.slope[rena, channel]))

    def energy(
        self,
        renas: npt.ArrayLike,
        channels: npt.ArrayLike,
        phas: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Return ``slope * pha + intercept`` per hit in keV (NaN when uncalibrated).

        Args:
            renas: RENA of each hit (0 or 1).
            channels: Channel of each hit (0-35).
            phas: Raw PHA of each hit.
        """
        r = np.asarray(renas, dtype=np.intp)
        c = np.asarray(channels, dtype=np.intp)
        return np.asarray(
            self.slope[r, c] * np.asarray(phas, dtype=np.float64) + self.intercept[r, c],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Calibrations:
    """The valid linear calibrations of every channel and their provenance.

    Attributes:
        table: ``(slope, intercept)`` per valid channel, keyed by
            :class:`AnodeKey` (cathodes use the same key type).
        source: ``"cache"`` or the resolved ``.kev`` path.
        source_sha256: SHA-256 of the ``.kev`` file (``None`` for the cache).
        fingerprint: :func:`calibration_fingerprint` of ``table``.
    """

    table: LinearTable
    source: str
    source_sha256: str | None
    fingerprint: str

    @classmethod
    def from_table(
        cls, table: LinearTable, source: str, source_sha256: str | None = None
    ) -> Calibrations:
        """Build from a table (normalising keys and values) and fingerprint it."""
        normalised = {
            AnodeKey(int(k[0]), int(k[1]), int(k[2]), int(k[3])): (float(s), float(i))
            for k, (s, i) in table.items()
        }
        return cls(normalised, source, source_sha256, calibration_fingerprint(normalised))

    def __len__(self) -> int:
        return len(self.table)

    def is_calibrated(self, key: AnodeKey) -> bool:
        """Return True if the channel has a valid calibration."""
        return key in self.table

    def board_lut(self, node: int, board: int) -> BoardLUT:
        """Return the (RENA, channel)-indexed calibration of one board."""
        slope = np.full((N_RENAS, N_CHANNELS), np.nan, dtype=np.float64)
        intercept = np.full((N_RENAS, N_CHANNELS), np.nan, dtype=np.float64)
        for key, (s, i) in self.table.items():
            if key.node == node and key.board == board:
                slope[key.rena, key.channel] = s
                intercept[key.rena, key.channel] = i
        return BoardLUT(node, board, slope, intercept)

    def board_luts(self) -> dict[tuple[int, int], BoardLUT]:
        """Return the LUT of every board that has at least one calibrated channel."""
        boards = sorted({(key.node, key.board) for key in self.table})
        return {nb: self.board_lut(*nb) for nb in boards}


def calibration_fingerprint(table: LinearTable) -> str:
    """Return the SHA-256 hex digest of the sorted calibration table.

    Each channel contributes one line ``node board rena channel slope
    intercept`` with the floats in ``float.hex`` form, so any change of any
    value changes the digest and the digest does not depend on dict order.
    """
    digest = hashlib.sha256()
    for key in sorted(table):
        slope, intercept = table[key]
        node, board, rena, channel = (int(v) for v in key)
        digest.update(
            f"{node} {board} {rena} {channel} "
            f"{float(slope).hex()} {float(intercept).hex()}\n".encode("ascii")
        )
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a whole file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibrations(cache_path: Path, kev_path: Path | None = None) -> Calibrations:
    """Load the valid linear calibrations from the cache or a ``.kev`` override.

    The cache is opened read-only (through adc2kev).

    Args:
        cache_path: adc2kev calibration cache (``*.cache.h5``).
        kev_path: ``.kev`` file that replaces the cache's calibrations.

    Returns:
        The calibrations with their provenance and fingerprint.

    Raises:
        CalibrationError: If the ``.kev`` file is missing or malformed, or no
            channel has a valid calibration.
    """
    if kev_path is not None:
        kev_path = Path(kev_path)
        try:
            results = KEVExporter().read(kev_path)
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"Cannot read calibration file {kev_path}: {exc}") from exc
        source = str(kev_path.resolve())
        sha = file_sha256(kev_path)
    else:
        results = CalibrationCache(Path(cache_path)).load_all_calibrations()
        source = CALIBRATION_SOURCE_CACHE
        sha = None
    table = {
        AnodeKey(key.node, key.board, key.rena, key.channel): value
        for key, value in valid_linear_calibrations(results).items()
    }
    if not table:
        where = kev_path if kev_path is not None else f"the cache {cache_path}"
        raise CalibrationError(f"No valid keV calibration in {where}")
    calibrations = Calibrations.from_table(table, source, sha)
    logger.info(
        "Loaded %d valid calibrations from %s (fingerprint %s)",
        len(calibrations),
        source,
        calibrations.fingerprint[:12],
    )
    return calibrations


# ---------------------------------------------------------------------------
# Energies and C/A
# ---------------------------------------------------------------------------


def source_e0(source: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
    """Return the photopeak energy E0 in keV of each row's source."""
    table = np.array([SOURCE_ENERGY_KEV[s] for s in SOURCE_IDS], dtype=np.float64)
    return np.asarray(table[np.asarray(source, dtype=np.intp)], dtype=np.float64)


@dataclass(frozen=True)
class EventEnergies:
    """Calibrated energies of a board's events (parallel to the event rows).

    Attributes:
        anode_kev: Anode energy in keV (NaN when the anode is uncalibrated).
        cathode_kev: Cathode energy in keV (NaN when the cathode is uncalibrated).
        x: ``anode_kev / E0`` of the row's source.
        r: ``cathode_kev / anode_kev`` (NaN when either side is uncalibrated).
    """

    anode_kev: npt.NDArray[np.float64]
    cathode_kev: npt.NDArray[np.float64]
    x: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]

    @property
    def anode_calibrated(self) -> npt.NDArray[np.bool_]:
        """True for rows whose anode has a calibration."""
        return np.asarray(~np.isnan(self.anode_kev))

    @property
    def cathode_calibrated(self) -> npt.NDArray[np.bool_]:
        """True for rows whose cathode has a calibration."""
        return np.asarray(~np.isnan(self.cathode_kev))


def event_energies(events: BoardEvents, lut: BoardLUT) -> EventEnergies:
    """Convert a board's events to keV, x and r with the board's calibration.

    Args:
        events: The board's 1A1C events.
        lut: The same board's calibration.

    Returns:
        The energies, one row per event.

    Raises:
        ValueError: If ``lut`` belongs to another board.
    """
    if (lut.node, lut.board) != (events.node, events.board):
        raise ValueError(
            f"LUT of n{lut.node} b{lut.board} used for events of n{events.node} b{events.board}"
        )
    anode = lut.energy(events.anode_rena, events.anode_channel, events.anode_pha)
    cathode = lut.energy(events.cathode_rena, events.cathode_channel, events.cathode_pha)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = anode / source_e0(events.source)
        r = cathode / anode
    return EventEnergies(anode, cathode, np.asarray(x), np.asarray(r))
