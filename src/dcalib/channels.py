"""Anode and cathode channel sets, electrode labels and strip positions.

A board carries two RENA-3 ASICs with 47 channels wired to electrodes: 39
anodes (``A01``-``A39``) and 8 cathodes (``C01``-``C08``). Which (RENA,
channel) reads which electrode depends on the board parity and comes from
adc2kev's :class:`~adc2kev.tools.electrode_map.ElectrodeMap` (the packaged
``.cmf`` load-balance files). Physical strip positions follow
``ElectrodeMap.physical_strip_position``: position 1 is the anode on the
low-node side of the panel.

The adc2kev cache splits every board's hits into ``anode_*`` and ``cathode_*``
datasets with the parser's polarity map; for the active boards (15-30) that
split agrees with the ``.cmf`` files, so the event layer trusts the cache and
this module supplies the channel lists, labels and positions.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from adc2kev.tools.electrode_map import ElectrodeMap
from adc2kev.tools.geometry import (
    ACTIVE_BOARDS,
    ANODES_PER_BOARD,
    CATHODES_PER_BOARD,
    PANEL_NODES,
    is_active_board,
    panel_for_node,
)

__all__ = [
    "ACTIVE_BOARDS",
    "ACTIVE_NODES",
    "ANODES_PER_BOARD",
    "CATHODES_PER_BOARD",
    "N_CHANNELS",
    "N_RENAS",
    "PANEL_NODES",
    "AnodeKey",
    "anode_channels",
    "anodes_by_strip",
    "cathode_channels",
    "cathode_mask_lut",
    "electrode_label",
    "electrode_map",
    "is_active_board",
    "is_active_channel",
    "is_cathode",
    "panel_for_node",
    "strip_position",
]

N_RENAS = 2
"""RENA-3 ASICs per board."""

N_CHANNELS = 36
"""Channels per RENA-3 ASIC (only 4-28 on RENA 0 and 7-28 on RENA 1 are wired)."""

ACTIVE_NODES: tuple[int, ...] = tuple(node for nodes in PANEL_NODES.values() for node in nodes)
"""The ten detector nodes (1-5 in panel 1, 6-10 in panel 2)."""

_ACTIVE_RANGES: dict[int, tuple[int, int]] = {0: (4, 28), 1: (7, 28)}


class AnodeKey(NamedTuple):
    """Address of one anode channel; sorts by (node, board, rena, channel)."""

    node: int
    board: int
    rena: int
    channel: int

    def __str__(self) -> str:
        return f"n{self.node} b{self.board} r{self.rena} ch{self.channel}"


@functools.lru_cache(maxsize=1)
def electrode_map() -> ElectrodeMap:
    """Return the shared adc2kev :class:`ElectrodeMap` (packaged ``.cmf`` files)."""
    return ElectrodeMap.default()


def is_active_channel(rena: int, channel: int) -> bool:
    """Return True for the wired channels: RENA 0 channels 4-28, RENA 1 channels 7-28."""
    bounds = _ACTIVE_RANGES.get(rena)
    return bounds is not None and bounds[0] <= channel <= bounds[1]


def electrode_label(board: int, rena: int, channel: int) -> str:
    """Return the electrode label of a channel, e.g. ``"A17"`` or ``"C03"``.

    Raises:
        KeyError: If ``(rena, channel)`` is not wired to an electrode.
    """
    return electrode_map().electrode_label(board, rena, channel)


def is_cathode(board: int, rena: int, channel: int) -> bool:
    """Return True if the channel reads a cathode strip.

    Raises:
        KeyError: If ``(rena, channel)`` is not wired to an electrode.
    """
    return electrode_map().is_cathode(board, rena, channel)


def strip_position(board: int, rena: int, channel: int) -> int:
    """Return the anode's physical strip position 1..39 (1 = low-node side).

    Raises:
        ValueError: If the channel is a cathode.
        KeyError: If ``(rena, channel)`` is not wired to an electrode.
    """
    return electrode_map().physical_strip_position(board, rena, channel)


@functools.lru_cache(maxsize=2)
def _parity_channels(even: bool) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """(anodes, cathodes) of one board parity, each sorted by (rena, channel)."""
    board = 16 if even else 15
    emap = electrode_map()
    anodes: list[tuple[int, int]] = []
    cathodes: list[tuple[int, int]] = []
    for rena, (first, last) in sorted(_ACTIVE_RANGES.items()):
        for channel in range(first, last + 1):
            if emap.is_cathode(board, rena, channel):
                cathodes.append((rena, channel))
            else:
                anodes.append((rena, channel))
    return tuple(anodes), tuple(cathodes)


def anode_channels(board: int) -> tuple[tuple[int, int], ...]:
    """Return the board's 39 anode ``(rena, channel)`` pairs, sorted."""
    return _parity_channels(board % 2 == 0)[0]


def cathode_channels(board: int) -> tuple[tuple[int, int], ...]:
    """Return the board's 8 cathode ``(rena, channel)`` pairs, sorted."""
    return _parity_channels(board % 2 == 0)[1]


def anodes_by_strip(board: int) -> tuple[tuple[int, int], ...]:
    """Return the board's anode ``(rena, channel)`` pairs in physical strip order 1..39."""
    return tuple(
        electrode_map().channel_for_strip_position(board, position)
        for position in range(1, ANODES_PER_BOARD + 1)
    )


def cathode_mask_lut(board: int) -> npt.NDArray[np.bool_]:
    """Return a ``(2, 36)`` table, True at the board's cathode (rena, channel) slots."""
    lut = np.zeros((N_RENAS, N_CHANNELS), dtype=np.bool_)
    for rena, channel in cathode_channels(board):
        lut[rena, channel] = True
    return lut
