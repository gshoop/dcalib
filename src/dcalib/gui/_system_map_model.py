"""Model behind the System Map dock.

Forked from uvcorr (``uvcorr/gui/_system_map_model.py``), itself forked from
adc2kev 2.2.7 ``gui/_system_map_model.py``; keep the structure close to them so
fixes can be carried across. uvcorr's changes (views instead of calibration
dictionaries, colour modes, lazy tooltips, incremental updates, no-data vs
not-fitted channels, unplaced results) are kept. Changes for dcalib:

- **Anodes and cathodes mean different things.** An anode's view carries its
  depth-fit status, flags, options source (batch or override), review state
  and metrics; a cathode's view only its keV calibration status
  (``calibrated`` / ``uncalibrated``, :data:`~dcalib.gui.map_colors.CATHODE_STATUSES`).
  Cathodes keep their calibration colours in every colour mode (plan
  section 9: "cathode cells are coloured by keV-calibration status"), so
  the metric limits are computed over the anodes only.
- **Review.** A rejected anode is drawn in the ``rejected`` category whatever
  its status (:attr:`ChannelView.review`).
- The metric modes are dcalib's (``cv_gain``, ``peak_spread``, the FWHMs, the
  Ge/Cs difference), and ``fwhm_511_change`` is derived from the two 511 keV
  FWHMs when a view does not carry it.

The model uses Qt only for ``QColor``, a value type that needs no
``QApplication``, so it is unit tested without a display. The widgets in
:mod:`dcalib.gui.system_map` consume :class:`SystemMapModel` and only handle
painting and hit-testing.

The model covers the fixed deployment grid (nodes 1-10 x boards 15-30, from
``adc2kev.tools.geometry``) and lays each board out in physical electrode
order via :class:`~adc2kev.tools.electrode_map.ElectrodeMap`: 39 anode
strips in panel coordinates (position 1 on the low-node side, so odd boards
show A39 first) and 8 cathodes C01..C08.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
from adc2kev.tools.electrode_map import ElectrodeMap
from adc2kev.tools.geometry import (
    ACTIVE_BOARDS,
    ANODES_PER_BOARD,
    CATHODES_PER_BOARD,
    PANEL_NODES,
    is_even_board,
)
from PyQt6.QtGui import QColor

from dcalib.channels import electrode_map as default_electrode_map
from dcalib.gui.map_colors import (
    ANODE_CATEGORIES,
    CATEGORY_CATHODE_CALIBRATED,
    CATEGORY_FLAGGED,
    CATEGORY_LABELS,
    CATEGORY_NO_DATA,
    CATEGORY_NOT_FITTED,
    CATHODE_CATEGORIES,
    CATHODE_STATUSES,
    METRIC_SPECS,
    MODE_STATUS,
    MetricSpec,
    category_qcolor,
    check_color_mode,
    metric_colors,
    metric_limits,
    status_category,
)
from dcalib.options import REVIEW_STATES, STATUS_OK, STATUSES

__all__ = [
    "GRID_BOARDS",
    "GRID_NODES",
    "KINDS",
    "NO_DATA_SUMMARY",
    "OPTIONS_BATCH",
    "OPTIONS_OVERRIDE",
    "OPTIONS_SOURCES",
    "TOOLTIP_METRICS",
    "BoardCells",
    "ChannelAddress",
    "ChannelKeyT",
    "ChannelTuple",
    "ChannelView",
    "MapCell",
    "SystemMapModel",
    "as_address",
    "build_cell_tooltip",
    "build_system_map",
    "classify_view",
    "format_board_summary",
    "format_summary",
    "is_in_grid",
    "metric_values",
    "recolor_system_map",
    "status_counts",
    "update_system_map",
]

ChannelTuple = tuple[int, int, int, int]
"""A channel as a plain ``(node, board, rena, channel)`` tuple (the input key type)."""

ChannelKeyT = TypeVar("ChannelKeyT", bound=ChannelTuple)
"""Key type of a views mapping: plain tuples, :class:`ChannelAddress` or any other
4-int ``NamedTuple``. A ``Mapping`` is invariant in its key type, so the
functions taking views are generic over it rather than typed
``Mapping[ChannelTuple, ...]``, which would reject a ``dict[ChannelAddress, ...]``."""


class ChannelAddress(NamedTuple):
    """Hardware address of one channel.

    A ``NamedTuple``, so it equals and hashes like the plain
    ``(node, board, rena, channel)`` tuple: a mapping keyed by plain tuples
    (or by another ``NamedTuple`` with the same fields) is looked up with it
    directly.
    """

    node: int
    board: int
    rena: int
    channel: int


def as_address(channel: ChannelTuple) -> ChannelAddress:
    """Normalise a ``(node, board, rena, channel)`` tuple to a :class:`ChannelAddress`."""
    if isinstance(channel, ChannelAddress):
        return channel
    node, board, rena, ch = channel
    return ChannelAddress(int(node), int(board), int(rena), int(ch))


OPTIONS_BATCH = "batch"
"""``ChannelView.options_source`` of a channel fitted by the batch run."""

OPTIONS_OVERRIDE = "override"
"""``ChannelView.options_source`` of a channel re-fitted with its own options."""

OPTIONS_SOURCES: tuple[str, ...] = (OPTIONS_BATCH, OPTIONS_OVERRIDE)


@dataclass(frozen=True)
class ChannelView:
    """What the map shows of one channel.

    Built by the main window from the results (one per anode with a result)
    and the calibrations (one per cathode of a board with events); the map
    never sees the result objects themselves. Views are immutable and
    hashable: the hash covers ``status``, ``flags``, ``options_source`` and
    ``review`` (not ``metrics``), and equality compares all five.

    Attributes:
        status: An anode's depth-fit status (``dcalib.options.STATUSES``) or a
            cathode's calibration status (``map_colors.CATHODE_STATUSES``).
        flags: Warning flags, in the order they are shown in the tooltip.
        options_source: ``"batch"`` or ``"override"`` (plan section 6.2).
        review: ``""`` or ``"rejected"`` (plan D10).
        metrics: Metric values by key; a missing key, None or NaN means
            unavailable. The map reads the keys of ``map_colors.METRIC_SPECS``
            and ``n_events``; result fields can be passed verbatim, and
            ``fwhm_511_change`` is derived from ``fwhm_511_before`` and
            ``fwhm_511_after`` when missing. Stored as a read-only copy.

    Raises:
        ValueError: If ``status``, ``options_source`` or ``review`` is not a
            known value.
    """

    status: str
    flags: tuple[str, ...] = ()
    options_source: str = OPTIONS_BATCH
    metrics: Mapping[str, float] = field(default_factory=dict, hash=False)
    review: str = ""

    def __post_init__(self) -> None:
        """Validate; store ``flags`` as a tuple and ``metrics`` as a read-only copy."""
        if self.status not in STATUSES and self.status not in CATHODE_STATUSES:
            raise ValueError(
                f"status must be one of {STATUSES + CATHODE_STATUSES}, got {self.status!r}"
            )
        if self.options_source not in OPTIONS_SOURCES:
            raise ValueError(
                f"options_source must be one of {OPTIONS_SOURCES}, got {self.options_source!r}"
            )
        if self.review not in REVIEW_STATES:
            raise ValueError(f"review must be one of {REVIEW_STATES}, got {self.review!r}")
        if not isinstance(self.flags, tuple):
            object.__setattr__(self, "flags", tuple(self.flags))
        metrics = dict(self.metrics)
        if metrics.get("fwhm_511_change") is None:
            before = metrics.get("fwhm_511_before")
            after = metrics.get("fwhm_511_after")
            if before is not None and after is not None and float(before) > 0:
                metrics["fwhm_511_change"] = float(after) / float(before) - 1.0
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def is_cathode(self) -> bool:
        """Whether this is a cathode's calibration view."""
        return self.status in CATHODE_STATUSES

    @property
    def is_rejected(self) -> bool:
        """Whether the anode was rejected in the review."""
        return bool(self.review)

    @property
    def is_override(self) -> bool:
        """Whether the channel was re-fitted with its own options."""
        return self.options_source == OPTIONS_OVERRIDE

    def metric(self, key: str) -> float:
        """Return metric ``key`` as a float, NaN when it is missing or None."""
        value = self.metrics.get(key)
        return math.nan if value is None else float(value)


def _format_count(value: float) -> str:
    return f"{value:.0f}"


TOOLTIP_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("n_events", "Events", formatter=_format_count),
    MetricSpec("n_selected", "Selected", formatter=_format_count),
    MetricSpec("degree", "Degree", formatter=_format_count),
    *METRIC_SPECS.values(),
)
"""Metrics listed in a cell tooltip (when finite), in order."""

# Nodes and boards of the fixed deployment grid, in drawing order.
GRID_NODES: tuple[int, ...] = tuple(node for nodes in PANEL_NODES.values() for node in nodes)
GRID_BOARDS: tuple[int, ...] = ACTIVE_BOARDS

_GRID_NODE_SET = frozenset(GRID_NODES)
_GRID_BOARD_SET = frozenset(GRID_BOARDS)

KINDS: tuple[str, ...] = ("anode", "cathode")

NO_DATA_SUMMARY = "No board data loaded"

_NAN = math.nan
_NBSP = " "


@dataclass(frozen=True)
class MapCell:
    """One electrode on the map.

    Attributes:
        channel: Hardware address of the electrode.
        label: Electrode label from the ``.cmf`` map, e.g. ``"A17"`` or ``"C03"``.
        position: Physical strip position 1..39 for anodes; cathode index 1..8
            for cathodes.
        kind: ``"anode"`` or ``"cathode"``.
        category: Status category (one of ``map_colors.CATEGORIES``); it does
            not depend on the colour mode.
        fill: Fill colour in the model's colour mode. Shared between cells;
            do not mutate it.
        view: The channel's result view, if it has one.
        value: The metric value coloured in a metric mode (NaN when missing,
            and always NaN in the status mode).
    """

    channel: ChannelAddress
    label: str
    position: int
    kind: str
    category: str
    fill: QColor
    view: ChannelView | None = None
    value: float = _NAN
    # Tooltip cache: None until first read (see ``tooltip``). Not compared.
    _tooltip: str | None = field(default=None, repr=False, compare=False)

    @property
    def tooltip(self) -> str:
        """Multi-line plain-text description for hover, built on first access.

        It does not depend on the colour mode or the informational flags.
        """
        text = self._tooltip
        if text is None:
            text = build_cell_tooltip(
                self.channel,
                self.label,
                self.position,
                self.kind,
                self.view,
                has_data=self.category != CATEGORY_NO_DATA,
            )
            object.__setattr__(self, "_tooltip", text)  # a cache, not state
        return text

    @property
    def is_override(self) -> bool:
        """Whether the channel's result comes from a per-channel override fit."""
        return self.view is not None and self.view.is_override


@dataclass(frozen=True)
class BoardCells:
    """All cells of one board.

    Attributes:
        node: Node number.
        board: Board number.
        anodes: Exactly 39 cells ordered by physical strip position 1..39.
        cathodes: Exactly 8 cells ordered C01..C08.
        has_data: Whether the board has events. Boards without data are
            drawn as uniform dark tiles and excluded from the summaries.
    """

    node: int
    board: int
    anodes: tuple[MapCell, ...]
    cathodes: tuple[MapCell, ...]
    has_data: bool

    def cells(self, kind: str) -> tuple[MapCell, ...]:
        """Return the anode or cathode cells (``kind`` is ``"anode"`` / ``"cathode"``)."""
        _check_kind(kind)
        return self.anodes if kind == "anode" else self.cathodes


@dataclass(frozen=True)
class SystemMapModel:
    """Snapshot of the whole detector for one repaint.

    Attributes:
        boards: One entry per ``(node, board)`` in the fixed grid, in
            node-major drawing order. Always holds all 160 boards.
        unmapped_boards: Sorted, distinct ``(node, board)`` pairs that carry
            results or data but lie outside the grid, so the widget can warn
            about them instead of silently dropping them.
        color_mode: Colour mode the fills were computed for
            (``map_colors.COLOR_MODES``).
        informational_flags: Flags that did not make an ``ok`` channel
            ``flagged``.
        limits: In a metric mode, the colour limits of each kind (``"anode"``,
            ``"cathode"``; None when the kind has no finite value). Empty in
            the status mode.
        unplaced_channels: Sorted keys of views on grid boards that are not
            electrodes (inactive channels, bad RENA numbers); they are not
            drawn and do not give their board data.
        data_channels_known: Whether the model was built with a list of the
            channels that have events (``build_system_map(data_channels=...)``).
    """

    boards: dict[tuple[int, int], BoardCells]
    unmapped_boards: tuple[tuple[int, int], ...]
    color_mode: str = MODE_STATUS
    informational_flags: frozenset[str] = frozenset()
    limits: Mapping[str, tuple[float, float] | None] = field(default_factory=dict)
    unplaced_channels: tuple[ChannelAddress, ...] = ()
    data_channels_known: bool = False

    def locate(self, channel: ChannelTuple) -> tuple[BoardCells, str, int] | None:
        """Find the cell that shows ``channel``.

        Returns:
            ``(board, kind, position)`` with ``kind`` ``"anode"`` or
            ``"cathode"`` and ``position`` 1-based within that kind, or
            ``None`` when the channel's board is outside the grid (or the
            channel is not an electrode).
        """
        board = self.boards.get((channel[0], channel[1]))
        if board is None:
            return None
        for kind in KINDS:
            for cell in board.cells(kind):
                if cell.channel == channel:
                    return board, kind, cell.position
        return None

    def cell(self, channel: ChannelTuple) -> MapCell | None:
        """Return the cell that shows ``channel``, or None (see :meth:`locate`)."""
        located = self.locate(channel)
        if located is None:
            return None
        board, kind, position = located
        return board.cells(kind)[position - 1]

    def limits_for(self, kind: str) -> tuple[float, float] | None:
        """Colour limits of ``kind`` in the metric mode (None in the status mode)."""
        _check_kind(kind)
        return self.limits.get(kind)


def is_in_grid(node: int, board: int) -> bool:
    """Return True when ``(node, board)`` is drawn by the map."""
    return node in _GRID_NODE_SET and board in _GRID_BOARD_SET


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def classify_view(
    view: ChannelView | None,
    has_data: bool = True,
    informational_flags: Collection[str] = frozenset(),
) -> str:
    """Derive the status category of one channel.

    Rules, evaluated in order:

    1. No view: ``no_data`` when the channel has no events (``has_data`` is
       False), otherwise ``not_fitted``.
    2. A view whose ``n_events`` metric is exactly 0 -> ``no_data``.
    3. Otherwise the view's status and flags decide (see
       :func:`dcalib.gui.map_colors.status_category`): ``ok`` / ``flagged`` /
       ``failed`` / ``too_few``.

    A view always counts as data, whatever ``has_data`` says.

    Args:
        view: The channel's view, if any.
        has_data: Whether the channel has events (only used without a view).
        informational_flags: Flags that do not make an ``ok`` channel
            ``flagged``.

    Returns:
        One of ``map_colors.CATEGORIES``.
    """
    if view is None:
        return status_category(None, has_data=has_data)
    if view.metric("n_events") == 0:
        return CATEGORY_NO_DATA
    return status_category(view.status, view.flags, informational_flags, review=view.review)


def build_cell_tooltip(
    channel: ChannelTuple,
    label: str,
    position: int,
    kind: str,
    view: ChannelView | None,
    has_data: bool = True,
) -> str:
    """Render the plain-text hover tooltip for one cell.

    Args:
        channel: Hardware address of the electrode.
        label: Electrode label, e.g. ``"A17"``.
        position: Strip position (anodes) or cathode index (cathodes).
        kind: ``"anode"`` or ``"cathode"``.
        view: The channel's view, if any.
        has_data: Whether the channel has events (only used without a view).

    Returns:
        Newline-separated text: header, hardware address and status line,
        then (with a view) the flags, the options source and every finite
        metric of :data:`TOOLTIP_METRICS`.
    """
    _check_kind(kind)
    node, board, rena, ch = channel
    where = f"strip position {position}" if kind == "anode" else f"cathode {position}"
    lines = [f"{label} ({where})", f"Node {node}  Board {board}  RENA {rena}  Ch {ch}"]
    if view is None:
        category = CATEGORY_NOT_FITTED if has_data else CATEGORY_NO_DATA
        lines.append(f"Status: {CATEGORY_LABELS[category].lower()}")
        return "\n".join(lines)

    if view.is_cathode:
        lines.append(f"keV calibration: {view.status}")
        return "\n".join(lines)
    lines.append(f"Status: {view.status}")
    if view.review:
        lines.append(f"Review: {view.review}")
    if view.flags:
        lines.append("Flags: " + ", ".join(view.flags))
    lines.append(f"Options: {view.options_source}")
    for spec in TOOLTIP_METRICS:
        value = view.metric(spec.key)
        if math.isfinite(value):
            lines.append(f"{spec.label}: {spec.format(value)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _Slot:
    """Static identity of one electrode position on one board."""

    channel: ChannelAddress
    label: str
    position: int


@dataclass(frozen=True)
class _BoardLayout:
    """Static layout of one board: 39 anode slots then 8 cathode slots."""

    anodes: tuple[_Slot, ...]
    cathodes: tuple[_Slot, ...]
    members: frozenset[ChannelAddress]

    def slots(self, kind: str) -> tuple[_Slot, ...]:
        return self.anodes if kind == "anode" else self.cathodes


@functools.lru_cache(maxsize=512)
def _board_layout(electrode_map: ElectrodeMap, node: int, board: int) -> _BoardLayout:
    """Lay out one board's anodes (physical order) and cathodes (C01..C08).

    The layout never changes between refreshes, so it is cached per electrode
    map instance and board (160 entries for the default map).
    """
    anodes: list[_Slot] = []
    for position in range(1, ANODES_PER_BOARD + 1):
        rena, ch = electrode_map.channel_for_strip_position(board, position)
        label = electrode_map.electrode_label(board, rena, ch)
        anodes.append(_Slot(ChannelAddress(node, board, rena, ch), label, position))

    cathodes: list[_Slot] = []
    for index in range(1, CATHODES_PER_BOARD + 1):
        rena, ch = electrode_map.channel_for_cathode_label(board, index)
        label = electrode_map.electrode_label(board, rena, ch)
        cathodes.append(_Slot(ChannelAddress(node, board, rena, ch), label, index))

    members = frozenset(slot.channel for slot in anodes + cathodes)
    return _BoardLayout(anodes=tuple(anodes), cathodes=tuple(cathodes), members=members)


class _Seed(NamedTuple):
    """Everything about a cell except its fill (which depends on the colour mode)."""

    channel: ChannelAddress
    label: str
    position: int
    kind: str
    category: str
    view: ChannelView | None
    tooltip: str | None  # a tooltip already built, carried over


@dataclass(frozen=True)
class _BoardSeed:
    node: int
    board: int
    has_data: bool
    anodes: list[_Seed]
    cathodes: list[_Seed]

    def seeds(self, kind: str) -> list[_Seed]:
        return self.anodes if kind == "anode" else self.cathodes


@dataclass
class _Placement:
    """Where the keys of the views and data channels land on the grid."""

    data_boards: set[tuple[int, int]]
    outside: set[tuple[int, int]]
    unplaced: set[ChannelAddress]


def _place(
    keys: Iterable[ChannelTuple],
    electrode_map: ElectrodeMap,
    placement: _Placement,
    record_unplaced: bool,
) -> None:
    """Sort ``keys`` into boards with data, out-of-grid boards and unplaced keys."""
    for key in keys:
        node, board = key[0], key[1]
        if not is_in_grid(node, board):
            placement.outside.add((node, board))
        elif key in _board_layout(electrode_map, node, board).members:
            placement.data_boards.add((node, board))
        elif record_unplaced:
            placement.unplaced.add(as_address(key))


def _check_view_kind(channel: ChannelTuple, kind: str, view: ChannelView | None) -> None:
    """Refuse a cathode status on an anode or an anode status on a cathode."""
    if view is not None and view.is_cathode != (kind == "cathode"):
        raise ValueError(f"{kind} {tuple(channel)} has a view with status {view.status!r}")


def _seed_board(
    layout: _BoardLayout,
    node: int,
    board: int,
    views: Mapping[ChannelTuple, ChannelView],
    board_has_data: bool,
    data_channels: frozenset[ChannelTuple] | None,
    informational_flags: frozenset[str],
) -> _BoardSeed:
    """Classify one board's cells (everything but the fills)."""
    per_kind: list[list[_Seed]] = []
    for kind in KINDS:
        seeds: list[_Seed] = []
        for slot in layout.slots(kind):
            channel = slot.channel
            view = views.get(channel) if board_has_data else None
            _check_view_kind(channel, kind, view)
            has_data = board_has_data and (
                view is not None or data_channels is None or channel in data_channels
            )
            category = classify_view(view, has_data, informational_flags)
            seeds.append(_Seed(channel, slot.label, slot.position, kind, category, view, None))
        per_kind.append(seeds)
    return _BoardSeed(node, board, board_has_data, per_kind[0], per_kind[1])


def _reseed(cell: MapCell, category: str | None = None) -> _Seed:
    """Seed from an existing cell, keeping its view and any built tooltip."""
    return _Seed(
        cell.channel,
        cell.label,
        cell.position,
        cell.kind,
        cell.category if category is None else category,
        cell.view,
        cell._tooltip,
    )


def _assemble(
    board_seeds: list[_BoardSeed],
    *,
    unmapped: tuple[tuple[int, int], ...],
    unplaced: tuple[ChannelAddress, ...],
    color_mode: str,
    informational_flags: frozenset[str],
    data_channels_known: bool,
) -> SystemMapModel:
    """Colour the seeds for ``color_mode`` and build the model.

    In a metric mode the limits of each kind come from the finite values of
    that kind over every board with data; channels without events keep the
    no-data fill.
    """
    no_data_fill = category_qcolor(CATEGORY_NO_DATA)
    limits: dict[str, tuple[float, float] | None] = {}
    cells_by_kind: dict[str, Iterator[MapCell]] = {}
    for kind in KINDS:
        flat = [seed for board_seed in board_seeds for seed in board_seed.seeds(kind)]
        if color_mode == MODE_STATUS or kind == "cathode":
            cells = [
                MapCell(
                    s.channel,
                    s.label,
                    s.position,
                    s.kind,
                    s.category,
                    category_qcolor(s.category),
                    s.view,
                    _NAN,
                    s.tooltip,
                )
                for s in flat
            ]
        else:
            values = [
                (
                    s.view.metric(color_mode)
                    if s.view is not None and s.category != CATEGORY_NO_DATA
                    else _NAN
                )
                for s in flat
            ]
            kind_limits = metric_limits(values)
            limits[kind] = kind_limits
            fills = metric_colors(values, kind_limits)
            cells = [
                MapCell(
                    s.channel,
                    s.label,
                    s.position,
                    s.kind,
                    s.category,
                    no_data_fill if s.category == CATEGORY_NO_DATA else fill,
                    s.view,
                    value,
                    s.tooltip,
                )
                for s, fill, value in zip(flat, fills, values)
            ]
        cells_by_kind[kind] = iter(cells)

    anode_cells, cathode_cells = cells_by_kind["anode"], cells_by_kind["cathode"]
    boards: dict[tuple[int, int], BoardCells] = {}
    for board_seed in board_seeds:
        boards[(board_seed.node, board_seed.board)] = BoardCells(
            node=board_seed.node,
            board=board_seed.board,
            anodes=tuple(next(anode_cells) for _ in board_seed.anodes),
            cathodes=tuple(next(cathode_cells) for _ in board_seed.cathodes),
            has_data=board_seed.has_data,
        )
    return SystemMapModel(
        boards=boards,
        unmapped_boards=unmapped,
        color_mode=color_mode,
        informational_flags=informational_flags,
        limits=limits,
        unplaced_channels=unplaced,
        data_channels_known=data_channels_known,
    )


def build_system_map(
    views: Mapping[ChannelKeyT, ChannelView],
    active_boards: Iterable[tuple[int, int]] | None = None,
    data_channels: Iterable[ChannelTuple] | None = None,
    *,
    color_mode: str = MODE_STATUS,
    informational_flags: Iterable[str] = (),
    electrode_map: ElectrodeMap | None = None,
) -> SystemMapModel:
    """Build the map model for the whole grid.

    Args:
        views: One view per channel with a result, keyed by
            ``(node, board, rena, channel)`` (plain tuples,
            :class:`ChannelAddress` or another 4-int ``NamedTuple``). Keys on
            grid boards that are not electrodes are recorded in
            ``unplaced_channels`` and otherwise ignored.
        active_boards: ``(node, board)`` pairs with events (the cache's
            boards). Boards with a placed view or data channel count as well.
            When ``None``, only those do.
        data_channels: Channels with events (e.g. from the cache's per-board
            channel counts). A channel on a board with data that has neither
            a view nor an entry here is ``no data``. When ``None``, every
            channel on a board with data counts as having events (so one
            without a view is ``not fitted``).
        color_mode: Colour mode of the fills (``map_colors.COLOR_MODES``).
        informational_flags: Flags that do not make an ``ok`` channel
            ``flagged``.
        electrode_map: Electrode map to lay boards out with; defaults to the
            packaged ``ElectrodeMap.default()``.

    Returns:
        A model with all 160 grid boards, the sorted out-of-grid boards that
        carry results or data, and the sorted unplaced view keys.

    Raises:
        ValueError: If ``color_mode`` is unknown, or a view's status does not
            fit its electrode (a cathode status on an anode or the reverse).
    """
    check_color_mode(color_mode)
    emap = electrode_map if electrode_map is not None else default_electrode_map()
    informational = frozenset(informational_flags)
    channel_set = None if data_channels is None else frozenset(data_channels)
    # Looked up with ChannelAddress keys, which hash and compare like the
    # caller's 4-tuples.
    lookup = cast(Mapping[ChannelTuple, ChannelView], views)

    placement = _Placement(set(), set(), set())
    _place(lookup, emap, placement, record_unplaced=True)
    if channel_set is not None:
        _place(channel_set, emap, placement, record_unplaced=False)
    for node, board in active_boards if active_boards is not None else ():
        if is_in_grid(node, board):
            placement.data_boards.add((node, board))
        else:
            placement.outside.add((node, board))

    board_seeds = [
        _seed_board(
            _board_layout(emap, node, board),
            node,
            board,
            lookup,
            (node, board) in placement.data_boards,
            channel_set,
            informational,
        )
        for node in GRID_NODES
        for board in GRID_BOARDS
    ]
    return _assemble(
        board_seeds,
        unmapped=tuple(sorted(placement.outside)),
        unplaced=tuple(sorted(placement.unplaced)),
        color_mode=color_mode,
        informational_flags=informational,
        data_channels_known=channel_set is not None,
    )


def recolor_system_map(
    model: SystemMapModel,
    color_mode: str | None = None,
    informational_flags: Iterable[str] | None = None,
) -> SystemMapModel:
    """Return ``model`` in another colour mode or with other informational flags.

    Reuses every cell's view and any tooltip already built, so nothing is
    re-read or re-formatted: only the categories of cells with a view (when
    the informational flags change) and the fills are recomputed.

    Args:
        model: Model to recolour.
        color_mode: New colour mode; ``None`` keeps the model's.
        informational_flags: New informational flags; ``None`` keeps the
            model's.

    Returns:
        A new model (``model`` is not modified).

    Raises:
        ValueError: If ``color_mode`` is unknown.
    """
    mode = model.color_mode if color_mode is None else color_mode
    check_color_mode(mode)
    informational = (
        model.informational_flags if informational_flags is None else frozenset(informational_flags)
    )
    reclassify = informational != model.informational_flags

    def reseed(cell: MapCell) -> _Seed:
        if reclassify and cell.view is not None:
            return _reseed(cell, classify_view(cell.view, True, informational))
        return _reseed(cell)

    board_seeds = [
        _BoardSeed(
            board.node,
            board.board,
            board.has_data,
            [reseed(cell) for cell in board.anodes],
            [reseed(cell) for cell in board.cathodes],
        )
        for board in model.boards.values()
    ]
    return _assemble(
        board_seeds,
        unmapped=model.unmapped_boards,
        unplaced=model.unplaced_channels,
        color_mode=mode,
        informational_flags=informational,
        data_channels_known=model.data_channels_known,
    )


def update_system_map(
    model: SystemMapModel,
    changed: Mapping[ChannelKeyT, ChannelView],
    *,
    electrode_map: ElectrodeMap | None = None,
) -> SystemMapModel:
    """Return ``model`` with the views in ``changed`` added or replaced.

    For single-channel and board re-fits. Only the boards that hold a changed
    channel are re-classified; every other cell keeps its view and built
    tooltip, and the fills (and metric limits) are recomputed for the whole
    grid. The result equals a full :func:`build_system_map` over the merged
    views with the same boards and data channels.

    A board without data that receives a view gains data. Its other channels
    become ``not fitted``, unless the model was built with a list of data
    channels (which then did not list them, so they stay ``no data``).

    Args:
        model: Model to update.
        changed: New views, keyed like :func:`build_system_map`'s ``views``.
        electrode_map: Electrode map the model was built with; defaults to
            the packaged one.

    Returns:
        A new model (``model`` is not modified).

    Raises:
        ValueError: If a view's status does not fit its electrode.
    """
    emap = electrode_map if electrode_map is not None else default_electrode_map()
    lookup = cast(Mapping[ChannelTuple, ChannelView], changed)
    placement = _Placement(set(), set(model.unmapped_boards), set(model.unplaced_channels))
    by_board: dict[tuple[int, int], dict[ChannelTuple, ChannelView]] = {}
    for key, view in lookup.items():
        node, board = key[0], key[1]
        if not is_in_grid(node, board):
            placement.outside.add((node, board))
        elif key in _board_layout(emap, node, board).members:
            by_board.setdefault((node, board), {})[key] = view
        else:
            placement.unplaced.add(as_address(key))

    informational = model.informational_flags
    board_seeds: list[_BoardSeed] = []
    for board_key, board_cells in model.boards.items():
        updates = by_board.get(board_key)
        if not updates:
            board_seeds.append(
                _BoardSeed(
                    board_cells.node,
                    board_cells.board,
                    board_cells.has_data,
                    [_reseed(cell) for cell in board_cells.anodes],
                    [_reseed(cell) for cell in board_cells.cathodes],
                )
            )
            continue
        gains_data = not board_cells.has_data
        per_kind: list[list[_Seed]] = []
        for kind in KINDS:
            seeds: list[_Seed] = []
            for cell in board_cells.cells(kind):
                new_view = updates.get(cell.channel)
                if new_view is not None:
                    _check_view_kind(cell.channel, kind, new_view)
                    category = classify_view(new_view, True, informational)
                    seeds.append(
                        _Seed(
                            cell.channel, cell.label, cell.position, kind, category, new_view, None
                        )
                    )
                elif gains_data:
                    category = classify_view(None, not model.data_channels_known)
                    seeds.append(
                        _Seed(cell.channel, cell.label, cell.position, kind, category, None, None)
                    )
                else:
                    seeds.append(_reseed(cell))
            per_kind.append(seeds)
        board_seeds.append(
            _BoardSeed(board_cells.node, board_cells.board, True, per_kind[0], per_kind[1])
        )
    return _assemble(
        board_seeds,
        unmapped=tuple(sorted(placement.outside)),
        unplaced=tuple(sorted(placement.unplaced)),
        color_mode=model.color_mode,
        informational_flags=informational,
        data_channels_known=model.data_channels_known,
    )


def status_counts(model: SystemMapModel, kind: str) -> dict[str, int]:
    """Count cells per status category over boards that have data.

    Args:
        model: Map model.
        kind: ``"anode"`` or ``"cathode"``.

    Returns:
        A dict with an entry for every category of the kind (zero when
        absent), in ``map_colors.ANODE_CATEGORIES`` / ``CATHODE_CATEGORIES``
        order.
    """
    _check_kind(kind)
    counts = dict.fromkeys(ANODE_CATEGORIES if kind == "anode" else CATHODE_CATEGORIES, 0)
    for board in model.boards.values():
        if not board.has_data:
            continue
        for cell in board.cells(kind):
            counts[cell.category] += 1
    return counts


def metric_values(model: SystemMapModel, kind: str) -> npt.NDArray[np.float64]:
    """Return the finite metric values of ``kind`` over boards with data.

    Empty in the status mode.
    """
    _check_kind(kind)
    values = [
        cell.value
        for board in model.boards.values()
        if board.has_data
        for cell in board.cells(kind)
        if math.isfinite(cell.value)
    ]
    return np.asarray(values, dtype=np.float64)


# Summary words per category; non-breaking spaces keep "63 too few events"
# on one line when the summary label wraps.
_SUMMARY_WORDS: dict[str, str] = {
    category: CATEGORY_LABELS[category].lower().replace(" ", _NBSP)
    for category in (*ANODE_CATEGORIES, *CATHODE_CATEGORIES)
}
_SUMMARY_WORDS[CATEGORY_CATHODE_CALIBRATED] = "calibrated"
_SUMMARY_WORDS["cathode_uncalibrated"] = "uncalibrated"


def _count(n: int, singular: str, plural: str) -> str:
    """``"1 anode"`` / ``"5 anodes"``, joined by a non-breaking space."""
    return f"{n}{_NBSP}{singular if n == 1 else plural}"


def format_summary(model: SystemMapModel) -> str:
    """Render the one-line header summary for the map.

    Returns :data:`NO_DATA_SUMMARY` when no board has data. In the status
    mode it counts the categories (zero counts are left out)::

        Anodes: 2668 corrected, 120 flagged, 672 no gain, ... | Cathodes: 692 calibrated, ...

    In a metric mode it gives the median and the colour range of the anodes::

        CV gain | Anodes: median 4.27%, colour range 1.2% to 25% (3340 anodes)
    """
    if not any(board.has_data for board in model.boards.values()):
        return NO_DATA_SUMMARY

    if model.color_mode == MODE_STATUS:
        parts: list[str] = []
        for kind, title in (("anode", "Anodes"), ("cathode", "Cathodes")):
            counts = status_counts(model, kind)
            listed = ", ".join(f"{n}{_NBSP}{_SUMMARY_WORDS[c]}" for c, n in counts.items() if n)
            parts.append(f"{title}: {listed or 'none'}")
        return " | ".join(parts)

    spec = METRIC_SPECS[model.color_mode]
    values = metric_values(model, "anode")
    limits = model.limits_for("anode")
    if values.size == 0 or limits is None:
        return f"{spec.label} | Anodes: no values"
    return (
        f"{spec.label} | Anodes: median {spec.format(float(np.median(values)))}, colour range "
        f"{spec.format(limits[0])} to {spec.format(limits[1])} "
        f"({_count(int(values.size), 'anode', 'anodes')})"
    )


def format_board_summary(board: BoardCells) -> str:
    """Render the header line of the magnified board strip.

    Example::

        Node 3 Board 17 (odd): anodes 20/39 corrected (2 flagged, 1 rejected), cathodes 6/8 calibrated

    ``" - no data"`` is appended when the board has no data.
    """
    parity = "even" if is_even_board(board.board) else "odd"
    corrected = sum(
        1
        for cell in board.anodes
        if cell.view is not None and cell.view.status == STATUS_OK and not cell.view.review
    )
    flagged = sum(1 for cell in board.anodes if cell.category == CATEGORY_FLAGGED)
    rejected = sum(1 for cell in board.anodes if cell.view is not None and cell.view.review)
    extras = [f"{flagged} flagged"] if flagged else []
    extras += [f"{rejected} rejected"] if rejected else []
    anodes = f"anodes {corrected}/{len(board.anodes)} corrected"
    if extras:
        anodes += f" ({', '.join(extras)})"
    calibrated = sum(1 for cell in board.cathodes if cell.category == CATEGORY_CATHODE_CALIBRATED)
    text = (
        f"Node {board.node} Board {board.board} ({parity}): {anodes}, "
        f"cathodes {calibrated}/{len(board.cathodes)} calibrated"
    )
    if not board.has_data:
        text += " - no data"
    return text
