"""Tests for the System Map model and colour logic (no widgets, no display needed).

Covers ``dcalib.gui._system_map_model`` (layout, categories, tooltips,
summaries, recolouring) and the pure parts of ``dcalib.gui.map_colors``
(status categories, metric limits and colours, legend entries).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from adc2kev.tools.electrode_map import ElectrodeMap
from adc2kev.tools.geometry import ANODES_PER_BOARD, CATHODES_PER_BOARD
from PyQt6.QtGui import QColor

from dcalib.channels import anode_channels, cathode_channels
from dcalib.gui._system_map_model import (
    GRID_BOARDS,
    GRID_NODES,
    NO_DATA_SUMMARY,
    TOOLTIP_METRICS,
    ChannelAddress,
    ChannelTuple,
    ChannelView,
    SystemMapModel,
    build_cell_tooltip,
    build_system_map,
    classify_view,
    format_board_summary,
    format_summary,
    is_in_grid,
    metric_values,
    recolor_system_map,
    status_counts,
    update_system_map,
)
from dcalib.gui.map_colors import (
    ANODE_CATEGORIES,
    CATEGORIES,
    CATEGORY_CATHODE_CALIBRATED,
    CATEGORY_CATHODE_UNCALIBRATED,
    CATEGORY_COLORS,
    CATEGORY_FAILED,
    CATEGORY_FLAGGED,
    CATEGORY_LABELS,
    CATEGORY_NO_DATA,
    CATEGORY_NOT_FITTED,
    CATEGORY_OK,
    CATEGORY_TOO_FEW,
    CATHODE_CALIBRATED,
    CATHODE_UNCALIBRATED,
    COLOR_MODES,
    LUT_SIZE,
    METRIC_SPECS,
    MODE_STATUS,
    NO_DATA_COLOR,
    NOT_FITTED_COLOR,
    category_qcolor,
    check_color_mode,
    clip_range_text,
    is_metric_mode,
    label_colors_for_fills,
    legend_entries,
    lut_indices,
    metric_color,
    metric_colors,
    metric_limits,
    sequential_lut,
    status_category,
    text_color_for,
    text_color_for_fills,
)
from dcalib.options import (
    FLAG_CONVEX_CURVE,
    FLAG_EXTRAPOLATION_RISK,
    FLAG_SOURCE_INCONSISTENT,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    STATUSES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMAP = ElectrodeMap.default()
ACTIVE_CHANNELS = tuple(sorted(anode_channels(16) + cathode_channels(16)))
NODE, BOARD = 3, 17  # odd board, panel 1
EVEN_BOARD = 18

_LAYOUT: list[SystemMapModel] = []


def _layout() -> SystemMapModel:
    """Static layout of every grid board (no views), built on first use."""
    if not _LAYOUT:
        _LAYOUT.append(build_system_map({}))
    return _LAYOUT[0]


def _anode(node: int, board: int, position: int) -> ChannelAddress:
    return _layout().boards[(node, board)].anodes[position - 1].channel


def _cathode(node: int, board: int, index: int) -> ChannelAddress:
    return _layout().boards[(node, board)].cathodes[index - 1].channel


def _ok(**metrics: float) -> ChannelView:
    return ChannelView(STATUS_OK, metrics={"n_events": 1000.0, **metrics})


def _qc(hex_color: str) -> QColor:
    return QColor(hex_color)


NBSP = "\u00a0"


def _cell_state(model: SystemMapModel) -> list[tuple[object, ...]]:
    """Everything drawn about each cell (NaN values as None, so they compare equal)."""
    return [
        (
            board.has_data,
            cell.channel,
            cell.kind,
            cell.category,
            cell.fill.rgb(),
            None if math.isnan(cell.value) else cell.value,
            cell.view,
            cell.tooltip,
        )
        for board in model.boards.values()
        for cell in board.anodes + board.cathodes
    ]


# ---------------------------------------------------------------------------
# map_colors: status categories
# ---------------------------------------------------------------------------


class TestStatusCategory:
    """status -> category rules, including the informational flags."""

    @pytest.mark.parametrize(
        ("status", "flags", "informational", "has_data", "expected"),
        [
            (None, (), (), True, CATEGORY_NOT_FITTED),
            (None, (), (), False, CATEGORY_NO_DATA),
            (STATUS_OK, (), (), False, CATEGORY_NO_DATA),
            (STATUS_OK, (), (), True, CATEGORY_OK),
            (STATUS_OK, (FLAG_CONVEX_CURVE,), (), True, CATEGORY_FLAGGED),
            (STATUS_OK, (FLAG_CONVEX_CURVE,), (FLAG_CONVEX_CURVE,), True, CATEGORY_OK),
            (
                STATUS_OK,
                (FLAG_CONVEX_CURVE, FLAG_SOURCE_INCONSISTENT),
                (FLAG_CONVEX_CURVE,),
                True,
                CATEGORY_FLAGGED,
            ),
            (STATUS_FIT_FAILED, (), (), True, CATEGORY_FAILED),
            (STATUS_FIT_FAILED, (FLAG_SOURCE_INCONSISTENT,), (), True, CATEGORY_FAILED),
            (STATUS_TOO_FEW_EVENTS, (), (), True, CATEGORY_TOO_FEW),
        ],
    )
    def test_rules(
        self,
        status: str | None,
        flags: tuple[str, ...],
        informational: tuple[str, ...],
        has_data: bool,
        expected: str,
    ) -> None:
        assert status_category(status, flags, frozenset(informational), has_data=has_data) == (
            expected
        )

    def test_unknown_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="status"):
            status_category("maybe")

    def test_category_colours(self) -> None:
        assert set(CATEGORY_COLORS) == set(CATEGORIES) == set(CATEGORY_LABELS)
        for category in CATEGORIES:
            assert category_qcolor(category) == _qc(CATEGORY_COLORS[category])
        # Distinct fills, and the no-data / not-fitted fills are the exported ones.
        assert len(set(CATEGORY_COLORS.values())) == len(CATEGORIES)
        assert CATEGORY_COLORS[CATEGORY_NO_DATA] == NO_DATA_COLOR
        assert CATEGORY_COLORS[CATEGORY_NOT_FITTED] == NOT_FITTED_COLOR
        with pytest.raises(ValueError, match="category"):
            category_qcolor("purple")


class TestColorModes:
    def test_modes(self) -> None:
        assert COLOR_MODES[0] == MODE_STATUS
        assert COLOR_MODES[1:] == (
            "cv_gain",
            "peak_spread",
            "fwhm_511_after",
            "fwhm_662_after",
            "fwhm_511_change",
            "ge_cs_max_diff",
        )
        assert not is_metric_mode(MODE_STATUS)
        assert all(is_metric_mode(mode) for mode in METRIC_SPECS)
        with pytest.raises(ValueError, match="color mode"):
            check_color_mode("rainbow")
        with pytest.raises(ValueError, match="color mode"):
            is_metric_mode("rainbow")

    def test_legend_entries(self) -> None:
        status = legend_entries(MODE_STATUS)
        assert [label for label, _ in status] == [CATEGORY_LABELS[c] for c in CATEGORIES]
        assert [color for _, color in status] == [CATEGORY_COLORS[c] for c in CATEGORIES]
        for mode in METRIC_SPECS:
            assert legend_entries(mode) == (
                ("No value", NOT_FITTED_COLOR),
                ("No data", NO_DATA_COLOR),
                ("Cathode calibrated", CATEGORY_COLORS[CATEGORY_CATHODE_CALIBRATED]),
                ("Cathode uncalibrated", CATEGORY_COLORS[CATEGORY_CATHODE_UNCALIBRATED]),
            )

    def test_metric_spec_format(self) -> None:
        assert METRIC_SPECS["ge_cs_max_diff"].format(0.01234) == "1.23%"
        assert METRIC_SPECS["cv_gain"].format(0.045678) == "4.57%"
        assert METRIC_SPECS["fwhm_511_change"].format(-0.0512) == "-5.12%"
        assert METRIC_SPECS["fwhm_511_after"].format(5.8765) == "5.88 %"
        assert METRIC_SPECS["peak_spread"].format(math.nan) == "n/a"


# ---------------------------------------------------------------------------
# map_colors: metric colours
# ---------------------------------------------------------------------------


class TestMetricColours:
    """Percentile limits, LUT indices and fills, including NaN and equal values."""

    def test_lut_is_viridis(self) -> None:
        lut = sequential_lut()
        assert len(lut) == LUT_SIZE
        assert lut[0].getRgb()[:3] == (68, 1, 84)
        assert lut[-1].getRgb()[:3] == (253, 231, 37)
        assert sequential_lut() is lut  # cached
        with pytest.raises(ValueError, match="size"):
            sequential_lut(size=1)

    def test_limits_are_clipped_percentiles(self) -> None:
        values = np.arange(101, dtype=float)  # 0..100: percentiles equal the values
        assert metric_limits(values) == pytest.approx((2.0, 98.0))
        assert metric_limits(values, (10.0, 90.0)) == pytest.approx((10.0, 90.0))
        # One huge outlier moves the upper limit only a little.
        with_outlier = np.append(np.linspace(1.0, 2.0, 99), 1e6)
        low, high = metric_limits(with_outlier) or (math.nan, math.nan)
        assert 1.0 <= low < high < 2.1

    def test_limits_ignore_non_finite(self) -> None:
        assert metric_limits([math.nan, 1.0, math.inf, 3.0, -math.inf], (0.0, 100.0)) == (1.0, 3.0)
        assert metric_limits([math.nan, math.nan]) is None
        assert metric_limits([]) is None

    def test_limits_all_equal(self) -> None:
        assert metric_limits([5.0, 5.0, math.nan, 5.0]) == (5.0, 5.0)

    def test_limits_invalid_percentiles(self) -> None:
        with pytest.raises(ValueError, match="percentiles"):
            metric_limits([1.0], (50.0, 10.0))
        with pytest.raises(ValueError, match="percentiles"):
            metric_limits([1.0], (-1.0, 10.0))

    def test_lut_indices(self) -> None:
        indices = lut_indices([0.0, 10.0, 5.0, -3.0, 99.0, math.nan], (0.0, 10.0))
        assert indices.tolist() == [0, LUT_SIZE - 1, 128, 0, LUT_SIZE - 1, -1]
        assert lut_indices([1.0, 2.0], None).tolist() == [-1, -1]
        # Equal limits: every finite value maps to the middle.
        assert lut_indices([4.0, 4.0, math.nan], (4.0, 4.0)).tolist() == [127, 127, -1]

    def test_metric_colours(self) -> None:
        lut = sequential_lut()
        colors = metric_colors([0.0, 10.0, math.nan, 100.0], (0.0, 10.0))
        assert colors[0] == lut[0]
        assert colors[1] == lut[-1]
        assert colors[2] == _qc(NOT_FITTED_COLOR)
        assert colors[3] == lut[-1]  # clipped
        assert metric_color(math.nan, (0.0, 1.0)) == _qc(NOT_FITTED_COLOR)
        assert metric_color(1.0, None) == _qc(NOT_FITTED_COLOR)

    def test_text_colour_contrasts_with_fill(self) -> None:
        lut = sequential_lut()
        dark, light = text_color_for(lut[-1]), text_color_for(lut[0])
        assert dark.lightness() < 64 < 192 < light.lightness()
        assert text_color_for(_qc(CATEGORY_COLORS[CATEGORY_OK])) == dark


# ---------------------------------------------------------------------------
# ChannelView and ChannelAddress
# ---------------------------------------------------------------------------


class TestChannelView:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="status"):
            ChannelView("excellent")
        with pytest.raises(ValueError, match="options_source"):
            ChannelView(STATUS_OK, options_source="manual")
        for status in STATUSES:
            assert ChannelView(status).status == status

    def test_flags_tuple_and_metrics(self) -> None:
        view = ChannelView(STATUS_OK, flags=[FLAG_SOURCE_INCONSISTENT], metrics={"peak_spread": 0.02})  # type: ignore[arg-type]
        assert view.flags == (FLAG_SOURCE_INCONSISTENT,)
        assert view.metric("peak_spread") == 0.02
        assert math.isnan(view.metric("cv_gain"))
        assert math.isnan(ChannelView(STATUS_OK, metrics={"x": None}).metric("x"))  # type: ignore[dict-item]

    def test_address_is_a_tuple(self) -> None:
        address = ChannelAddress(1, 15, 0, 4)
        assert address == (1, 15, 0, 4)
        assert {(1, 15, 0, 4): "x"}[address] == "x"
        assert (address.node, address.board, address.rena, address.channel) == (1, 15, 0, 4)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    """Grid coverage and physical strip order from ElectrodeMap."""

    def test_grid(self) -> None:
        model = _layout()
        assert tuple(range(1, 11)) == GRID_NODES
        assert tuple(range(15, 31)) == GRID_BOARDS
        assert list(model.boards) == [(n, b) for n in GRID_NODES for b in GRID_BOARDS]
        assert is_in_grid(1, 15) and is_in_grid(10, 30)
        assert not is_in_grid(0, 15) and not is_in_grid(1, 14) and not is_in_grid(11, 20)

    @pytest.mark.parametrize("board", GRID_BOARDS)
    def test_strip_order_matches_electrode_map(self, board: int) -> None:
        cells = _layout().boards[(NODE, board)]
        assert len(cells.anodes) == ANODES_PER_BOARD
        assert len(cells.cathodes) == CATHODES_PER_BOARD
        for position, cell in enumerate(cells.anodes, start=1):
            rena, ch = EMAP.channel_for_strip_position(board, position)
            assert cell.channel == (NODE, board, rena, ch)
            assert isinstance(cell.channel, ChannelAddress)
            assert cell.position == position
            assert cell.label == EMAP.electrode_label(board, rena, ch)
            assert EMAP.physical_strip_position(board, rena, ch) == position
        for index, cell in enumerate(cells.cathodes, start=1):
            assert cell.label == f"C{index:02d}"
            assert cell.channel == (NODE, board, *EMAP.channel_for_cathode_label(board, index))
        # Every active channel appears exactly once.
        channels = {(c.channel.rena, c.channel.channel) for c in cells.anodes + cells.cathodes}
        assert channels == set(ACTIVE_CHANNELS)

    def test_parity_flip(self) -> None:
        assert _layout().boards[(NODE, EVEN_BOARD)].anodes[0].label == "A01"
        assert _layout().boards[(NODE, BOARD)].anodes[0].label == "A39"

    def test_locate_and_cell(self) -> None:
        model = _layout()
        cathode = _cathode(NODE, BOARD, 3)
        located = model.locate(cathode)
        assert located is not None
        board, kind, position = located
        assert (board.node, board.board, kind, position) == (NODE, BOARD, "cathode", 3)
        assert model.cell(tuple(cathode)) is board.cathodes[2]  # type: ignore[arg-type]
        assert model.locate((NODE, 40, 0, 4)) is None  # board outside the grid
        assert model.locate((NODE, BOARD, 0, 1)) is None  # not an electrode
        assert model.cell((NODE, BOARD, 0, 1)) is None


# ---------------------------------------------------------------------------
# build_system_map: categories and data
# ---------------------------------------------------------------------------


def _mixed_views() -> dict[ChannelTuple, ChannelView]:
    """Anodes 1-6 of (3, 17): ok, flagged, failed, too few, n_events 0, override."""
    return {
        _anode(NODE, BOARD, 1): _ok(cv_gain=10.0),
        _anode(NODE, BOARD, 2): ChannelView(
            STATUS_OK, (FLAG_CONVEX_CURVE,), metrics={"cv_gain": 20.0}
        ),
        _anode(NODE, BOARD, 3): ChannelView(STATUS_FIT_FAILED, metrics={"n_events": 500.0}),
        _anode(NODE, BOARD, 4): ChannelView(STATUS_TOO_FEW_EVENTS, metrics={"n_events": 12.0}),
        _anode(NODE, BOARD, 5): ChannelView(STATUS_TOO_FEW_EVENTS, metrics={"n_events": 0.0}),
        _anode(NODE, BOARD, 6): ChannelView(
            STATUS_OK, options_source="override", metrics={"cv_gain": 30.0}
        ),
    }


class TestBuild:
    def test_categories_of_a_mixed_board(self) -> None:
        views = _mixed_views()
        model = build_system_map(views)
        anodes = model.boards[(NODE, BOARD)].anodes
        assert [cell.category for cell in anodes[:7]] == [
            CATEGORY_OK,
            CATEGORY_FLAGGED,
            CATEGORY_FAILED,
            CATEGORY_TOO_FEW,
            CATEGORY_NO_DATA,
            CATEGORY_OK,
            CATEGORY_NOT_FITTED,
        ]
        for cell in anodes[:7]:
            assert cell.fill == category_qcolor(cell.category)
            assert math.isnan(cell.value)
        assert anodes[0].view is views[_anode(NODE, BOARD, 1)]
        assert anodes[6].view is None
        assert model.limits == {}

    def test_informational_flags(self) -> None:
        model = build_system_map(_mixed_views(), informational_flags=[FLAG_CONVEX_CURVE])
        assert model.informational_flags == frozenset({FLAG_CONVEX_CURVE})
        assert model.boards[(NODE, BOARD)].anodes[1].category == CATEGORY_OK

    def test_classify_view(self) -> None:
        assert classify_view(None, has_data=True) == CATEGORY_NOT_FITTED
        assert classify_view(None, has_data=False) == CATEGORY_NO_DATA
        # A view always counts as data (the channel has a result).
        assert classify_view(_ok(), has_data=False) == CATEGORY_OK
        zero = ChannelView(STATUS_TOO_FEW_EVENTS, metrics={"n_events": 0})
        assert classify_view(zero) == CATEGORY_NO_DATA
        flagged = ChannelView(STATUS_OK, (FLAG_EXTRAPOLATION_RISK,))
        assert classify_view(flagged) == CATEGORY_FLAGGED
        assert classify_view(flagged, True, {FLAG_EXTRAPOLATION_RISK}) == CATEGORY_OK

    def test_boards_inferred_from_views(self) -> None:
        model = build_system_map(_mixed_views())
        assert [nb for nb, b in model.boards.items() if b.has_data] == [(NODE, BOARD)]
        # Boards without data: every cell is no data.
        other = model.boards[(1, 15)]
        assert not other.has_data
        assert {cell.category for cell in other.anodes + other.cathodes} == {CATEGORY_NO_DATA}

    def test_board_with_data_but_no_views_is_not_fitted(self) -> None:
        model = build_system_map({}, active_boards=[(5, 20), (9, 30)])
        for key in ((5, 20), (9, 30)):
            board = model.boards[key]
            assert board.has_data
            assert {c.category for c in board.anodes + board.cathodes} == {CATEGORY_NOT_FITTED}
            assert all(c.fill == _qc(NOT_FITTED_COLOR) for c in board.anodes)
        assert not model.boards[(5, 21)].has_data

    def test_data_channels_separate_no_data_from_not_fitted(self) -> None:
        with_events = _anode(NODE, BOARD, 10)
        views = {_anode(NODE, BOARD, 1): _ok()}
        model = build_system_map(views, active_boards=[(NODE, BOARD)], data_channels=[with_events])
        anodes = model.boards[(NODE, BOARD)].anodes
        assert anodes[0].category == CATEGORY_OK  # a view wins over data_channels
        assert anodes[9].category == CATEGORY_NOT_FITTED
        assert anodes[1].category == CATEGORY_NO_DATA
        assert anodes[1].fill == _qc(NO_DATA_COLOR)
        assert "Status: no data" in anodes[1].tooltip
        assert "Status: not fitted" in anodes[9].tooltip

    def test_data_channels_mark_their_board(self) -> None:
        model = build_system_map({}, data_channels=[_cathode(7, 22, 1)])
        assert model.boards[(7, 22)].has_data
        assert model.boards[(7, 22)].cathodes[0].category == CATEGORY_NOT_FITTED
        assert model.boards[(7, 22)].cathodes[1].category == CATEGORY_NO_DATA

    def test_unmapped_boards(self) -> None:
        views = {(1, 14, 0, 4): _ok(), (11, 20, 1, 9): _ok(), _anode(NODE, BOARD, 1): _ok()}
        model = build_system_map(
            views, active_boards=[(0, 16), (2, 18)], data_channels=[(4, 31, 0, 5)]
        )
        assert model.unmapped_boards == ((0, 16), (1, 14), (4, 31), (11, 20))
        assert (1, 14) not in model.boards
        assert model.boards[(2, 18)].has_data

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="color mode"):
            build_system_map({}, color_mode="bogus")

    def test_custom_electrode_map(self) -> None:
        model = build_system_map({}, electrode_map=ElectrodeMap.from_packaged_files())
        assert model.boards[(NODE, BOARD)].anodes[0].label == "A39"


# ---------------------------------------------------------------------------
# Metric modes and recolouring
# ---------------------------------------------------------------------------


def _metric_views() -> dict[ChannelTuple, ChannelView]:
    """(3, 17) anodes 1..20 with cv_gain 1..20 %, anode 21 NaN, anode 22 no events;
    cathodes 1..3 calibrated, cathode 4 not."""
    views: dict[ChannelTuple, ChannelView] = {
        _anode(NODE, BOARD, p): _ok(cv_gain=p / 100) for p in range(1, 21)
    }
    views[_anode(NODE, BOARD, 21)] = ChannelView(STATUS_FIT_FAILED, metrics={"n_events": 50.0})
    views[_anode(NODE, BOARD, 22)] = ChannelView(
        STATUS_TOO_FEW_EVENTS, metrics={"n_events": 0.0, "cv_gain": 5.0}
    )
    for index in range(1, 4):
        views[_cathode(NODE, BOARD, index)] = ChannelView(CATHODE_CALIBRATED)
    views[_cathode(NODE, BOARD, 4)] = ChannelView(CATHODE_UNCALIBRATED)
    return views


class TestMetricMode:
    def test_fills_and_limits_per_kind(self) -> None:
        model = build_system_map(_metric_views(), color_mode="cv_gain")
        lut = sequential_lut()
        anode_limits = metric_limits([p / 100 for p in range(1, 21)])
        assert model.limits_for("anode") == anode_limits
        assert model.limits_for("cathode") is None  # cathodes are never metric-coloured

        anodes = model.boards[(NODE, BOARD)].anodes
        assert anodes[0].value == 0.01 and anodes[0].fill == lut[0]  # below the 2nd pct
        assert anodes[19].fill == lut[-1]  # above the 98th pct
        assert anodes[9].fill == metric_color(0.10, anode_limits)
        assert math.isnan(anodes[20].value)
        assert anodes[20].fill == _qc(NOT_FITTED_COLOR)  # failed: no value
        assert anodes[21].fill == _qc(NO_DATA_COLOR)  # no events, value ignored
        assert math.isnan(anodes[21].value)
        assert anodes[30].fill == _qc(NOT_FITTED_COLOR)  # has data, no view

        cathodes = model.boards[(NODE, BOARD)].cathodes
        assert cathodes[0].fill == _qc(CATEGORY_COLORS[CATEGORY_CATHODE_CALIBRATED])
        assert cathodes[3].fill == _qc(CATEGORY_COLORS[CATEGORY_CATHODE_UNCALIBRATED])
        assert all(math.isnan(cell.value) for cell in cathodes)
        assert cathodes[5].fill == _qc(NOT_FITTED_COLOR)
        # Boards without data keep the no-data fill.
        assert model.boards[(1, 15)].anodes[0].fill == _qc(NO_DATA_COLOR)

    def test_all_equal_values_use_the_middle_colour(self) -> None:
        views = {_anode(NODE, BOARD, p): _ok(peak_spread=0.02) for p in range(1, 5)}
        model = build_system_map(views, color_mode="peak_spread")
        assert model.limits_for("anode") == (0.02, 0.02)
        assert model.limits_for("cathode") is None
        middle = sequential_lut()[(LUT_SIZE - 1) // 2]
        assert {cell.fill.name() for cell in model.boards[(NODE, BOARD)].anodes[:4]} == {
            middle.name()
        }

    def test_recolor_reuses_tooltips_and_views(self) -> None:
        model = build_system_map(_mixed_views(), active_boards=[(1, 15)])
        built = [c.tooltip for b in model.boards.values() for c in b.anodes + b.cathodes]
        recolored = recolor_system_map(model, "cv_gain")
        assert recolored.color_mode == "cv_gain"
        assert model.color_mode == MODE_STATUS  # unchanged
        assert recolored.unmapped_boards == model.unmapped_boards
        for key, board in model.boards.items():
            other = recolored.boards[key]
            assert other.has_data == board.has_data
            for a, b in zip(board.anodes + board.cathodes, other.anodes + other.cathodes):
                assert b.tooltip is a.tooltip  # built before the recolour: carried over
                assert b.view is a.view
                assert b.category == a.category
                assert (b.channel, b.label, b.position) == (a.channel, a.label, a.position)
        anodes = recolored.boards[(NODE, BOARD)].anodes
        assert [c.value for c in anodes[:2]] == [10.0, 20.0]
        # Back to status: the original fills again.
        back = recolor_system_map(recolored, MODE_STATUS)
        assert [c.fill for c in back.boards[(NODE, BOARD)].anodes] == [
            c.fill for c in model.boards[(NODE, BOARD)].anodes
        ]
        assert back.limits == {}
        assert len(built) == 160 * (ANODES_PER_BOARD + CATHODES_PER_BOARD)

    def test_recolor_equals_rebuild_after_informational_change(self) -> None:
        """In a metric mode, recolouring with new informational flags is a full rebuild."""
        views = {**_metric_views(), **_mixed_views()}
        model = build_system_map(views, [(1, 15)], color_mode="cv_gain")
        recolored = recolor_system_map(model, informational_flags=[FLAG_CONVEX_CURVE])
        rebuilt = build_system_map(
            views, [(1, 15)], color_mode="cv_gain", informational_flags=[FLAG_CONVEX_CURVE]
        )
        assert _cell_state(recolored) == _cell_state(rebuilt)
        assert recolored.limits == rebuilt.limits
        assert _cell_state(recolored) != _cell_state(model)  # anode 2 is no longer flagged

    def test_recolor_informational_flags(self) -> None:
        model = build_system_map(_mixed_views())
        relaxed = recolor_system_map(model, informational_flags=[FLAG_CONVEX_CURVE])
        assert relaxed.color_mode == MODE_STATUS
        assert relaxed.boards[(NODE, BOARD)].anodes[1].category == CATEGORY_OK
        assert relaxed.boards[(NODE, BOARD)].anodes[1].fill == category_qcolor(CATEGORY_OK)
        assert model.boards[(NODE, BOARD)].anodes[1].category == CATEGORY_FLAGGED
        strict = recolor_system_map(relaxed, informational_flags=())
        assert strict.boards[(NODE, BOARD)].anodes[1].category == CATEGORY_FLAGGED
        with pytest.raises(ValueError, match="color mode"):
            recolor_system_map(model, "bogus")

    def test_full_system_model(self) -> None:
        """~7k views over 155 boards: every data cell classified, limits from all boards."""
        boards = [(n, b) for n in GRID_NODES for b in GRID_BOARDS][:155]
        rng = np.random.default_rng(3)
        views: dict[ChannelTuple, ChannelView] = {}
        for node, board in boards:
            for rena, ch in anode_channels(board):
                views[(node, board, rena, ch)] = _ok(fwhm_511_after=float(rng.uniform(5.0, 7.0)))
            for rena, ch in cathode_channels(board):
                views[(node, board, rena, ch)] = ChannelView(CATHODE_CALIBRATED)
        model = build_system_map(views, boards, color_mode="fwhm_511_after")
        counts = status_counts(model, "anode")
        assert counts[CATEGORY_OK] == 155 * ANODES_PER_BOARD
        cathodes = status_counts(model, "cathode")
        assert cathodes[CATEGORY_CATHODE_CALIBRATED] == 155 * CATHODES_PER_BOARD
        assert metric_values(model, "anode").size == 155 * ANODES_PER_BOARD
        assert metric_values(model, "cathode").size == 0
        low, high = model.limits_for("anode") or (math.nan, math.nan)
        assert 5.0 < low < 5.1 and 6.9 < high < 7.0


# ---------------------------------------------------------------------------
# Tooltips and summaries
# ---------------------------------------------------------------------------


class TestTooltip:
    def test_full_tooltip(self) -> None:
        channel = ChannelAddress(3, 17, 0, 12)
        view = ChannelView(
            STATUS_OK,
            (FLAG_CONVEX_CURVE, FLAG_SOURCE_INCONSISTENT),
            "override",
            {
                "n_events": 17646.0,
                "n_selected": 15012.0,
                "degree": 2.0,
                "cv_gain": 0.012345,
                "peak_spread": 0.0321,
                "fwhm_511_after": 5.876,
                "fwhm_662_after": 4.51,
                "fwhm_511_change": -0.0512,
                "ge_cs_max_diff": 0.005,
            },
            review="rejected",
        )
        text = build_cell_tooltip(channel, "A17", 23, "anode", view)
        assert text.splitlines() == [
            "A17 (strip position 23)",
            "Node 3  Board 17  RENA 0  Ch 12",
            "Status: ok",
            "Review: rejected",
            "Flags: convex_curve, source_inconsistent",
            "Options: override",
            "Events: 17646",
            "Selected: 15012",
            "Degree: 2",
            "CV gain: 1.23%",
            "Peak spread: 3.21%",
            "FWHM 511 keV: 5.88 %",
            "FWHM 662 keV: 4.51 %",
            "FWHM 511 change: -5.12%",
            "Ge/Cs difference: 0.5%",
        ]
        assert len(TOOLTIP_METRICS) == 9

    def test_missing_metrics_are_left_out(self) -> None:
        view = ChannelView(STATUS_FIT_FAILED, metrics={"n_events": 40.0, "cv_gain": math.nan})
        text = build_cell_tooltip((1, 16, 1, 25), "A03", 3, "anode", view)
        assert text.splitlines() == [
            "A03 (strip position 3)",
            "Node 1  Board 16  RENA 1  Ch 25",
            "Status: fit_failed",
            "Options: batch",
            "Events: 40",
        ]

    def test_cathode(self) -> None:
        view = ChannelView(CATHODE_UNCALIBRATED)
        text = build_cell_tooltip((1, 16, 1, 25), "C03", 3, "cathode", view)
        assert text.splitlines() == [
            "C03 (cathode 3)",
            "Node 1  Board 16  RENA 1  Ch 25",
            "keV calibration: uncalibrated",
        ]

    def test_without_view(self) -> None:
        text = build_cell_tooltip((1, 16, 0, 5), "A02", 2, "anode", None)
        assert text.endswith("Status: not fitted")
        text = build_cell_tooltip((1, 16, 0, 5), "A02", 2, "anode", None, has_data=False)
        assert text.endswith("Status: no data")
        with pytest.raises(ValueError, match="kind"):
            build_cell_tooltip((1, 16, 0, 5), "A02", 2, "strip", None)

    def test_model_cells_carry_their_tooltip(self) -> None:
        views = _mixed_views()
        model = build_system_map(views)
        cell = model.boards[(NODE, BOARD)].anodes[5]
        assert cell.tooltip == build_cell_tooltip(
            cell.channel, cell.label, 6, "anode", views[cell.channel]
        )
        assert "Options: override" in cell.tooltip


class TestSummaries:
    def test_no_data(self) -> None:
        assert format_summary(build_system_map({})) == NO_DATA_SUMMARY
        assert format_summary(build_system_map({}, color_mode="peak_spread")) == NO_DATA_SUMMARY

    def test_status_counts_and_summary(self) -> None:
        model = build_system_map(_mixed_views(), active_boards=[(1, 15)])
        counts = status_counts(model, "anode")
        assert list(counts) == list(ANODE_CATEGORIES)
        assert {category: n for category, n in counts.items() if n} == {
            CATEGORY_OK: 2,
            CATEGORY_FLAGGED: 1,
            CATEGORY_FAILED: 1,
            CATEGORY_TOO_FEW: 1,
            CATEGORY_NOT_FITTED: 2 * ANODES_PER_BOARD - 6,
            CATEGORY_NO_DATA: 1,
        }
        assert status_counts(model, "cathode")[CATEGORY_NOT_FITTED] == 2 * CATHODES_PER_BOARD
        text = format_summary(model)
        assert text.replace(NBSP, " ") == (
            "Anodes: 2 corrected, 1 flagged, 1 too few events, 1 fit failed, 72 not fitted,"
            " 1 no data | Cathodes: 16 not fitted"
        )
        # A count and its words never split across lines.
        assert f"1{NBSP}too{NBSP}few{NBSP}events" in text
        assert f"72{NBSP}not{NBSP}fitted" in text
        with pytest.raises(ValueError, match="kind"):
            status_counts(model, "strips")

    def test_metric_summary(self) -> None:
        model = build_system_map(_metric_views(), color_mode="cv_gain")
        low, high = model.limits_for("anode") or (math.nan, math.nan)
        text = format_summary(model)
        spec = METRIC_SPECS["cv_gain"]
        assert text.startswith("CV gain | Anodes: median 10.5%, colour range ")
        assert text.endswith(f"{spec.format(low)} to {spec.format(high)} (20{NBSP}anodes)")
        assert "Cathodes" not in text
        np.testing.assert_allclose(np.sort(metric_values(model, "anode")), np.arange(1, 21) / 100)

    def test_metric_summary_singular(self) -> None:
        model = build_system_map(
            {_anode(NODE, BOARD, 1): _ok(peak_spread=0.1)}, color_mode="peak_spread"
        )
        assert format_summary(model).endswith(f"(1{NBSP}anode)")

    def test_metric_summary_without_values(self) -> None:
        model = build_system_map({_anode(NODE, BOARD, 1): _ok()}, color_mode="ge_cs_max_diff")
        assert format_summary(model) == "Ge/Cs difference | Anodes: no values"

    def test_board_summary(self) -> None:
        model = build_system_map(_mixed_views(), active_boards=[(1, 15)])
        assert format_board_summary(model.boards[(NODE, BOARD)]) == (
            "Node 3 Board 17 (odd): anodes 3/39 corrected (1 flagged), cathodes 0/8 calibrated"
        )
        assert format_board_summary(model.boards[(1, 15)]) == (
            "Node 1 Board 15 (odd): anodes 0/39 corrected, cathodes 0/8 calibrated"
        )
        assert format_board_summary(model.boards[(2, 16)]).endswith(" - no data")


# ---------------------------------------------------------------------------
# Review fixes: ChannelView copies, lazy tooltips, unplaced keys, updates
# ---------------------------------------------------------------------------


class TestViewKind:
    def test_status_must_fit_the_electrode(self) -> None:
        with pytest.raises(ValueError, match="cathode .* status 'ok'"):
            build_system_map({_cathode(NODE, BOARD, 1): _ok()})
        with pytest.raises(ValueError, match="anode .* status 'calibrated'"):
            build_system_map({_anode(NODE, BOARD, 1): ChannelView(CATHODE_CALIBRATED)})
        model = build_system_map({_anode(NODE, BOARD, 1): _ok()})
        with pytest.raises(ValueError, match="cathode"):
            update_system_map(model, {_cathode(NODE, BOARD, 2): _ok()})


class TestChannelViewStorage:
    def test_metrics_are_a_read_only_copy(self) -> None:
        metrics = {"n_events": 10.0, "cv_gain": 1.0}
        view = ChannelView(STATUS_OK, metrics=metrics)
        metrics["cv_gain"] = 99.0
        assert view.metric("cv_gain") == 1.0
        with pytest.raises(TypeError):
            view.metrics["cv_gain"] = 5.0  # type: ignore[index]

    def test_hashable(self) -> None:
        a = ChannelView(STATUS_OK, (FLAG_SOURCE_INCONSISTENT,), metrics={"cv_gain": 1.0})
        b = ChannelView(STATUS_OK, (FLAG_SOURCE_INCONSISTENT,), metrics={"cv_gain": 1.0})
        c = ChannelView(STATUS_OK, (FLAG_SOURCE_INCONSISTENT,), metrics={"cv_gain": 2.0})
        assert a == b and hash(a) == hash(b)
        assert a != c  # metrics take part in equality, not in the hash
        assert len({a, b, c}) == 2

    def test_override(self) -> None:
        views = {
            _anode(NODE, BOARD, 1): ChannelView(STATUS_OK, options_source="override"),
            _anode(NODE, BOARD, 2): _ok(),
        }
        anodes = build_system_map(views).boards[(NODE, BOARD)].anodes
        assert anodes[0].is_override and anodes[0].view is not None and anodes[0].view.is_override
        assert not anodes[1].is_override
        assert not anodes[2].is_override  # no view


class TestLazyTooltips:
    def test_built_on_first_access_and_cached(self) -> None:
        views = _mixed_views()
        cell = build_system_map(views).boards[(NODE, BOARD)].anodes[0]
        assert cell._tooltip is None
        text = cell.tooltip
        assert text == build_cell_tooltip(cell.channel, cell.label, 1, "anode", cell.view)
        assert cell.tooltip is text

    def test_not_built_tooltips_match_after_recolor(self) -> None:
        model = build_system_map(_mixed_views(), data_channels=[_anode(NODE, BOARD, 9)])
        recolored = recolor_system_map(model, "fwhm_511_after")
        for key in ((NODE, BOARD), (1, 15)):
            for a, b in zip(model.boards[key].anodes, recolored.boards[key].anodes):
                assert b.tooltip == a.tooltip
        anodes = recolored.boards[(NODE, BOARD)].anodes
        assert anodes[8].tooltip.endswith("Status: not fitted")
        assert anodes[9].tooltip.endswith("Status: no data")


class TestUnplacedViews:
    def test_non_electrode_keys_do_not_mark_their_board(self) -> None:
        views = {(1, 15, 0, 1): _ok(), (1, 15, 2, 4): _ok(), (1, 16, 1, 6): _ok()}
        model = build_system_map(views, data_channels=[(1, 17, 0, 2)])
        assert not model.boards[(1, 15)].has_data
        assert not model.boards[(1, 16)].has_data
        assert not model.boards[(1, 17)].has_data  # data channels on inactive channels too
        assert model.unplaced_channels == ((1, 15, 0, 1), (1, 15, 2, 4), (1, 16, 1, 6))
        assert all(isinstance(c, ChannelAddress) for c in model.unplaced_channels)
        assert model.unmapped_boards == ()
        assert format_summary(model) == NO_DATA_SUMMARY

    def test_with_a_placed_view(self) -> None:
        views = {(1, 15, 0, 1): _ok(), _anode(1, 15, 3): _ok()}
        model = build_system_map(views)
        assert model.boards[(1, 15)].has_data
        assert model.unplaced_channels == ((1, 15, 0, 1),)
        assert status_counts(model, "anode")[CATEGORY_OK] == 1


def _update_cases() -> list[tuple[str, dict[ChannelTuple, ChannelView]]]:
    return [
        ("replace", {_anode(NODE, BOARD, 1): ChannelView(STATUS_FIT_FAILED)}),
        ("flag", {_anode(NODE, BOARD, 7): ChannelView(STATUS_OK, (FLAG_CONVEX_CURVE,))}),
        (
            "new board",
            {
                _anode(9, 30, 2): _ok(cv_gain=0.03),
                _cathode(9, 30, 2): ChannelView(CATHODE_CALIBRATED),
            },
        ),
        (
            "board refit",
            {
                _anode(NODE, EVEN_BOARD, p): _ok(cv_gain=40.0 + p, peak_spread=0.1)
                for p in range(1, 40)
            },
        ),
        ("outside", {(11, 20, 0, 4): _ok(), (1, 14, 0, 4): _ok()}),
        ("unplaced", {(NODE, BOARD, 0, 2): _ok()}),
    ]


class TestUpdate:
    """update_system_map equals a full rebuild over the merged views."""

    @pytest.mark.parametrize("mode", [MODE_STATUS, "cv_gain"])
    @pytest.mark.parametrize("with_channels", [False, True])
    @pytest.mark.parametrize(("name", "changed"), _update_cases(), ids=lambda v: str(v)[:12])
    def test_equals_rebuild(
        self,
        mode: str,
        with_channels: bool,
        name: str,
        changed: dict[ChannelTuple, ChannelView],
    ) -> None:
        base = _metric_views()
        boards = [(1, 15), (NODE, BOARD), (NODE, EVEN_BOARD)]
        channels = list(base) + [_anode(1, 15, 4)] if with_channels else None
        model = build_system_map(base, boards, channels, color_mode=mode)
        updated = update_system_map(model, changed)
        rebuilt = build_system_map({**base, **changed}, boards, channels, color_mode=mode)
        assert _cell_state(updated) == _cell_state(rebuilt), name
        assert updated.limits == rebuilt.limits
        assert updated.unmapped_boards == rebuilt.unmapped_boards
        assert updated.unplaced_channels == rebuilt.unplaced_channels
        assert updated.data_channels_known == with_channels

    def test_keeps_untouched_cells(self) -> None:
        model = build_system_map(_metric_views())
        untouched = model.boards[(NODE, BOARD)].anodes[5]
        text = untouched.tooltip
        changed_cell = model.boards[(NODE, BOARD)].anodes[0]
        old_text = changed_cell.tooltip
        updated = update_system_map(model, {changed_cell.channel: ChannelView(STATUS_FIT_FAILED)})
        assert updated.boards[(NODE, BOARD)].anodes[5].tooltip is text
        new_cell = updated.boards[(NODE, BOARD)].anodes[0]
        assert new_cell._tooltip is None
        assert new_cell.tooltip != old_text and "Status: fit_failed" in new_cell.tooltip
        assert model.boards[(NODE, BOARD)].anodes[0].category == CATEGORY_OK  # not modified
        assert update_system_map(model, {}).boards[(NODE, BOARD)].anodes[5].tooltip is text


class TestColourText:
    @pytest.mark.parametrize(
        ("percentiles", "expected"),
        [
            ((2.0, 98.0), "2nd-98th percentile"),
            ((1.0, 99.0), "1st-99th percentile"),
            ((3.0, 97.0), "3rd-97th percentile"),
            ((11.0, 13.0), "11th-13th percentile"),
            ((21.0, 22.0), "21st-22nd percentile"),
            ((2.5, 97.5), "2.5th-97.5th percentile"),
        ],
    )
    def test_clip_range_text(self, percentiles: tuple[float, float], expected: str) -> None:
        assert clip_range_text(percentiles) == expected
        assert clip_range_text() == "2nd-98th percentile"

    def test_text_colour_over_several_fills(self) -> None:
        lut = sequential_lut()
        dark, light = text_color_for(lut[-1]), text_color_for(lut[0])
        assert text_color_for_fills([lut[0]]) == light
        assert text_color_for_fills([lut[-1]]) == dark
        assert text_color_for_fills([]) == light
        # One fill, or fills of similar lightness: no halo needed.
        assert label_colors_for_fills([lut[0]]) == (light, None)
        assert label_colors_for_fills([lut[-1], _qc(CATEGORY_COLORS[CATEGORY_OK])]) == (dark, None)
        # A label over both ends of viridis cannot contrast with both: it gets
        # the colour with the better worst case plus a halo in the other one.
        text, halo = label_colors_for_fills([lut[0], lut[-1]])
        assert halo is not None and {text.name(), halo.name()} == {dark.name(), light.name()}
        # Black and yellow: dark text (worst contrast 1.26 vs 1.06 for light).
        assert label_colors_for_fills([_qc("#000000"), _qc("#ffff00")]) == (dark, light)
