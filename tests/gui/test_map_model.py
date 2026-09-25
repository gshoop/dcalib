"""dcalib's changes to the forked System Map model and colours."""

from __future__ import annotations

import math

import pytest

from dcalib.gui import map_colors as mc
from dcalib.gui._system_map_model import (
    ChannelView,
    build_system_map,
    format_board_summary,
    format_summary,
    recolor_system_map,
    status_counts,
)

ANODE = (1, 16, 0, 10)
ANODE2 = (1, 16, 0, 11)
CATHODE = (1, 16, 0, 25)


def _views() -> dict[tuple[int, int, int, int], ChannelView]:
    return {
        ANODE: ChannelView(
            "ok", metrics={"cv_gain": 0.05, "fwhm_511_before": 6.0, "fwhm_511_after": 5.4}
        ),
        ANODE2: ChannelView("no_gain", review="rejected", metrics={"cv_gain": 0.001}),
        (1, 16, 1, 12): ChannelView("no_calibrated_cathode"),
        (1, 16, 1, 13): ChannelView("ok", flags=("source_inconsistent",)),
        CATHODE: ChannelView("calibrated"),
        (1, 16, 0, 26): ChannelView("uncalibrated"),
    }


def test_status_category_rules() -> None:
    assert mc.status_category("ok") == mc.CATEGORY_OK
    assert mc.status_category("ok", ["convex_curve"]) == mc.CATEGORY_FLAGGED
    assert (
        mc.status_category("ok", ["partial_cathode_coverage"], mc.DEFAULT_INFORMATIONAL_FLAGS)
        == mc.CATEGORY_OK
    )
    assert mc.status_category("no_gain", review="rejected") == mc.CATEGORY_REJECTED
    assert mc.status_category("no_anode_calibration") == mc.CATEGORY_NO_CALIBRATION
    assert mc.status_category("no_depth_dependence") == mc.CATEGORY_NO_DEPTH
    assert mc.status_category("calibrated") == mc.CATEGORY_CATHODE_CALIBRATED
    assert mc.status_category(None) == mc.CATEGORY_NOT_FITTED
    assert mc.status_category("ok", has_data=False) == mc.CATEGORY_NO_DATA
    with pytest.raises(ValueError):
        mc.status_category("bogus")
    assert set(mc.CATEGORY_COLORS) == set(mc.CATEGORIES)
    assert len(set(mc.CATEGORY_COLORS.values())) == len(mc.CATEGORIES)


def test_views() -> None:
    view = _views()[ANODE]
    assert view.metric("fwhm_511_change") == pytest.approx(-0.1)
    assert not view.is_cathode and _views()[CATHODE].is_cathode
    with pytest.raises(ValueError):
        ChannelView("ok", review="maybe")
    with pytest.raises(ValueError):
        ChannelView("fitted")


def test_model_categories_and_summaries() -> None:
    model = build_system_map(_views(), active_boards=[(1, 16)], data_channels=list(_views()))
    assert model.cell(ANODE).category == mc.CATEGORY_OK  # type: ignore[union-attr]
    assert model.cell(ANODE2).category == mc.CATEGORY_REJECTED  # type: ignore[union-attr]
    assert model.cell(CATHODE).category == mc.CATEGORY_CATHODE_CALIBRATED  # type: ignore[union-attr]
    anodes = status_counts(model, "anode")
    assert anodes[mc.CATEGORY_OK] == 1 and anodes[mc.CATEGORY_FLAGGED] == 1
    assert anodes[mc.CATEGORY_NO_DATA] == 35  # anodes without a view, with data_channels given
    cathodes = status_counts(model, "cathode")
    assert cathodes == {"cathode_calibrated": 1, "cathode_uncalibrated": 1, "no_data": 6}
    summary = format_summary(model)
    assert "1 corrected" in summary and "1 calibrated" in summary
    board = format_board_summary(model.boards[(1, 16)])
    # A flagged anode is corrected too (it is in the .dcc).
    assert "anodes 2/39 corrected (1 flagged, 1 rejected)" in board and "cathodes 1/8" in board


def test_metric_mode_keeps_cathode_colours() -> None:
    model = recolor_system_map(build_system_map(_views(), active_boards=[(1, 16)]), "cv_gain")
    assert model.limits_for("anode") == pytest.approx((0.001 + 0.02 * 0.049, 0.05 - 0.02 * 0.049))
    assert model.limits_for("cathode") is None
    cathode = model.cell(CATHODE)
    assert cathode is not None and cathode.fill == mc.category_qcolor(
        mc.CATEGORY_CATHODE_CALIBRATED
    )
    assert math.isnan(cathode.value)
    assert "CV gain | Anodes: median" in format_summary(model)
    assert ("Cathode calibrated", mc.CATEGORY_COLORS["cathode_calibrated"]) in mc.legend_entries(
        "cv_gain"
    )
