"""Tests of the analysis driver: AnodeResult, analyze_anode/board/all and merging."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dcalib import analysis
from dcalib.analysis import AnodeResult
from dcalib.calib import load_calibrations
from dcalib.channels import AnodeKey
from dcalib.events import build_board_events
from dcalib.options import DepthOptions

EXPECTED_STATUS = {
    AnodeKey(3, 15, 0, 9): "ok",
    AnodeKey(3, 16, 0, 10): "ok",
    AnodeKey(3, 16, 0, 11): "no_depth_dependence",
    AnodeKey(3, 16, 0, 12): "too_few_events",
    AnodeKey(3, 16, 1, 12): "no_anode_calibration",
    AnodeKey(3, 16, 1, 13): "no_calibrated_cathode",
}


def _row(**kwargs: object) -> AnodeResult:
    values: dict[str, object] = {"node": 1, "board": 16, "rena": 0, "channel": 10, "status": "ok"}
    values.update(kwargs)
    return AnodeResult(**values)  # type: ignore[arg-type]


class TestAnodeResult:
    def test_normalisation(self) -> None:
        row = _row(
            n_events=np.int64(5),
            p0=np.float32(505.0),
            p1=float("nan"),
            degree=np.int64(1),
            flags=["partial_cathode_coverage", "slice_fit_failed"],
        )
        assert type(row.n_events) is int and row.p1 is None and row.degree == 1
        assert row.flags == ("slice_fit_failed", "partial_cathode_coverage")
        assert row.flags_text == "slice_fit_failed;partial_cathode_coverage"
        assert row.key == AnodeKey(1, 16, 0, 10) and row.ok and row.exported

    @pytest.mark.parametrize(
        "kwargs",
        [{"status": "good"}, {"review": "maybe"}, {"options_source": "gui"}, {"flags": ["x"]}],
    )
    def test_invalid_values(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _row(**kwargs)

    def test_types(self) -> None:
        with pytest.raises(TypeError):
            _row(n_events=1.5)
        with pytest.raises(TypeError):
            _row(flags="slice_fit_failed")
        with pytest.raises(ValueError):
            _row(degree=-1)

    def test_curve_and_export(self) -> None:
        row = _row(p0=511.0, p1=5.11, p2=-10.22, degree=2)
        assert row.coefficients == (511.0, 5.11, -10.22)
        np.testing.assert_allclose(row.g(np.array([0.0, 1.0])), [1.0, 1.0 + 0.01 - 0.02])
        assert np.isnan(_row().g(np.array([0.5]))).all() and _row().coefficients is None
        assert not _row(review="rejected").exported and not _row(status="no_gain").exported

    def test_dict_round_trip(self) -> None:
        row = _row(p0=500.0, flags=("convex_curve",), review="rejected")
        data = row.to_dict()
        assert tuple(data) == analysis.CSV_COLUMNS
        data["flags"] = "convex_curve"
        assert AnodeResult.from_dict(data) == row
        with pytest.raises(ValueError, match="Unknown"):
            AnodeResult.from_dict({**data, "bogus": 1})
        assert AnodeResult.from_dict({**data, "bogus": 1}, strict=False) == row


class TestAnalyze:
    def test_board_statuses_and_counts(self, depth_cache: Path) -> None:
        cals = load_calibrations(depth_cache)
        out = analysis.analyze_all(depth_cache, cals, workers=1)
        statuses = {r.key: r.status for r in out.results}
        assert statuses == EXPECTED_STATUS
        by_key = {r.key: r for r in out.results}
        steep = by_key[AnodeKey(3, 16, 0, 10)]
        assert steep.n_events == 4000 and steep.n_ge + steep.n_cs == 4000
        assert steep.degree == 2 and steep.p0 is not None and steep.cv_gain is not None
        assert steep.p0 == pytest.approx(511 * 1.01, rel=0.01)
        assert steep.electrode.startswith("A") and steep.n_slices is not None
        assert steep.fwhm_511_after is not None and steep.fwhm_511_before is not None
        assert steep.fwhm_511_after < steep.fwhm_511_before
        assert steep.peak_511 == pytest.approx(1.0, abs=0.01)
        flat = by_key[AnodeKey(3, 16, 0, 11)]
        assert flat.fwhm_511_after == flat.fwhm_511_before and flat.degree == 0
        uncal = by_key[AnodeKey(3, 16, 1, 13)]
        assert uncal.n_uncal_cathode == uncal.n_events == 300
        assert uncal.flags == ("partial_cathode_coverage",)
        # Slice rows: pooled and per-source curves of the fitted anodes only.
        curves = {(s.key, s.curve) for s in out.slices}
        assert (AnodeKey(3, 16, 0, 10), "both") in curves
        assert (AnodeKey(3, 16, 0, 10), "ge") in curves
        assert all(s.key in by_key and by_key[s.key].n_slices for s in out.slices)
        rows = [s for s in out.slices if s.key == steep.key]
        rebuilt = analysis.slices_from_rows(rows)
        assert len(rebuilt["both"]) == steep.n_slices

    def test_analyze_board_subset_and_events(self, depth_cache: Path) -> None:
        cals = load_calibrations(depth_cache)
        events = build_board_events(depth_cache, 3, 16)
        out = analysis.analyze_board(
            None,
            3,
            16,
            cals.board_lut(3, 16),
            DepthOptions(max_degree=1),
            events=events,
            anodes=[(0, 10)],
            options_source="override",
        )
        assert [r.key for r in out.results] == [AnodeKey(3, 16, 0, 10)]
        assert out.results[0].options_source == "override" and out.results[0].degree == 1
        with pytest.raises(ValueError):
            analysis.analyze_board(None, 3, 16, cals.board_lut(3, 16))

    def test_stop_flag(self, depth_cache: Path) -> None:
        cals = load_calibrations(depth_cache)
        stop = threading.Event()
        stop.set()
        with pytest.raises(analysis.AnalysisCancelled):
            analysis.analyze_all(depth_cache, cals, workers=1, stop_flag=stop)
        with pytest.raises(ValueError):
            analysis.analyze_all(depth_cache, cals, workers=0)

    def test_progress_and_board_filter(self, depth_cache: Path) -> None:
        cals = load_calibrations(depth_cache)
        seen: list[tuple[int, int]] = []
        out = analysis.analyze_all(
            depth_cache,
            cals,
            workers=1,
            progress_cb=lambda d, t: seen.append((d, t)),
            boards=[(3, 15)],
        )
        assert {r.board for r in out.results} == {15}
        assert seen[0][0] == 0 and seen[-1][0] == seen[-1][1] > 0

    @pytest.mark.slow
    def test_pool_matches_in_process(self, depth_cache: Path) -> None:
        cals = load_calibrations(depth_cache)
        single = analysis.analyze_all(depth_cache, cals, workers=1)
        pooled = analysis.analyze_all(depth_cache, cals, workers=2)
        assert pooled.results == single.results
        assert pooled.slices == single.slices

    def test_schedule_and_workers(self) -> None:
        assert analysis.schedule_boards({(1, 15): 5, (1, 16): 9, (2, 15): 9}) == [
            (1, 16),
            (2, 15),
            (1, 15),
        ]
        assert 1 <= analysis.default_workers() <= analysis.MAX_DEFAULT_WORKERS


def test_merge_results() -> None:
    batch = [_row(channel=10, p0=500.0), _row(channel=11, status="no_gain")]
    override = replace(batch[1], status="ok", p0=505.0, options_source="batch")
    merged = analysis.merge_results(batch, [override], {AnodeKey(1, 16, 0, 10): "rejected"})
    assert [r.channel for r in merged] == [10, 11]
    assert merged[0].review == "rejected" and not merged[0].exported
    assert merged[1].options_source == "override" and merged[1].p0 == 505.0
    assert analysis.merge_results(batch, [])[0] is batch[0]
    assert not math.isnan(merged[1].p0 or 0.0)
