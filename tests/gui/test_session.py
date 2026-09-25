"""Tests of the Qt-free GUI session layer."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from dcalib.analysis import AnalysisCancelled
from dcalib.channels import AnodeKey
from dcalib.gui import session as ses
from dcalib.options import DepthOptions
from tests.conftest import DEPTH_BOARDS, UNCALIBRATED
from tests.synthetic_cache import depth_calibrations, write_kev

STEEP = AnodeKey(3, 16, 0, 10)
FLAT = AnodeKey(3, 16, 0, 11)


def test_open_without_results(depth_cache: Path) -> None:
    opened = ses.open_cache(depth_cache)
    assert opened.stored is None and opened.validity is None
    assert set(opened.board_hits) == {(3, 15), (3, 16)}
    session = ses.DepthSession()
    session.install(opened)
    assert session.is_open and not session.has_results and not session.stale
    assert len(session.anodes_on_board(3, 16)) == 39  # every anode before a Fit All
    views = session.views()
    assert views[AnodeKey(3, 16, 0, 25)].status == "calibrated"
    assert views[AnodeKey(3, 16, 1, 26)].status == "uncalibrated"
    assert "no results" in session.describe()


def test_open_errors(tmp_path: Path) -> None:
    with pytest.raises(ses.SessionError, match="not found"):
        ses.open_cache(tmp_path / "none.cache.h5")
    bogus = tmp_path / "x.cache.h5"
    bogus.write_text("x")
    with pytest.raises(ses.SessionError, match="not an HDF5"):
        ses.open_cache(bogus)


def test_results_navigation_and_anode_data(results_cache: Path) -> None:
    session = ses.DepthSession(board_cache_size=1)
    session.install(ses.open_cache(results_cache))
    assert session.has_results and not session.stale
    assert session.result(STEEP) is not None and session.result(STEEP).ok  # type: ignore[union-attr]
    assert session.anodes_on_board(3, 16)[0] == STEEP
    assert session.step_anode(None, 1) == AnodeKey(3, 15, 0, 9)
    assert session.step_anode(AnodeKey(3, 15, 0, 9), 1) == STEEP
    assert session.step_anode(STEEP, -5) == AnodeKey(3, 15, 0, 9)
    data = session.anode_data(STEEP)
    assert data.key == STEEP and data.corrected and data.curve is not None
    assert len(data.x) == data.n_events == 4000 and data.n_uncal_cathode == 0
    assert set(data.slices) >= {"both", "ge"} and set(data.source_curves) == {"ge", "cs"}
    np.testing.assert_allclose(data.g(np.array([0.5])), data.result.g(np.array([0.5])))  # type: ignore[union-attr]
    assert np.all(np.isfinite(data.corrected_x()))
    assert data.cathodes() == [(0, 25)] and data.options == DepthOptions()
    other = session.anode_data(AnodeKey(3, 15, 0, 9))
    assert session.cached_boards() == [(3, 15)] and other.corrected  # LRU of one board
    uncal = session.anode_data(AnodeKey(3, 16, 1, 13))
    assert len(uncal.x) == 0 and uncal.n_uncal_cathode == 300 and not uncal.corrected


def test_run_batch_apply_and_stale(depth_cache: Path, tmp_path: Path) -> None:
    session = ses.DepthSession()
    session.install(ses.open_cache(depth_cache))
    outcome = session.run_batch(DepthOptions(max_degree=1), workers=1)
    assert "anodes" in outcome.describe() and outcome.stored is not None
    views = session.apply_batch(outcome)
    assert views[STEEP].status == "ok" and session.batch_options == DepthOptions(max_degree=1)
    assert session.options == DepthOptions(max_degree=1)
    # Another calibration (.kev with other values): the stored results are stale.
    kev = write_kev(
        tmp_path / "cal.kev",
        {
            k: (v[0] * 1.001, v[1])
            for k, v in depth_calibrations(DEPTH_BOARDS, UNCALIBRATED).items()
        },
    )
    stale = ses.DepthSession()
    stale.install(ses.open_cache(depth_cache, kev_path=kev))
    assert stale.stale and stale.has_results and "stale" in stale.describe()
    assert stale.options == DepthOptions()  # a stale batch's options are not taken over


def test_batch_needs_a_writable_results_file(depth_cache: Path, tmp_path: Path) -> None:
    session = ses.DepthSession()
    session.install(ses.open_cache(depth_cache, results_path=tmp_path / "gone" / "r.depth.h5"))
    with pytest.raises(ses.SessionError, match="--results PATH"):
        session.run_batch(DepthOptions(), workers=1)  # refused before the analysis
    assert not session.busy


def test_batch_stop_and_busy(depth_cache: Path) -> None:
    session = ses.DepthSession()
    session.install(ses.open_cache(depth_cache))
    stop = threading.Event()
    stop.set()
    with pytest.raises(AnalysisCancelled):
        session.run_batch(DepthOptions(), workers=1, stop_flag=stop)
    assert not session.busy
    with session._exclusive("Fit All"), pytest.raises(ses.SessionBusyError):
        session.run_batch(DepthOptions(), workers=1)


def test_titles() -> None:
    assert ses.anode_title((1, 16, 0, 10)).startswith("n1 b16 A")
    assert "?" in ses.anode_title((1, 16, 0, 2))


# ---------------------------------------------------------------------------
# Phase 7: re-fits, reverts, review, export, board grid
# ---------------------------------------------------------------------------


def _session(cache: Path) -> ses.DepthSession:
    session = ses.DepthSession()
    session.install(ses.open_cache(cache))
    return session


def test_refit_override_and_revert(results_cache: Path) -> None:
    session = _session(results_cache)
    assert session.refit_blocked_reason() == ""
    request = session.refit_request(3, 16, ((0, 10),), DepthOptions(max_degree=1))
    assert request.as_override and not request.is_board and "override" in request.describe_start()
    outcome = session.run_refit(request)
    assert outcome.n_saved == 1 and "1 override(s) stored" in outcome.describe()
    assert session.apply_refit(outcome) == (STEEP,)
    result = session.result(STEEP)
    assert result is not None and result.options_source == "override" and result.degree == 1
    assert session.has_override(STEEP) and session.override_keys(3, 16) == [STEEP]
    assert session.options_for(STEEP) == DepthOptions(max_degree=1)
    assert session.anode_data(STEEP).curve.degree == 1  # type: ignore[union-attr]
    # A board re-fit with the batch options reverts instead.
    request = session.refit_request(3, 16, None, DepthOptions())
    assert not request.as_override and request.is_board
    outcome = session.run_refit(request)
    session.apply_refit(outcome)
    assert outcome.n_deleted == 1 and not session.has_override(STEEP)
    # Explicit revert.
    session.apply_refit(
        session.run_refit(session.refit_request(3, 16, ((0, 10),), DepthOptions(max_degree=1)))
    )
    assert session.revert([STEEP]) == 1 and session.revert([STEEP]) == 0
    assert session.result(STEEP).options_source == "batch"  # type: ignore[union-attr]


def test_refit_blocked(depth_cache: Path) -> None:
    session = ses.DepthSession()
    assert "open a cache" in session.refit_blocked_reason()
    session.install(ses.open_cache(depth_cache))
    assert "Fit All" in session.refit_blocked_reason()
    with pytest.raises(ses.SessionError, match="Cannot re-fit"):
        session.refit_request(3, 16)
    with pytest.raises(ses.SessionError, match="Fit All first"):
        session.set_review(STEEP, True)
    with pytest.raises(ses.SessionError, match="Nothing to export"):
        session.export_outputs(depth_cache.parent / "out")


def test_review_and_export(results_cache: Path, tmp_path: Path) -> None:
    session = _session(results_cache)
    session.set_review(STEEP, True, "odd curve")
    assert session.result(STEEP).review == "rejected"  # type: ignore[union-attr]
    assert session.review(STEEP).note == "odd curve"  # type: ignore[union-attr]
    summary = session.export_outputs(tmp_path / "export")
    assert summary.n_lines == 1 and summary.n_rows == 6 and "Exported 1" in summary.describe()
    from dcalib.io.dcc import read_dcc

    assert set(read_dcc(summary.dcc_path)) == {AnodeKey(3, 15, 0, 9)}
    # Persisted: a new session sees the rejection.
    again = _session(results_cache)
    assert again.result(STEEP).review == "rejected"  # type: ignore[union-attr]
    again.set_review(STEEP, False)
    assert again.review(STEEP) is None and _session(results_cache).review(STEEP) is None


def test_board_grid_entries(results_cache: Path) -> None:
    session = _session(results_cache)
    entries = session.board_grid(3, 16)
    assert len(entries) == 39 and [e.position for e in entries] == list(range(1, 40))
    steep = next(e for e in entries if e.key == STEEP)
    assert steep.result is not None and steep.curve is not None and "both" in steep.slices
    assert sum(1 for e in entries if e.result is not None) == 5
