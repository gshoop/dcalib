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
