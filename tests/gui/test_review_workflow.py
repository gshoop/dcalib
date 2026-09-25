"""Plan 10.6: open the stored results, Reject/unreject, Fit Channel override and Revert,
export contents, the stale-results banner, and the Spectra, Board grid and Fleet tabs."""

from __future__ import annotations

from pathlib import Path

from pytestqt.qtbot import QtBot

from dcalib.channels import AnodeKey
from dcalib.io.dcc import read_dcc
from dcalib.io.summary_csv import read_summary_csv
from dcalib.options import DepthOptions
from dcalib.results import ResultsFile
from tests.conftest import DEPTH_BOARDS, UNCALIBRATED
from tests.gui.conftest import WAIT_MS, MakeWindow, wait_open
from tests.synthetic_cache import depth_calibrations, write_kev

STEEP = AnodeKey(3, 16, 0, 10)
CONCAVE = AnodeKey(3, 15, 0, 9)


def _open(qtbot: QtBot, make_window: MakeWindow, cache: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    window, dialogs = make_window(**kwargs)
    window.open_cache(cache)
    wait_open(qtbot, window)
    return window, dialogs


def _select(qtbot: QtBot, window, key: AnodeKey) -> None:  # type: ignore[no-untyped-def]
    window.select_anode(key)
    qtbot.waitUntil(
        lambda: window.depth_view.data is not None and window.depth_view.data.key == key,
        timeout=WAIT_MS,
    )


def test_reject_and_unreject(qtbot: QtBot, make_window: MakeWindow, results_cache: Path) -> None:
    window, dialogs = _open(qtbot, make_window, results_cache)
    _select(qtbot, window, STEEP)
    window._ask_note = lambda _key: "bad cathode"  # type: ignore[method-assign]
    window.band.reject_button.click()
    assert window.band.reject_button.isChecked() and window.band.reject_button.text() == "Rejected"
    assert window.session.result(STEEP).review == "rejected"  # type: ignore[union-attr]
    assert "rejected" in window.status_message()
    rows = window.inspector.rows()
    assert rows["review"] == "rejected" and rows["note"] == "bad cathode"
    cell = window.system_map.model.cell(STEEP)
    assert cell is not None and cell.category == "rejected"
    stored = ResultsFile(results_cache.with_name("data_20260911_124617.depth.h5")).load()
    assert stored is not None and stored.review[STEEP].note == "bad cathode"
    # Cancelling the note dialog does not reject.
    window.band.reject_button.click()  # unreject
    assert window.session.result(STEEP).review == ""  # type: ignore[union-attr]
    window._ask_note = lambda _key: None  # type: ignore[method-assign]
    window.band.reject_button.click()
    assert not window.band.reject_button.isChecked() and window.session.result(STEEP).review == ""  # type: ignore[union-attr]
    # From the map's context menu.
    window._ask_note = lambda _key: ""  # type: ignore[method-assign]
    menu = window.system_map.build_context_menu(3, 16, STEEP)
    reject = next(a for a in menu.actions() if a.text().startswith("Reject"))
    reject.trigger()
    assert window.session.result(STEEP).review == "rejected"  # type: ignore[union-attr]
    menu = window.system_map.build_context_menu(3, 16, STEEP)
    next(a for a in menu.actions() if a.text() == "Unreject").trigger()
    assert window.session.result(STEEP).review == ""  # type: ignore[union-attr]
    assert not dialogs.errors


def test_fit_channel_override_and_revert(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path
) -> None:
    window, dialogs = _open(qtbot, make_window, results_cache)
    _select(qtbot, window, STEEP)
    assert not window.band.revert_button.isEnabled()
    window.band.set_options(DepthOptions(max_degree=1))
    window.band._options = DepthOptions(max_degree=1)
    assert window.refit_channel()
    qtbot.waitUntil(lambda: not window.is_busy, timeout=WAIT_MS)
    assert not dialogs.errors
    result = window.session.result(STEEP)
    assert result is not None and result.options_source == "override" and result.degree == 1
    assert "override(s) stored" in window.status_message()
    qtbot.waitUntil(lambda: window.depth_view.data is not None and window.depth_view.data.curve.degree == 1, timeout=WAIT_MS)  # type: ignore[union-attr]
    assert window.inspector.rows()["options_source"] == "override"
    assert window.system_map.model.cell(STEEP).is_override  # type: ignore[union-attr]
    assert window.band.revert_button.isEnabled()
    # The map's menu offers the reverts.
    texts = [a.text() for a in window.system_map.build_context_menu(3, 16, STEEP).actions()]
    assert "Revert Channel to batch" in texts and any(t.startswith("Revert Board") for t in texts)
    assert window.revert_channel()
    assert window.session.result(STEEP).options_source == "batch"  # type: ignore[union-attr]
    assert "Reverted 1 override" in window.status_message()
    assert not window.band.revert_button.isEnabled()
    # A board re-fit with the batch options stores nothing.
    window.band._options = DepthOptions()
    assert window.refit_board()
    qtbot.waitUntil(lambda: not window.is_busy, timeout=WAIT_MS)
    assert "reverted" in window.status_message() and not window.session.override_keys()


def test_export_contents(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path, tmp_path: Path
) -> None:
    window, dialogs = _open(qtbot, make_window, results_cache)
    window.session.set_review(CONCAVE, True, "")
    out = tmp_path / "export"
    window._choose_directory = lambda _title, _start: str(out)  # type: ignore[method-assign]
    window._on_export()
    assert not dialogs.errors
    dcc = read_dcc(out / "data_20260911_124617.dcc")
    assert set(dcc) == {STEEP}  # the rejected anode is left out
    rows = {r.key: r for r in read_summary_csv(out / "depth_summary.csv")}
    assert rows[CONCAVE].review == "rejected" and len(rows) == 6
    assert "Exported 1 corrected" in window.status_message()


def test_stale_banner(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path, tmp_path: Path
) -> None:
    kev = write_kev(
        tmp_path / "other.kev",
        {
            k: (v[0] * 1.001, v[1])
            for k, v in depth_calibrations(DEPTH_BOARDS, UNCALIBRATED).items()
        },
    )
    window, dialogs = _open(qtbot, make_window, results_cache, kev_path=kev)
    assert window.session.stale and window.stale_banner_visible
    assert "calibration changed" in window.stale_label.text()
    assert (
        not window.fit_channel_action.isEnabled() and not window.band.fit_channel_button.isEnabled()
    )
    window.refit_channel(STEEP)
    assert dialogs.errors and "stale" in dialogs.errors[-1][1]
    # Fit All replaces the stale results: the banner goes.
    assert window.start_fit_all()
    qtbot.waitUntil(lambda: not window.is_busy, timeout=WAIT_MS)
    assert not window.session.stale and not window.stale_banner_visible


def test_valid_results_have_no_banner(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path
) -> None:
    window, _ = _open(qtbot, make_window, results_cache)
    assert not window.stale_banner_visible


def test_tabs(qtbot: QtBot, make_window: MakeWindow, results_cache: Path) -> None:
    window, _ = _open(qtbot, make_window, results_cache)
    _select(qtbot, window, STEEP)
    assert window.spectra_view.data is not None and window.spectra_view.data.key == STEEP
    assert "after FWHM" in window.spectra_view.panel_text(0)
    assert "after FWHM" in window.spectra_view.panel_text(1)
    # Board grid: the board of the selection, click opens an anode.
    assert window.board_grid.board == (3, 16)
    flat = AnodeKey(3, 16, 0, 11)
    center = window.board_grid.cell_center(flat)
    assert center is not None
    window.board_grid.activate_at(center)
    qtbot.waitUntil(lambda: window.depth_view.data.key == flat, timeout=WAIT_MS)  # type: ignore[union-attr]
    assert "not corrected" in window.spectra_view.title_label.text()
    # Fleet summary: counts and the worst table; clicking a row opens it.
    assert "6 anodes" in window.fleet_view.counts_label.text()
    keys = window.fleet_view.table_keys()
    assert keys and set(keys) <= {STEEP, CONCAVE}
    window.fleet_view._on_cell_clicked(0, 0)
    qtbot.waitUntil(lambda: window.depth_view.data.key == keys[0], timeout=WAIT_MS)  # type: ignore[union-attr]
    window.spectra_view.log_check.setChecked(True)
    window.close()
    again, _ = make_window()
    assert again.spectra_view.log_scale()
