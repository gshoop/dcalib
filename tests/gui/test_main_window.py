"""Tests of the main window (plan 10.6, phase 6 part): open, browse from the map, Fit All."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from pytestqt.qtbot import QtBot

from dcalib.channels import AnodeKey, anode_channels
from dcalib.gui.controls import ControlBand
from dcalib.gui.depth_view import DepthDisplay
from dcalib.gui.main import main as gui_main
from dcalib.gui.window import TAB_TITLES
from tests.gui.conftest import WAIT_MS, MakeWindow, wait_open

STEEP = AnodeKey(3, 16, 0, 10)
CONCAVE = AnodeKey(3, 15, 0, 9)


def _click_map_anode(qtbot: QtBot, window: object, key: AnodeKey) -> None:
    system_map = window.system_map  # type: ignore[attr-defined]
    model = system_map.model
    located = model.locate(key)
    assert located is not None
    board, kind, position = located
    grid = system_map.panel_grids[0] if key.node <= 5 else system_map.panel_grids[1]
    rect = grid.cell_rect(key.node, key.board, position)
    QTest.mouseClick(
        grid, Qt.MouseButton.LeftButton, pos=QPoint(int(rect.center().x()), int(rect.center().y()))
    )


def test_open_with_results_and_browse_from_map(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path
) -> None:
    window, dialogs = make_window()
    assert window.open_cache(results_cache)
    wait_open(qtbot, window)
    assert not dialogs.errors
    assert "2 corrected" in window.status_message()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == list(TAB_TITLES)
    # The first anode is shown at once.
    assert window.depth_view.data is not None and window.depth_view.data.key == CONCAVE
    assert "corrected" in window.system_map.summary_label.text()
    # Browse: click the steep anode on the map.
    _click_map_anode(qtbot, window, STEEP)
    qtbot.waitUntil(
        lambda: window.depth_view.data is not None and window.depth_view.data.key == STEEP,
        timeout=WAIT_MS,
    )
    assert window.band.selected_anode() == STEEP and window.band.status_label.text() == "ok"
    rows = window.inspector.rows()
    assert rows["status"] == "ok" and rows["degree"] == "2" and rows["options_source"] == "batch"
    assert window.depth_view.after_plot.titleLabel.text.startswith("After correction")
    # Prev / Next.
    window.step_anode(1)
    qtbot.waitUntil(lambda: window.depth_view.data.key == AnodeKey(3, 16, 0, 11), timeout=WAIT_MS)  # type: ignore[union-attr]
    window.step_anode(-1)
    qtbot.waitUntil(lambda: window.depth_view.data.key == STEEP, timeout=WAIT_MS)  # type: ignore[union-attr]


def test_fit_all_from_scratch(qtbot: QtBot, make_window: MakeWindow, depth_cache: Path) -> None:
    window, dialogs = make_window()
    window.open_cache(depth_cache)
    wait_open(qtbot, window)
    assert window.session.stored is None and "no results" in window.status_message()
    assert window.inspector.rows() == {} or "not fitted" in str(window.inspector.tree.topLevelItem(0).text(1))  # type: ignore[union-attr]
    assert window.start_fit_all()
    qtbot.waitUntil(lambda: not window.is_busy, timeout=WAIT_MS)
    assert not dialogs.errors
    assert window.session.has_results and "Fit All" in window.status_message()
    assert depth_cache.with_name("data_20260911_124617.depth.h5").is_file()
    qtbot.waitUntil(lambda: not window.data_pending, timeout=WAIT_MS)
    window.select_anode(STEEP)
    qtbot.waitUntil(
        lambda: window.depth_view.data is not None and window.depth_view.data.corrected,
        timeout=WAIT_MS,
    )


def test_depth_view_display_settings_persist(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path
) -> None:
    window, _ = make_window()
    window.open_cache(results_cache)
    wait_open(qtbot, window)
    window.depth_view.set_display(DepthDisplay(sources="cs", units="kev", ridges=True))
    window.depth_view.display_changed.emit()
    window.close()
    again, _ = make_window()
    display = again.depth_view.display()
    assert (display.sources, display.units, display.ridges) == ("cs", "kev", True)


def test_open_error_and_cli_help(qtbot: QtBot, make_window: MakeWindow, tmp_path: Path) -> None:
    window, dialogs = make_window()
    bogus = tmp_path / "bogus.cache.h5"
    bogus.write_text("x")
    window.open_cache(bogus)
    qtbot.waitUntil(lambda: bool(dialogs.errors), timeout=WAIT_MS)
    assert "not an HDF5" in dialogs.errors[0][1] and not window.session.is_open
    assert gui_main.__module__ == "dcalib.gui.main"


def test_ctrl_arrows_follow_the_focus(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path
) -> None:
    """Ctrl+Left/Right: Prev/Next anode, but a node step while the map's strip has the focus."""
    window, _ = make_window()
    window.open_cache(results_cache)
    wait_open(qtbot, window)
    first = window.session.selection
    assert first == CONCAVE
    window.tabs.setFocus()
    QTest.keyClick(window.tabs, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    after_next = window.session.selection
    assert after_next is not None and after_next == window.session.step_anode(first, 1)
    strip = window.system_map.board_strip
    strip.setFocus()
    # Node 4 has no events in the synthetic cache: the map moves there, no anode is selected.
    QTest.keyClick(strip, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert window.system_map.current_board == (after_next.node + 1, after_next.board)
    assert window.session.selection is None
    assert window.band.status_label.text().endswith("no events on this board")
    assert not window.fit_channel_action.isEnabled()
    # And back, at the same electrode.
    QTest.keyClick(strip, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: window.session.selection == after_next, timeout=WAIT_MS)
    assert window.band.selected_anode() == after_next


def test_boards_and_anodes_without_events(
    qtbot: QtBot, make_window: MakeWindow, results_cache: Path
) -> None:
    window, _ = make_window()
    window.open_cache(results_cache)
    wait_open(qtbot, window)
    window.system_map.board_selected.emit(9, 30)
    message = "Node 9 Board 30: no events on this board"
    assert window.session.selection is None and not window.data_pending
    assert window.band.status_label.text() == message
    assert window.band.selected_anode() is None
    assert window.inspector.title_label.text() == message
    assert window.depth_view.data is None and window.spectra_view.data is None
    # An anode of a board with events but without a 1A1C event of its own.
    fitted = set(window.session.anodes_on_board(3, 15))
    empty = next(
        AnodeKey(3, 15, rena, channel)
        for rena, channel in anode_channels(15)
        if AnodeKey(3, 15, rena, channel) not in fitted
    )
    window.select_anode(CONCAVE)
    wait_open(qtbot, window)
    window.select_anode(empty)
    assert window.session.selection is None and not window.data_pending
    assert window.band.status_label.text().endswith("no 1A1C events")
    assert window.board_grid.board == (3, 15)


def test_refit_tooltips_give_the_reason(qtbot: QtBot) -> None:
    band = ControlBand()
    qtbot.addWidget(band)
    tip = band.fit_board_button.toolTip()
    reason = "run Fit All first: there are no batch results to override"
    band.set_actions_enabled(refit=False, revert=False, reject=False, fit_all=True, reason=reason)
    assert band.fit_board_button.toolTip() == f"Unavailable: {reason}"
    band.set_actions_enabled(refit=True, revert=False, reject=False, fit_all=True)
    assert band.fit_board_button.toolTip() == tip
