"""The main window of ``dcalib-gui`` (plan section 9).

The shell follows uvcorr and adc2kev: a control band above the tabs
(:class:`~dcalib.gui.controls.ControlBand`), the tabs (Depth; the Spectra,
Board grid and Fleet summary tabs arrive in phase 7), and two docks: the
System Map (forked from uvcorr, :mod:`dcalib.gui.system_map`) and the Fit
Inspector. *File > Open* opens an adc2kev calibration cache (in a thread:
reading the calibrations takes ~1.5 s); stored, valid results in its sidecar
are shown at once. *Process > Fit All* runs :func:`~dcalib.analysis.analyze_all`
in a ``spawn`` pool from a worker thread and stores the batch.

Selecting an anode (map, selectors, Prev/Next) shows it in the band, the map
and the inspector at once, and loads its events in an
:class:`~dcalib.gui.threads.AnodeDataThread` (0.2-1.5 s when its board is not
loaded yet); a generation number drops the results of a superseded selection.
The session (:mod:`dcalib.gui.session`) owns all data; the window only wires
widgets, threads and the session together. The layout (geometry, docks), the
last directory, the map's colour mode and view and the Depth tab's display
settings persist in ``QSettings`` (:mod:`dcalib.gui._layout`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QAction, QCloseEvent, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dcalib import __version__
from dcalib.channels import AnodeKey
from dcalib.gui._layout import (
    FALLBACK_SCREEN,
    LAYOUT_VERSION,
    app_settings,
    initial_window_size,
    read_bool,
    read_bytes,
    read_int,
    read_str,
)
from dcalib.gui._system_map_model import ChannelAddress
from dcalib.gui.controls import ControlBand
from dcalib.gui.depth_view import DepthDisplay, DepthView
from dcalib.gui.inspector import FitInspector
from dcalib.gui.map_colors import COLOR_MODES, DEFAULT_INFORMATIONAL_FLAGS
from dcalib.gui.session import AnodeData, BatchOutcome, DepthSession, OpenedCache, anode_title
from dcalib.gui.system_map import VIEW_ANODES, VIEW_CATHODES, SystemMapWidget
from dcalib.gui.threads import AnodeDataThread, FitAllThread, OpenThread
from dcalib.options import DepthOptions

logger = logging.getLogger(__name__)

__all__ = ["TAB_TITLES", "MainWindow"]

TAB_DEPTH = "Depth"
TAB_TITLES: tuple[str, ...] = (TAB_DEPTH,)

KEY_LAYOUT_VERSION = "window/layout_version"
KEY_GEOMETRY = "window/geometry"
KEY_STATE = "window/state"
KEY_LAST_DIR = "files/last_directory"
KEY_COLOR_MODE = "map/color_mode"
KEY_MAP_VIEW = "map/view"
KEY_DEPTH_SOURCES = "depth/sources"
KEY_DEPTH_UNITS = "depth/units"
KEY_DEPTH_CURVES = "depth/source_curves"
KEY_DEPTH_RIDGES = "depth/ridges"
KEY_DEPTH_WINDOW = "depth/window"

_MAP_HEIGHT_FRACTION = 0.38
_INSPECTOR_WIDTH_FRACTION = 0.22


class MainWindow(QMainWindow):
    """The dcalib GUI (see the module docstring)."""

    def __init__(
        self,
        workers: int | None = None,
        results_path: str | Path | None = None,
        kev_path: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = DepthSession()
        self.workers = workers
        self._results_path = Path(results_path) if results_path is not None else None
        self._kev_path = Path(kev_path) if kev_path is not None else None
        self._open_thread: OpenThread | None = None
        self._fit_thread: FitAllThread | None = None
        self._data_thread: AnodeDataThread | None = None
        self._generation = 0
        self._retired: list[QThread] = []
        self.setWindowTitle("dcalib-gui")

        self._init_central()
        self._create_actions()
        self._create_menus()
        self._create_status_bar()
        self._create_docks()
        self._restore_settings()
        self._update_actions()

    # -- construction ------------------------------------------------------

    def _init_central(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.band = ControlBand()
        layout.addWidget(self.band)
        self.tabs = QTabWidget()
        self.depth_view = DepthView()
        self.tabs.addTab(self.depth_view, TAB_DEPTH)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.band.anode_requested.connect(self.select_anode)
        self.band.step_requested.connect(self.step_anode)
        self.band.options_changed.connect(self._on_options_changed)
        self.band.fit_all_clicked.connect(self.start_fit_all)
        self.depth_view.display_changed.connect(self._save_view_settings)

    def _action(
        self,
        text: str,
        slot: Callable[[], object],
        shortcut: str | QKeySequence | QKeySequence.StandardKey | None = None,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        return action

    def _create_actions(self) -> None:
        self.open_action = self._action(
            "&Open Cache...", self._on_open, QKeySequence.StandardKey.Open
        )
        self.quit_action = self._action("&Quit", self.close, QKeySequence.StandardKey.Quit)
        self.fit_all_action = self._action("Fit &All", self.start_fit_all, "Ctrl+F")
        self.stop_action = self._action("&Stop", self.stop_current, "Esc")
        self.prev_action = self._action("Previous Anode", lambda: self.step_anode(-1), "Ctrl+Left")
        self.next_action = self._action("Next Anode", lambda: self.step_anode(1), "Ctrl+Right")
        self.reset_layout_action = self._action("Reset Layout", self.reset_layout)
        self.about_action = self._action("About dcalib-gui", self._show_about)
        for action in (self.prev_action, self.next_action):
            self.addAction(action)

    def _create_menus(self) -> None:
        bar = self.menuBar()
        assert bar is not None
        file_menu = bar.addMenu("&File")
        assert file_menu is not None
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        process = bar.addMenu("&Process")
        assert process is not None
        process.addAction(self.fit_all_action)
        process.addAction(self.stop_action)
        self.view_menu = bar.addMenu("&View")
        help_menu = bar.addMenu("&Help")
        assert help_menu is not None
        help_menu.addAction(self.about_action)

    def _create_status_bar(self) -> None:
        status = self.statusBar()
        assert status is not None
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 100)
        self.progress.hide()
        status.addPermanentWidget(self.progress)

    def _create_docks(self) -> None:
        self.system_map = SystemMapWidget()
        self.system_map.set_informational_flags(DEFAULT_INFORMATIONAL_FLAGS)
        self.system_map.channel_selected.connect(self._on_map_channel_selected)
        self.system_map.board_selected.connect(self._on_map_board_selected)
        self.system_map.color_mode_changed.connect(lambda _mode: self._save_view_settings())
        self.system_map.view_changed.connect(lambda _view: self._save_view_settings())
        self.system_map_dock = QDockWidget("System Map", self)
        self.system_map_dock.setObjectName("system_map_dock")
        self.system_map_dock.setWidget(self.system_map)
        self.inspector = FitInspector()
        self.inspector_dock = QDockWidget("Fit Inspector", self)
        self.inspector_dock.setObjectName("inspector_dock")
        self.inspector_dock.setWidget(self.inspector)
        assert self.view_menu is not None
        for dock in (self.system_map_dock, self.inspector_dock):
            self.view_menu.addAction(dock.toggleViewAction())
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.reset_layout_action)
        self._apply_default_docks()

    def _apply_default_docks(self) -> None:
        for dock, area in (
            (self.system_map_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
            (self.inspector_dock, Qt.DockWidgetArea.RightDockWidgetArea),
        ):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()

    def _apply_default_size(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            available = FALLBACK_SCREEN
        else:
            rect = screen.availableGeometry()
            available = (rect.width(), rect.height())
        hint = self.minimumSizeHint()
        (width, height), maximize = initial_window_size(available, (hint.width(), hint.height()))
        if maximize:
            self.resize(*available)
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
            width, height = available
        else:
            self.resize(width, height)
        self.resizeDocks(
            [self.system_map_dock], [round(height * _MAP_HEIGHT_FRACTION)], Qt.Orientation.Vertical
        )
        self.resizeDocks(
            [self.inspector_dock],
            [round(width * _INSPECTOR_WIDTH_FRACTION)],
            Qt.Orientation.Horizontal,
        )

    def _restore_settings(self) -> None:
        settings = app_settings()
        geometry = read_bytes(settings, KEY_GEOMETRY)
        state = read_bytes(settings, KEY_STATE)
        restored = False
        if read_int(settings, KEY_LAYOUT_VERSION) == LAYOUT_VERSION and not state.isEmpty():
            restored = bool(self.restoreState(state, LAYOUT_VERSION))
            if restored and not self.restoreGeometry(geometry):
                self._apply_default_size()
        if not restored:
            self._apply_default_docks()
            self._apply_default_size()
        mode = read_str(settings, KEY_COLOR_MODE)
        if mode in COLOR_MODES:
            self.system_map.set_color_mode(mode)
        view = read_str(settings, KEY_MAP_VIEW)
        if view in (VIEW_ANODES, VIEW_CATHODES):
            self.system_map.set_view(view)
        self.depth_view.set_display(
            DepthDisplay(
                sources=read_str(settings, KEY_DEPTH_SOURCES, "both") or "both",
                units=read_str(settings, KEY_DEPTH_UNITS, "x") or "x",
                source_curves=read_bool(settings, KEY_DEPTH_CURVES, True),
                ridges=read_bool(settings, KEY_DEPTH_RIDGES, False),
                window=read_bool(settings, KEY_DEPTH_WINDOW, True),
            )
        )

    def _save_view_settings(self) -> None:
        settings = app_settings()
        settings.setValue(KEY_COLOR_MODE, self.system_map.color_mode)
        settings.setValue(KEY_MAP_VIEW, self.system_map.view)
        display = self.depth_view.display()
        settings.setValue(KEY_DEPTH_SOURCES, display.sources)
        settings.setValue(KEY_DEPTH_UNITS, display.units)
        settings.setValue(KEY_DEPTH_CURVES, display.source_curves)
        settings.setValue(KEY_DEPTH_RIDGES, display.ridges)
        settings.setValue(KEY_DEPTH_WINDOW, display.window)

    def save_settings(self) -> None:
        settings = app_settings()
        settings.setValue(KEY_LAYOUT_VERSION, LAYOUT_VERSION)
        settings.setValue(KEY_GEOMETRY, self.saveGeometry())
        settings.setValue(KEY_STATE, self.saveState(LAYOUT_VERSION))
        self._save_view_settings()

    def reset_layout(self) -> None:
        self._apply_default_docks()
        self._apply_default_size()

    # -- helpers -------------------------------------------------------------

    def _status(self, message: str, timeout_ms: int = 0) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, timeout_ms)

    def status_message(self) -> str:
        bar = self.statusBar()
        return bar.currentMessage() if bar is not None else ""

    def _show_error(self, title: str, text: str) -> None:
        self._status(text)
        QMessageBox.critical(self, title, text)

    @property
    def is_busy(self) -> bool:
        return self._open_thread is not None or self._fit_thread is not None

    def _update_actions(self) -> None:
        is_open = self.session.is_open
        busy = self.is_busy
        self.open_action.setEnabled(not busy)
        self.fit_all_action.setEnabled(is_open and not busy)
        self.stop_action.setEnabled(busy)
        self.band.fit_all_button.setEnabled(is_open and not busy)

    def _update_title(self) -> None:
        path = self.session.cache_path
        self.setWindowTitle(f"dcalib-gui - {path.name}" if path is not None else "dcalib-gui")

    def _show_progress(self, percent: int) -> None:
        self.progress.setValue(max(0, min(100, percent)))
        self.progress.show()

    def _retire(self, thread: QThread | None) -> None:
        """Keep a finished thread referenced until it has really stopped."""
        if thread is None:
            return
        self._retired = [t for t in self._retired if t.isRunning()]
        if thread.isRunning():
            self._retired.append(thread)

    # -- opening -----------------------------------------------------------

    def last_directory(self) -> str:
        return read_str(app_settings(), KEY_LAST_DIR, str(Path.home())) or str(Path.home())

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open adc2kev calibration cache",
            self.last_directory(),
            "adc2kev cache (*.cache.h5 *.h5);;All files (*)",
        )
        if path:
            self.open_cache(path)

    def open_cache(self, path: str | Path) -> bool:
        """Open a cache in a thread (the results path and ``.kev`` given at start-up apply)."""
        if self.is_busy:
            self._status("Busy: wait for the current operation to finish", 5000)
            return False
        cache = Path(path)
        app_settings().setValue(KEY_LAST_DIR, str(cache.parent))
        thread = OpenThread(cache, self._results_path, self._kev_path, self)
        thread.finished.connect(self._on_open_finished)
        thread.error.connect(self._on_open_error)
        self._open_thread = thread
        self._status(f"Opening {cache.name}...")
        self._update_actions()
        thread.start()
        return True

    def _end_open_thread(self) -> None:
        self._retire(self._open_thread)
        self._open_thread = None
        self._update_actions()

    def _on_open_finished(self, opened: OpenedCache) -> None:
        self._end_open_thread()
        self._drop_data_load()
        self.session.install(opened)
        self.band.set_options(self.session.options)
        self._refresh_all()
        self._update_title()
        self._status(self.session.describe())
        if self.session.stale and self.session.validity is not None:
            self._status(
                "Stored results are stale ("
                + "; ".join(self.session.validity.reasons)
                + "): run Fit All"
            )
        first = self.session.step_anode(None, 1)
        if first is not None:
            self.select_anode(first)

    def _on_open_error(self, message: str) -> None:
        self._end_open_thread()
        self._show_error("Cannot open the cache", message)

    def _refresh_all(self) -> None:
        """Rebuild the map and the selectors from the session."""
        views = self.session.views()
        # With results, an anode without a row has no 1A1C event (no data), not "not fitted".
        data_channels = list(views) if self.session.has_results else None
        self.system_map.set_state(
            views, active_boards=self.session.boards(), data_channels=data_channels
        )
        self.band.set_anodes(
            {nb: self.session.anodes_on_board(*nb) for nb in self.session.boards()}
        )

    # -- selection -----------------------------------------------------------

    def select_anode(self, requested: AnodeKey | tuple[int, int, int, int]) -> None:
        """Show an anode everywhere and load its events for the Depth tab."""
        selected = self.session.select(requested)
        if selected is None:
            return
        key = selected
        result = self.session.result(key)
        self.band.set_anode(
            key,
            result.status if result is not None else None,
            result.flags if result is not None else (),
            result.review if result is not None else "",
        )
        self.system_map.set_selection(key.node, key.board, key)
        note = ""
        stored = self.session.stored
        if stored is not None and key in stored.review:
            note = stored.review[key].note
        self.inspector.show_anode(key, result, self.session.options_for(key), note)
        self._request_data(key)

    def step_anode(self, delta: int) -> None:
        key = self.session.step_anode(self.session.selection, delta)
        if key is not None:
            self.select_anode(key)

    def _on_map_channel_selected(self, channel: ChannelAddress) -> None:
        view = self.session.views().get(AnodeKey(*channel))
        if view is not None and view.is_cathode:
            self._status(f"{anode_title(channel)}: cathode, keV calibration {view.status}", 5000)
            return
        self.select_anode(AnodeKey(*channel))

    def _on_map_board_selected(self, node: int, board: int) -> None:
        current = self.session.selection
        if current is not None and (current.node, current.board) == (node, board):
            return
        anodes = self.session.anodes_on_board(node, board)
        if anodes:
            self.select_anode(anodes[0])

    def _on_options_changed(self, options: DepthOptions) -> None:
        self.session.options = options

    @property
    def data_pending(self) -> bool:
        return self._data_thread is not None

    def _drop_data_load(self) -> None:
        self._generation += 1
        if self._data_thread is not None:
            self._data_thread.blockSignals(True)
            self._retire(self._data_thread)
            self._data_thread = None

    def _request_data(self, key: AnodeKey) -> None:
        self._drop_data_load()
        self.depth_view.set_loading(anode_title(key))
        thread = AnodeDataThread(self.session, key, self._generation, self)
        thread.done.connect(self._on_data_done)
        thread.failed.connect(self._on_data_failed)
        self._data_thread = thread
        thread.start()

    def _on_data_done(self, generation: int, data: AnodeData) -> None:
        if generation != self._generation:
            return
        self._retire(self._data_thread)
        self._data_thread = None
        self.depth_view.set_data(data)

    def _on_data_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._retire(self._data_thread)
        self._data_thread = None
        self.depth_view.clear(message)
        self._status(message, 8000)

    # -- Fit All -----------------------------------------------------------------

    def start_fit_all(self) -> bool:
        """Run the batch analysis with the band's options (a spawn pool, in a thread)."""
        if not self.session.is_open or self.is_busy:
            return False
        options = self.band.options
        thread = FitAllThread(self.session, options, self.workers, self)
        thread.progress.connect(self._on_fit_progress)
        thread.finished.connect(self._on_fit_finished)
        thread.error.connect(self._on_fit_error)
        thread.stopped.connect(self._on_fit_stopped)
        self._fit_thread = thread
        self._show_progress(0)
        self._status("Fit All: analysing every board...")
        self._update_actions()
        thread.start()
        return True

    def _end_fit_thread(self) -> None:
        self._retire(self._fit_thread)
        self._fit_thread = None
        self.progress.hide()
        self._update_actions()

    def _on_fit_progress(self, done: int, total: int) -> None:
        self._show_progress(int(100 * done / total) if total else 100)

    def _on_fit_finished(self, outcome: BatchOutcome) -> None:
        self._end_fit_thread()
        self.session.apply_batch(outcome)
        self._refresh_all()
        self._status(outcome.describe())
        if self.session.selection is not None:
            self.select_anode(self.session.selection)

    def _on_fit_error(self, message: str) -> None:
        self._end_fit_thread()
        self._show_error("Fit All failed", message)

    def _on_fit_stopped(self) -> None:
        self._end_fit_thread()
        self._status("Fit All stopped; nothing was stored", 8000)

    def stop_current(self) -> None:
        if self._fit_thread is not None:
            self._fit_thread.stop()
            self._status("Stopping Fit All...")

    # -- misc ------------------------------------------------------------------

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About dcalib-gui",
            f"dcalib {__version__}\n\nCZT depth-of-interaction photopeak calibration from the "
            "adc2kev calibration cache.",
        )

    def request_quit(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Stop running work, wait for the threads, save the layout."""
        if self._fit_thread is not None:
            self._fit_thread.stop()
        for thread in (self._open_thread, self._fit_thread, self._data_thread, *self._retired):
            if thread is not None:
                thread.blockSignals(True)
                thread.wait(60_000)
        self.save_settings()
        super().closeEvent(event)
