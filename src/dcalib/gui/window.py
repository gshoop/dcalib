"""The main window of ``dcalib-gui`` (plan section 9).

The shell follows uvcorr and adc2kev: a control band above the tabs
(:class:`~dcalib.gui.controls.ControlBand`), four tabs (Depth, Spectra, Board
grid, Fleet summary) and two docks, the System Map (forked from uvcorr,
:mod:`dcalib.gui.system_map`) and the Fit Inspector.

- *File > Open* opens an adc2kev calibration cache in a thread (reading the
  calibrations takes ~1.5 s); stored, valid results in its sidecar are shown
  at once. Stale results (another cache or calibration) are shown with a
  banner that offers Fit All; they cannot be re-fitted.
- *Process > Fit All* runs :func:`~dcalib.analysis.analyze_all` in a ``spawn``
  pool from a worker thread and stores the batch. *Fit Channel* / *Fit Board*
  re-fit with the control band's options, stored as per-anode overrides
  (options equal to the batch's revert the anodes instead); *Revert to batch*
  deletes an anode's or a board's overrides. The System Map's context menu
  offers the same actions.
- **Reject** (band, context menu) rejects an anode with an optional note;
  rejected anodes are left out of the exported ``.dcc``. Reviews persist in the
  sidecar and survive new batches.
- *File > Export* writes ``<name>.dcc`` and ``depth_summary.csv`` of the merged
  results (overrides and review applied) to a chosen directory.

Selecting an anode (map, selectors, Prev/Next, Board grid, Fleet table) shows
it in the band, the map and the inspector at once, and loads its events for
the Depth and Spectra tabs in an :class:`~dcalib.gui.threads.AnodeDataThread`
(0.2-1.5 s when its board is not loaded yet); a generation number drops the
results of a superseded selection. The session (:mod:`dcalib.gui.session`)
owns all data; the window only wires widgets, threads and the session
together. The layout, the last directories, the map's colour mode and view
and the tabs' display settings persist in ``QSettings``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
from dcalib.gui._system_map_model import ChannelAddress, ChannelTuple, as_address
from dcalib.gui.board_grid import BoardGridView
from dcalib.gui.controls import ControlBand
from dcalib.gui.depth_view import DepthDisplay, DepthView
from dcalib.gui.fleet import FleetView
from dcalib.gui.inspector import FitInspector
from dcalib.gui.map_colors import COLOR_MODES, DEFAULT_INFORMATIONAL_FLAGS
from dcalib.gui.session import (
    AnodeData,
    BatchOutcome,
    DepthSession,
    ExportSummary,
    OpenedCache,
    RefitOutcome,
    SessionError,
    anode_title,
)
from dcalib.gui.spectra import SpectraView
from dcalib.gui.system_map import VIEW_ANODES, VIEW_CATHODES, SystemMapWidget
from dcalib.gui.threads import AnodeDataThread, FitAllThread, OpenThread, RefitThread
from dcalib.options import DepthOptions

logger = logging.getLogger(__name__)

__all__ = ["TAB_TITLES", "MainWindow", "ReviewSystemMap"]

TAB_DEPTH = "Depth"
TAB_SPECTRA = "Spectra"
TAB_BOARD_GRID = "Board grid"
TAB_FLEET = "Fleet summary"
TAB_TITLES: tuple[str, ...] = (TAB_DEPTH, TAB_SPECTRA, TAB_BOARD_GRID, TAB_FLEET)

KEY_LAYOUT_VERSION = "window/layout_version"
KEY_GEOMETRY = "window/geometry"
KEY_STATE = "window/state"
KEY_LAST_DIR = "files/last_directory"
KEY_EXPORT_DIR = "files/export_directory"
KEY_COLOR_MODE = "map/color_mode"
KEY_MAP_VIEW = "map/view"
KEY_DEPTH_SOURCES = "depth/sources"
KEY_DEPTH_UNITS = "depth/units"
KEY_DEPTH_CURVES = "depth/source_curves"
KEY_DEPTH_RIDGES = "depth/ridges"
KEY_DEPTH_WINDOW = "depth/window"
KEY_SPECTRA_LOG = "spectra/log_scale"

_MAP_HEIGHT_FRACTION = 0.38
_INSPECTOR_WIDTH_FRACTION = 0.22


class ReviewSystemMap(SystemMapWidget):
    """The System Map with the review and revert actions in its context menu.

    Signals:
        revert_channel_requested(object), revert_board_requested(int, int),
        reject_requested(object): the extra menu actions.
    """

    revert_channel_requested = pyqtSignal(object)
    revert_board_requested = pyqtSignal(int, int)
    reject_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.can_revert_channel: Callable[[AnodeKey], bool] = lambda _key: False
        self.can_revert_board: Callable[[int, int], bool] = lambda _node, _board: False
        self.is_rejected: Callable[[AnodeKey], bool | None] = lambda _key: None

    def build_context_menu(self, node: int, board: int, channel: ChannelTuple | None) -> QMenu:
        menu = super().build_context_menu(node, board, channel)
        menu.addSeparator()
        if channel is not None:
            address = as_address(channel)
            key = AnodeKey(*address)
            rejected = self.is_rejected(key)
            if rejected is not None:
                reject = QAction("Unreject" if rejected else "Reject...", menu)
                reject.triggered.connect(lambda: self.reject_requested.emit(key))
                menu.addAction(reject)
            if self.can_revert_channel(key):
                revert = QAction("Revert Channel to batch", menu)
                revert.triggered.connect(lambda: self.revert_channel_requested.emit(address))
                menu.addAction(revert)
        if self.can_revert_board(node, board):
            revert_board = QAction(f"Revert Board  Node {node} Board {board}", menu)
            revert_board.triggered.connect(lambda: self.revert_board_requested.emit(node, board))
            menu.addAction(revert_board)
        return menu


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
        self._refit_thread: RefitThread | None = None
        self._data_thread: AnodeDataThread | None = None
        self._generation = 0
        self._retired: list[QThread] = []
        self._grid_board: tuple[int, int] | None = None
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
        self.stale_banner = QWidget()
        banner_layout = QHBoxLayout(self.stale_banner)
        banner_layout.setContentsMargins(6, 4, 6, 4)
        self.stale_label = QLabel("")
        self.stale_label.setWordWrap(True)
        self.stale_fit_button = QPushButton("Fit All")
        banner_layout.addWidget(self.stale_label, 1)
        banner_layout.addWidget(self.stale_fit_button)
        self.stale_banner.setStyleSheet("background: #7a3e00; color: #ffffff; font-weight: bold;")
        self.stale_banner.hide()
        layout.addWidget(self.stale_banner)
        self.band = ControlBand()
        layout.addWidget(self.band)
        self.tabs = QTabWidget()
        self.depth_view = DepthView()
        self.spectra_view = SpectraView()
        self.board_grid = BoardGridView()
        self.fleet_view = FleetView()
        for widget, title in zip(
            (self.depth_view, self.spectra_view, self.board_grid, self.fleet_view), TAB_TITLES
        ):
            self.tabs.addTab(widget, title)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.stale_fit_button.clicked.connect(self.start_fit_all)
        self.band.anode_requested.connect(self.select_anode)
        self.band.step_requested.connect(self.step_anode)
        self.band.options_changed.connect(self._on_options_changed)
        self.band.fit_all_clicked.connect(self.start_fit_all)
        self.band.fit_channel_clicked.connect(lambda: self.refit_channel())
        self.band.fit_board_clicked.connect(lambda: self.refit_board())
        self.band.revert_clicked.connect(lambda: self.revert_channel())
        self.band.reject_toggled.connect(self._on_reject_toggled)
        self.depth_view.display_changed.connect(self._save_view_settings)
        self.spectra_view.display_changed.connect(self._save_view_settings)
        self.board_grid.anode_activated.connect(self.select_anode)
        self.fleet_view.anode_activated.connect(self.select_anode)

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
        self.export_action = self._action("&Export .dcc and CSV...", self._on_export, "Ctrl+E")
        self.quit_action = self._action("&Quit", self.close, QKeySequence.StandardKey.Quit)
        self.fit_all_action = self._action("Fit &All", self.start_fit_all, "Ctrl+F")
        self.fit_channel_action = self._action(
            "Fit &Channel", lambda: self.refit_channel(), "Ctrl+R"
        )
        self.fit_board_action = self._action(
            "Fit &Board", lambda: self.refit_board(), "Ctrl+Shift+R"
        )
        self.revert_action = self._action("&Revert Channel to batch", lambda: self.revert_channel())
        self.revert_board_action = self._action(
            "Revert Board to batch", lambda: self.revert_board()
        )
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
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        process = bar.addMenu("&Process")
        assert process is not None
        for action in (
            self.fit_all_action,
            self.fit_channel_action,
            self.fit_board_action,
            self.revert_action,
            self.revert_board_action,
        ):
            process.addAction(action)
        process.addSeparator()
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
        self.system_map = ReviewSystemMap()
        self.system_map.set_informational_flags(DEFAULT_INFORMATIONAL_FLAGS)
        self.system_map.can_revert_channel = self.session.has_override
        self.system_map.can_revert_board = lambda n, b: bool(self.session.override_keys(n, b))
        self.system_map.is_rejected = self._rejected_state
        self.system_map.channel_selected.connect(self._on_map_channel_selected)
        self.system_map.board_selected.connect(self._on_map_board_selected)
        self.system_map.fit_channel_requested.connect(
            lambda channel: self.refit_channel(AnodeKey(*channel))
        )
        self.system_map.fit_board_requested.connect(lambda n, b: self.refit_board(n, b))
        self.system_map.revert_channel_requested.connect(
            lambda channel: self.revert_channel(AnodeKey(*channel))
        )
        self.system_map.revert_board_requested.connect(lambda n, b: self.revert_board(n, b))
        self.system_map.reject_requested.connect(self._on_map_reject)
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
        self.spectra_view.set_log_scale(read_bool(settings, KEY_SPECTRA_LOG, False))

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
        settings.setValue(KEY_SPECTRA_LOG, self.spectra_view.log_scale())

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

    def _ask_note(self, key: AnodeKey) -> str | None:
        """Ask for the note of a rejection; None cancels it (replaced in tests)."""
        text, ok = QInputDialog.getText(
            self, "Reject anode", f"Reject {anode_title(key)}. Note (optional):"
        )
        return text if ok else None

    def _choose_directory(self, title: str, start: str) -> str:
        """A directory chooser (replaced in tests)."""
        return QFileDialog.getExistingDirectory(self, title, start)

    @property
    def is_busy(self) -> bool:
        return (
            self._open_thread is not None
            or self._fit_thread is not None
            or self._refit_thread is not None
        )

    def _update_actions(self) -> None:
        session = self.session
        is_open = session.is_open
        busy = self.is_busy
        refit_reason = session.refit_blocked_reason()
        can_refit = not refit_reason and not busy and session.selection is not None
        key = session.selection
        can_revert = key is not None and session.has_override(key) and not busy
        can_reject = key is not None and session.has_results and session.result(key) is not None
        self.open_action.setEnabled(not busy)
        self.fit_all_action.setEnabled(is_open and not busy)
        self.fit_channel_action.setEnabled(can_refit)
        self.fit_board_action.setEnabled(can_refit)
        self.revert_action.setEnabled(can_revert)
        board = (key.node, key.board) if key is not None else None
        self.revert_board_action.setEnabled(
            board is not None and bool(session.override_keys(*board)) and not busy
        )
        self.export_action.setEnabled(session.has_results and not busy)
        self.stop_action.setEnabled(busy)
        self.band.set_actions_enabled(
            refit=can_refit,
            revert=can_revert,
            reject=can_reject and not busy,
            fit_all=is_open and not busy,
            reason=refit_reason,
        )
        self.stale_fit_button.setEnabled(is_open and not busy)

    def _update_title(self) -> None:
        path = self.session.cache_path
        self.setWindowTitle(f"dcalib-gui - {path.name}" if path is not None else "dcalib-gui")

    def _update_stale_banner(self) -> None:
        validity = self.session.validity
        if self.session.stale and validity is not None:
            self.stale_label.setText(
                "The stored results are stale: "
                + "; ".join(validity.reasons)
                + ". They are shown as stored; overrides do not apply and re-fits are "
                "disabled until Fit All replaces them. Rejections are kept."
            )
            self.stale_banner.show()
        else:
            self.stale_banner.hide()

    @property
    def stale_banner_visible(self) -> bool:
        return not self.stale_banner.isHidden()

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

    def _rejected_state(self, key: AnodeKey) -> bool | None:
        result = self.session.result(key) if self.session.has_results else None
        return None if result is None else bool(result.review)

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
        self._grid_board = None
        self._refresh_all()
        self._update_title()
        self._status(self.session.describe())
        first = self.session.step_anode(None, 1)
        if first is not None:
            self.select_anode(first)

    def _on_open_error(self, message: str) -> None:
        self._end_open_thread()
        self._show_error("Cannot open the cache", message)

    def _refresh_all(self) -> None:
        """Rebuild the map, the selectors, the fleet summary and the banner from the session."""
        views = self.session.views()
        # With results, an anode without a row has no 1A1C event (no data), not "not fitted".
        data_channels = list(views) if self.session.has_results else None
        self.system_map.set_state(
            views, active_boards=self.session.boards(), data_channels=data_channels
        )
        self.band.set_anodes(
            {nb: self.session.anodes_on_board(*nb) for nb in self.session.boards()}
        )
        if self.session.has_results:
            self.fleet_view.set_results(self.session.merged_results())
        else:
            self.fleet_view.clear()
        self._grid_board = None
        self._update_stale_banner()
        self._update_actions()

    # -- selection -----------------------------------------------------------

    def select_anode(self, requested: AnodeKey | tuple[int, int, int, int]) -> None:
        """Show an anode everywhere and load its events for the Depth and Spectra tabs."""
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
        entry = self.session.review(key)
        self.inspector.show_anode(
            key, result, self.session.options_for(key), entry.note if entry is not None else ""
        )
        self._update_board_grid(key)
        self._update_actions()
        self._request_data(key)

    def _update_board_grid(self, key: AnodeKey | None = None) -> None:
        key = key if key is not None else self.session.selection
        if key is None:
            self.board_grid.clear()
            return
        self.board_grid.set_board(
            key.node, key.board, self.session.board_grid(key.node, key.board), key
        )
        self._grid_board = (key.node, key.board)

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
        self.spectra_view.set_data(data)

    def _on_data_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._retire(self._data_thread)
        self._data_thread = None
        self.depth_view.clear(message)
        self.spectra_view.clear(message)
        self._status(message, 8000)

    def _after_results_change(self, message: str) -> None:
        """Refresh everything after the stored results changed, keeping the selection."""
        self._refresh_all()
        self._status(message)
        if self.session.selection is not None:
            self.select_anode(self.session.selection)

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
        self._after_results_change(outcome.describe())

    def _on_fit_error(self, message: str) -> None:
        self._end_fit_thread()
        self._show_error("Fit All failed", message)

    def _on_fit_stopped(self) -> None:
        self._end_fit_thread()
        self._status("Fit All stopped; nothing was stored", 8000)

    def stop_current(self) -> None:
        for thread in (self._fit_thread, self._refit_thread):
            if thread is not None:
                thread.stop()
        self._status("Stopping...")

    # -- re-fits, reverts, review --------------------------------------------------

    def refit_channel(self, key: AnodeKey | None = None) -> bool:
        """Re-fit one anode (default: the selected one) with the band's options."""
        key = key if key is not None else self.session.selection
        if key is None:
            return False
        return self._start_refit(key.node, key.board, ((key.rena, key.channel),))

    def refit_board(self, node: int | None = None, board: int | None = None) -> bool:
        """Re-fit every anode of a board (default: the selected anode's board)."""
        if node is None or board is None:
            key = self.session.selection
            if key is None:
                return False
            node, board = key.node, key.board
        return self._start_refit(node, board, None)

    def _start_refit(
        self, node: int, board: int, anodes: tuple[tuple[int, int], ...] | None
    ) -> bool:
        if self.is_busy:
            self._status("Busy: wait for the current operation to finish", 5000)
            return False
        try:
            request = self.session.refit_request(node, board, anodes, self.band.options)
        except SessionError as exc:
            self._show_error("Cannot re-fit", str(exc))
            return False
        thread = RefitThread(self.session, request, self)
        thread.finished.connect(self._on_refit_finished)
        thread.error.connect(self._on_refit_error)
        thread.stopped.connect(self._on_refit_stopped)
        self._refit_thread = thread
        self._status(request.describe_start())
        self._update_actions()
        thread.start()
        return True

    def _end_refit_thread(self) -> None:
        self._retire(self._refit_thread)
        self._refit_thread = None
        self._update_actions()

    def _on_refit_finished(self, outcome: RefitOutcome) -> None:
        self._end_refit_thread()
        self.session.apply_refit(outcome)
        self._after_results_change(outcome.describe())

    def _on_refit_error(self, message: str) -> None:
        self._end_refit_thread()
        self._show_error("Re-fit failed", message)

    def _on_refit_stopped(self) -> None:
        self._end_refit_thread()
        self._status("Re-fit stopped; nothing was stored", 8000)

    def revert_channel(self, key: AnodeKey | None = None) -> bool:
        """Delete an anode's override (default: the selected anode's)."""
        key = key if key is not None else self.session.selection
        if key is None or self.is_busy:
            return False
        return self._revert([key], anode_title(key))

    def revert_board(self, node: int | None = None, board: int | None = None) -> bool:
        """Delete every override of a board (default: the selected anode's board)."""
        if node is None or board is None:
            key = self.session.selection
            if key is None:
                return False
            node, board = key.node, key.board
        if self.is_busy:
            return False
        return self._revert(self.session.override_keys(node, board), f"board n{node} b{board}")

    def _revert(self, keys: list[AnodeKey], what: str) -> bool:
        try:
            deleted = self.session.revert(keys)
        except SessionError as exc:
            self._show_error("Revert failed", str(exc))
            return False
        if not deleted:
            self._status(f"{what}: no override to revert", 5000)
            return False
        self._after_results_change(f"Reverted {deleted} override(s) of {what} to the batch")
        return True

    def set_rejected(self, key: AnodeKey, rejected: bool, note: str = "") -> bool:
        """Reject or restore an anode and refresh the views."""
        try:
            self.session.set_review(key, rejected, note)
        except SessionError as exc:
            self._show_error("Review failed", str(exc))
            self.band.set_rejected(self._rejected_state(key) or False)
            return False
        state = "rejected" if rejected else "restored"
        self._after_results_change(f"{anode_title(key)} {state}")
        return True

    def _on_reject_toggled(self, checked: bool) -> None:
        key = self.session.selection
        if key is None:
            return
        note = ""
        if checked:
            asked = self._ask_note(key)
            if asked is None:
                self.band.set_rejected(False)
                return
            note = asked
        self.set_rejected(key, checked, note)

    def _on_map_reject(self, key: AnodeKey) -> None:
        rejected = self._rejected_state(key)
        if rejected is None:
            return
        if rejected:
            self.set_rejected(key, False)
            return
        note = self._ask_note(key)
        if note is not None:
            self.set_rejected(key, True, note)

    # -- export ---------------------------------------------------------------------

    def _on_export(self) -> None:
        start = read_str(app_settings(), KEY_EXPORT_DIR, self.last_directory()) or ""
        directory = self._choose_directory("Export .dcc and depth_summary.csv to", start)
        if directory:
            self.export_outputs(directory)

    def export_outputs(self, directory: str | Path) -> ExportSummary | None:
        """Write ``<name>.dcc`` and ``depth_summary.csv`` of the merged results."""
        try:
            summary = self.session.export_outputs(directory)
        except SessionError as exc:
            self._show_error("Export failed", str(exc))
            return None
        app_settings().setValue(KEY_EXPORT_DIR, str(Path(directory)))
        self._status(summary.describe())
        return summary

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
        for worker in (self._fit_thread, self._refit_thread):
            if worker is not None:
                worker.stop()
        running: list[QThread | None] = [
            self._open_thread,
            self._fit_thread,
            self._refit_thread,
            self._data_thread,
            *self._retired,
        ]
        for thread in running:
            if thread is not None:
                thread.blockSignals(True)
                thread.wait(60_000)
        self.save_settings()
        super().closeEvent(event)
