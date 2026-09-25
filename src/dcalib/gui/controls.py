"""The control band above the tabs (plan section 9).

- **Row 1**: Prev/Next, the node, board and anode selectors, the anode's status
  (in its map colour, readable on the palette) and flags.
- **Row 2**: the depth-fit options the next fit uses (sources, the photopeak
  window ``x_lo..x_hi``, the maximum degree, the minimum gain) and Fit All.

The band only reports what the user asked for (signals) and shows what the
main window tells it; it never touches the session.

Signals:
    anode_requested(object): an :class:`~dcalib.channels.AnodeKey` chosen in the
        selectors.
    step_requested(int): Prev (-1) or Next (+1).
    options_changed(object): the options fields changed (a ``DepthOptions``).
    fit_all_clicked(): the Fit All button.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dcalib.channels import AnodeKey, electrode_label
from dcalib.gui._contrast import readable_color
from dcalib.gui.map_colors import CATEGORY_COLORS, CATEGORY_NOT_FITTED, status_category
from dcalib.options import SOURCE_CHOICES, DepthOptions

__all__ = ["ControlBand"]

_SOURCE_LABELS = {"both": "Ge + Cs", "ge": "Ge-68", "cs": "Cs-137"}


class ControlBand(QWidget):
    """The two rows of controls (see the module docstring)."""

    anode_requested = pyqtSignal(object)
    step_requested = pyqtSignal(int)
    options_changed = pyqtSignal(object)
    fit_all_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._boards: list[tuple[int, int]] = []
        self._anodes: dict[tuple[int, int], list[AnodeKey]] = {}
        self._options = DepthOptions()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)
        row1 = QHBoxLayout()
        item: QWidget
        self.prev_button = QPushButton("◀ Prev")
        self.next_button = QPushButton("Next ▶")
        self.prev_button.setToolTip("Previous anode (Ctrl+Left)")
        self.next_button.setToolTip("Next anode (Ctrl+Right)")
        self.node_combo = QComboBox()
        self.board_combo = QComboBox()
        self.anode_combo = QComboBox()
        self.anode_combo.setMinimumContentsLength(14)
        self.status_label = QLabel("")
        self.flags_label = QLabel("")
        self.flags_label.setWordWrap(False)
        for item in (
            self.prev_button,
            self.next_button,
            QLabel("Node"),
            self.node_combo,
            QLabel("Board"),
            self.board_combo,
            QLabel("Anode"),
            self.anode_combo,
            self.status_label,
            self.flags_label,
        ):
            row1.addWidget(item)
        row1.addStretch(1)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        self.sources_combo = QComboBox()
        for key in SOURCE_CHOICES:
            self.sources_combo.addItem(_SOURCE_LABELS[key], key)
        self.x_lo_spin = self._spin(0.5, 0.99, 0.01, "Lower edge of the photopeak window (E/E0)")
        self.x_hi_spin = self._spin(1.01, 1.5, 0.01, "Upper edge of the photopeak window (E/E0)")
        self.degree_combo = QComboBox()
        for degree in (0, 1, 2):
            self.degree_combo.addItem(str(degree), degree)
        self.degree_combo.setToolTip("Highest polynomial degree of g(r)")
        self.gain_spin = self._spin(-0.5, 0.5, 0.005, "Minimum cross-validated gain to accept")
        self.gain_spin.setDecimals(3)
        self.fit_all_button = QPushButton("Fit All")
        self.fit_all_button.setToolTip("Analyse every board with these options (Process > Fit All)")
        widget: QWidget
        for widget in (
            QLabel("Sources"),
            self.sources_combo,
            QLabel("x window"),
            self.x_lo_spin,
            QLabel("–"),
            self.x_hi_spin,
            QLabel("Max degree"),
            self.degree_combo,
            QLabel("Min gain"),
            self.gain_spin,
            self.fit_all_button,
        ):
            row2.addWidget(widget)
        row2.addStretch(1)
        outer.addLayout(row2)

        self.prev_button.clicked.connect(lambda: self.step_requested.emit(-1))
        self.next_button.clicked.connect(lambda: self.step_requested.emit(1))
        self.node_combo.currentIndexChanged.connect(self._on_node_changed)
        self.board_combo.currentIndexChanged.connect(self._on_board_changed)
        self.anode_combo.currentIndexChanged.connect(self._on_anode_changed)
        for combo in (self.sources_combo, self.degree_combo):
            combo.currentIndexChanged.connect(self._on_options_edited)
        for spin in (self.x_lo_spin, self.x_hi_spin, self.gain_spin):
            spin.valueChanged.connect(self._on_options_edited)
        self.fit_all_button.clicked.connect(self.fit_all_clicked.emit)
        self.set_options(DepthOptions())
        self.set_enabled(False)

    @staticmethod
    def _spin(lo: float, hi: float, step: float, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setToolTip(tip)
        spin.setKeyboardTracking(False)
        return spin

    # -- contents ------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.prev_button,
            self.next_button,
            self.node_combo,
            self.board_combo,
            self.anode_combo,
            self.fit_all_button,
        ):
            widget.setEnabled(enabled)

    def set_anodes(self, anodes: dict[tuple[int, int], list[AnodeKey]]) -> None:
        """The boards with events and each board's anodes (fills the selectors)."""
        self._anodes = dict(anodes)
        self._boards = sorted(anodes)
        self._updating = True
        try:
            self.node_combo.clear()
            for node in sorted({n for n, _ in self._boards}):
                self.node_combo.addItem(str(node), node)
        finally:
            self._updating = False
        self._fill_boards()
        self.set_enabled(bool(self._boards))

    def _fill_boards(self) -> None:
        node = self.node_combo.currentData()
        self._updating = True
        try:
            self.board_combo.clear()
            for n, board in self._boards:
                if n == node:
                    self.board_combo.addItem(str(board), board)
        finally:
            self._updating = False
        self._fill_anodes()

    def _fill_anodes(self) -> None:
        node, board = self.node_combo.currentData(), self.board_combo.currentData()
        self._updating = True
        try:
            self.anode_combo.clear()
            for key in self._anodes.get((node, board), []):
                try:
                    label = electrode_label(key.board, key.rena, key.channel)
                except KeyError:
                    label = "?"
                self.anode_combo.addItem(f"{label}  r{key.rena} ch{key.channel}", key)
        finally:
            self._updating = False

    def set_anode(
        self, key: AnodeKey, status: str | None, flags: tuple[str, ...], review: str
    ) -> None:
        """Show the selected anode without emitting ``anode_requested``."""
        self._updating = True
        try:
            self.node_combo.setCurrentIndex(max(0, self.node_combo.findData(key.node)))
        finally:
            self._updating = False
        self._fill_boards()
        self._updating = True
        try:
            self.board_combo.setCurrentIndex(max(0, self.board_combo.findData(key.board)))
        finally:
            self._updating = False
        self._fill_anodes()
        self._updating = True
        try:
            for i in range(self.anode_combo.count()):
                if self.anode_combo.itemData(i) == key:
                    self.anode_combo.setCurrentIndex(i)
                    break
        finally:
            self._updating = False
        self.show_status(status, flags, review)

    def show_status(self, status: str | None, flags: tuple[str, ...], review: str) -> None:
        category = status_category(status, flags, review=review) if status else CATEGORY_NOT_FITTED
        base = self.palette().color(QPalette.ColorRole.Window)
        color = readable_color(QColor(CATEGORY_COLORS[category]), base)
        text = status or "not fitted"
        if review:
            text += f" ({review})"
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color.name()}; font-weight: bold;")
        self.flags_label.setText(", ".join(flags))

    def selected_anode(self) -> AnodeKey | None:
        data = self.anode_combo.currentData()
        return data if isinstance(data, AnodeKey) else None

    # -- options ---------------------------------------------------------------

    @property
    def options(self) -> DepthOptions:
        return self._options

    def set_options(self, options: DepthOptions) -> None:
        """Show ``options`` without emitting ``options_changed``."""
        self._options = options
        self._updating = True
        try:
            self.sources_combo.setCurrentIndex(max(0, self.sources_combo.findData(options.sources)))
            self.x_lo_spin.setValue(options.x_lo)
            self.x_hi_spin.setValue(options.x_hi)
            self.degree_combo.setCurrentIndex(options.max_degree)
            self.gain_spin.setValue(options.min_gain)
        finally:
            self._updating = False

    def _on_options_edited(self, *_args: object) -> None:
        if self._updating:
            return
        try:
            options = replace(
                self._options,
                sources=str(self.sources_combo.currentData()),
                x_lo=float(self.x_lo_spin.value()),
                x_hi=float(self.x_hi_spin.value()),
                max_degree=int(self.degree_combo.currentData()),
                min_gain=float(self.gain_spin.value()),
            )
        except (TypeError, ValueError):
            self.set_options(self._options)
            return
        self._options = options
        self.options_changed.emit(options)

    # -- selectors -------------------------------------------------------------

    def _on_node_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._fill_boards()
        self._emit_first_anode()

    def _on_board_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._fill_anodes()
        self._emit_first_anode()

    def _on_anode_changed(self, _index: int) -> None:
        if self._updating:
            return
        key = self.selected_anode()
        if key is not None:
            self.anode_requested.emit(key)

    def _emit_first_anode(self) -> None:
        key = self.selected_anode()
        if key is not None:
            self.anode_requested.emit(key)
