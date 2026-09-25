"""The Board grid tab: one board's anodes as small multiples (plan section 9).

The 39 anodes of the selected board in physical strip order (position 1 = the
low-node side, :meth:`ElectrodeMap.physical_strip_position`), eight to a row.
Each cell shows the stored pooled slice points and the curve g(r), and its
frame has the anode's status colour (the System Map's categories). Everything
comes from the stored results (:meth:`~dcalib.gui.session.DepthSession.board_grid`):
nothing is loaded or refitted. The cells share one vertical range (the 2-98 %
range of the board's slice points, widened a little), so the curves compare at
a glance. Clicking a cell opens that anode.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from dcalib.channels import AnodeKey, electrode_label
from dcalib.gui.map_colors import (
    CATEGORY_COLORS,
    CATEGORY_NO_DATA,
    DEFAULT_INFORMATIONAL_FLAGS,
    status_category,
)
from dcalib.gui.session import BoardGridEntry

__all__ = ["COLUMNS", "BoardGridView"]

COLUMNS = 8
N_CELLS = 39
_R = np.linspace(0.0, 1.3, 60)
DEFAULT_Y = (0.95, 1.03)


class _Cell:
    def __init__(self, plot: pg.PlotItem) -> None:
        self.plot = plot
        self.points = pg.ScatterPlotItem(size=4, brush=pg.mkBrush("#ffffff"), pen=None)
        self.curve = pg.PlotDataItem(pen=pg.mkPen("#ff5555", width=1.5))
        plot.addItem(self.points)
        plot.addItem(self.curve)
        plot.hideAxis("bottom")
        plot.hideAxis("left")
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        self.key: AnodeKey | None = None

    def set_entry(
        self, entry: BoardGridEntry | None, y_range: tuple[float, float], selected: bool
    ) -> None:
        empty = np.empty(0)
        vb = self.plot.getViewBox()
        if entry is None:
            self.key = None
            self.points.setData([], [])
            self.curve.setData(empty, empty)
            self.plot.setTitle("")
            vb.setBorder(None)
            return
        self.key = entry.key
        try:
            label = electrode_label(entry.key.board, entry.key.rena, entry.key.channel)
        except KeyError:
            label = "?"
        result = entry.result
        if result is None:
            category = CATEGORY_NO_DATA
        else:
            category = status_category(
                result.status, result.flags, DEFAULT_INFORMATIONAL_FLAGS, review=result.review
            )
        mark = " *" if result is not None and result.options_source == "override" else ""
        self.plot.setTitle(f"{entry.position} {label}{mark}", size="7pt")
        slices = entry.slices.get("both")
        if slices is not None and len(slices):
            self.points.setData(slices.r_med, slices.mu)
        else:
            self.points.setData([], [])
        if entry.curve is not None:
            self.curve.setData(_R, entry.curve(_R))
        else:
            self.curve.setData(empty, empty)
        width = 4 if selected else 2
        vb.setBorder(pg.mkPen(CATEGORY_COLORS[category], width=width))
        self.plot.setRange(xRange=(0.0, 1.3), yRange=y_range, padding=0.02)


class BoardGridView(QWidget):
    """The Board grid tab (see the module docstring).

    Signals:
        anode_activated(object): an :class:`AnodeKey` whose cell was clicked.
    """

    anode_activated = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title_label = QLabel("No board selected")
        layout.addWidget(self.title_label)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.cells: list[_Cell] = []
        for index in range(N_CELLS):
            plot = self.graphics.addPlot(row=index // COLUMNS, col=index % COLUMNS)
            self.cells.append(_Cell(plot))
        scene = self.graphics.scene()
        if scene is not None:
            scene.sigMouseClicked.connect(self._on_clicked)
        self.board: tuple[int, int] | None = None
        self.y_range = DEFAULT_Y

    def clear(self, message: str = "No board selected") -> None:
        self.board = None
        self.title_label.setText(message)
        for cell in self.cells:
            cell.set_entry(None, DEFAULT_Y, False)

    def set_board(
        self, node: int, board: int, entries: list[BoardGridEntry], selected: AnodeKey | None = None
    ) -> None:
        """Show a board's anodes (``entries`` in strip order)."""
        self.board = (node, board)
        points = np.concatenate(
            [e.slices["both"].mu for e in entries if "both" in e.slices and len(e.slices["both"])]
            or [np.empty(0)]
        )
        if len(points) >= 3:
            lo, hi = np.percentile(points, [2, 98])
            pad = max(0.2 * (hi - lo), 0.005)
            self.y_range = (float(lo - pad), float(hi + pad))
        else:
            self.y_range = DEFAULT_Y
        n_ok = sum(1 for e in entries if e.result is not None and e.result.exported)
        self.title_label.setText(
            f"Node {node} Board {board}: {n_ok}/{len(entries)} anodes corrected; strip position 1 "
            "is on the low-node side; * = override. Click an anode to open it."
        )
        for index, cell in enumerate(self.cells):
            entry = entries[index] if index < len(entries) else None
            cell.set_entry(entry, self.y_range, entry is not None and entry.key == selected)

    def cell_center(self, key: AnodeKey) -> QPointF | None:
        """Scene position of an anode's cell (for tests)."""
        for cell in self.cells:
            if cell.key == key:
                center: QPointF = cell.plot.sceneBoundingRect().center()
                return center
        return None

    def _on_clicked(self, event: object) -> None:
        pos = event.scenePos()  # type: ignore[attr-defined]
        for cell in self.cells:
            if cell.key is not None and cell.plot.sceneBoundingRect().contains(pos):
                self.anode_activated.emit(cell.key)
                return

    def activate_at(self, pos: QPointF) -> None:
        """Act as a click at scene position ``pos`` (for tests)."""
        for cell in self.cells:
            if cell.key is not None and cell.plot.sceneBoundingRect().contains(pos):
                self.anode_activated.emit(cell.key)
                return
