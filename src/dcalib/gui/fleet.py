"""The Fleet summary tab: the whole detector's results at a glance (plan section 9).

- Status and flag counts, and how many anodes carry an override or a rejection.
- Histograms of ``cv_gain`` (corrected and no-gain anodes), ``peak_spread``
  (every fitted anode), the relative FWHM and FWTM change at 511 keV of the
  corrected anodes, the degree chosen and ``ge_cs_max_diff``.
- A sortable table of the worst N anodes by a chosen criterion; clicking a
  row opens that anode.

Everything comes from the (merged) stored results; nothing is loaded.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dcalib.analysis import AnodeResult
from dcalib.channels import AnodeKey
from dcalib.gui.session import anode_title
from dcalib.options import FLAGS, STATUSES

__all__ = ["CRITERIA", "FleetView"]


@dataclass(frozen=True)
class Criterion:
    """A worst-N ordering: the value of an anode (None to leave it out), largest first."""

    label: str
    value: Callable[[AnodeResult], float | None]
    unit: str = "%"


def _change(result: AnodeResult, metric: str) -> float | None:
    before: float | None = getattr(result, f"{metric}_before")
    after: float | None = getattr(result, f"{metric}_after")
    if not result.ok or before is None or after is None or before <= 0:
        return None
    return 100.0 * (after / before - 1.0)


def _pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


CRITERIA: tuple[Criterion, ...] = (
    Criterion("FWHM 511 keV increase (corrected)", lambda r: _change(r, "fwhm_511")),
    Criterion("FWTM 511 keV increase (corrected)", lambda r: _change(r, "fwtm_511")),
    Criterion("FWHM 662 keV increase (corrected)", lambda r: _change(r, "fwhm_662")),
    Criterion(
        "Lowest CV gain (corrected)",
        lambda r: -100.0 * r.cv_gain if r.ok and r.cv_gain is not None else None,
    ),
    Criterion("Largest peak spread", lambda r: _pct(r.peak_spread)),
    Criterion("Largest Ge/Cs difference", lambda r: _pct(r.ge_cs_max_diff)),
    Criterion("Largest curve chi2/ndf", lambda r: r.chi2ndf, ""),
)

TABLE_COLUMNS = ("Anode", "Status", "Value", "cv_gain %", "FWHM 511 before", "after", "Flags")


class _NumberItem(QTableWidgetItem):
    """A table cell that sorts numerically."""

    def __init__(self, value: float | None, text: str) -> None:
        super().__init__(text)
        self.value = -math.inf if value is None else value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _NumberItem):
            return self.value < other.value
        return super().__lt__(other)


def _hist(plot: pg.PlotItem, values: list[float], bins: int, title: str, color: str) -> None:
    plot.clear()
    plot.setTitle(f"{title} ({len(values):,})", size="9pt")
    if not values:
        return
    array = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(array, [0.5, 99.5])
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    counts, edges = np.histogram(np.clip(array, lo, hi), bins=bins, range=(lo, hi))
    plot.addItem(
        pg.PlotDataItem(
            edges,
            counts,
            stepMode="center",
            fillLevel=0,
            brush=pg.mkBrush(color + "88"),
            pen=pg.mkPen(color),
        )
    )
    median = float(np.median(array))
    plot.addItem(
        pg.InfiniteLine(pos=median, angle=90, pen=pg.mkPen("#ffffff", style=Qt.PenStyle.DashLine))
    )


class FleetView(QWidget):
    """The Fleet summary tab (see the module docstring).

    Signals:
        anode_activated(object): an :class:`AnodeKey` whose table row was clicked.
    """

    anode_activated = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[AnodeResult] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.counts_label = QLabel("No results")
        self.counts_label.setWordWrap(True)
        self.counts_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.counts_label)
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)
        self.graphics = pg.GraphicsLayoutWidget()
        splitter.addWidget(self.graphics)
        self.plots = {
            name: self.graphics.addPlot(row=i // 3, col=i % 3)
            for i, name in enumerate(("cv_gain", "peak_spread", "fwhm", "fwtm", "degree", "ge_cs"))
        }
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.criterion_combo = QComboBox()
        for criterion in CRITERIA:
            self.criterion_combo.addItem(criterion.label)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(5, 500)
        self.n_spin.setValue(25)
        controls.addWidget(QLabel("Worst"))
        controls.addWidget(self.n_spin)
        controls.addWidget(QLabel("by"))
        controls.addWidget(self.criterion_combo)
        controls.addStretch(1)
        bottom_layout.addLayout(controls)
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        bottom_layout.addWidget(self.table, 1)
        splitter.addWidget(bottom)
        splitter.setSizes([400, 300])
        self.criterion_combo.currentIndexChanged.connect(self._fill_table)
        self.n_spin.valueChanged.connect(self._fill_table)
        self.table.cellClicked.connect(self._on_cell_clicked)

    def set_results(self, results: list[AnodeResult]) -> None:
        self._results = list(results)
        self._fill_counts()
        self._fill_plots()
        self._fill_table()

    def clear(self) -> None:
        self.set_results([])
        self.counts_label.setText("No results: run Process > Fit All")

    def _fill_counts(self) -> None:
        results = self._results
        if not results:
            self.counts_label.setText("No results")
            return
        statuses = Counter(r.status for r in results)
        flags = {flag: sum(1 for r in results if flag in r.flags) for flag in FLAGS}
        n_override = sum(1 for r in results if r.options_source == "override")
        n_rejected = sum(1 for r in results if r.review)
        n_dcc = sum(1 for r in results if r.exported)
        self.counts_label.setText(
            f"<b>{len(results):,} anodes</b>, {n_dcc:,} in the .dcc. Status: "
            + ", ".join(f"{s} {statuses.get(s, 0):,}" for s in STATUSES)
            + ". Flags: "
            + (", ".join(f"{f} {n:,}" for f, n in flags.items() if n) or "none")
            + f". Overrides: {n_override:,}. Rejected: {n_rejected:,}."
        )

    def _fill_plots(self) -> None:
        results = self._results
        fitted = [r for r in results if r.degree is not None]
        ok = [r for r in results if r.ok]
        _hist(
            self.plots["cv_gain"],
            [100 * r.cv_gain for r in results if r.cv_gain is not None],
            40,
            "CV gain (%)",
            "#a3be8c",
        )
        _hist(
            self.plots["peak_spread"],
            [100 * r.peak_spread for r in fitted if r.peak_spread is not None],
            40,
            "Peak spread (% of E0)",
            "#88c0d0",
        )
        _hist(
            self.plots["fwhm"],
            [v for v in (_change(r, "fwhm_511") for r in ok) if v is not None],
            40,
            "FWHM 511 keV change (%)",
            "#ebcb8b",
        )
        _hist(
            self.plots["fwtm"],
            [v for v in (_change(r, "fwtm_511") for r in ok) if v is not None],
            40,
            "FWTM 511 keV change (%)",
            "#d08770",
        )
        _hist(
            self.plots["ge_cs"],
            [100 * r.ge_cs_max_diff for r in fitted if r.ge_cs_max_diff is not None],
            40,
            "Ge/Cs max difference (%)",
            "#b48ead",
        )
        plot = self.plots["degree"]
        plot.clear()
        degrees = Counter(r.degree for r in fitted)
        plot.setTitle(f"Degree chosen ({len(fitted):,})", size="9pt")
        plot.addItem(
            pg.BarGraphItem(
                x=[0, 1, 2],
                height=[degrees.get(d, 0) for d in (0, 1, 2)],
                width=0.6,
                brush="#5e81ac",
            )
        )

    def _fill_table(self, *_args: object) -> None:
        criterion = CRITERIA[max(0, self.criterion_combo.currentIndex())]
        scored = [(criterion.value(r), r) for r in self._results]
        ranked = sorted(
            ((v, r) for v, r in scored if v is not None and math.isfinite(v)), key=lambda t: -t[0]
        )
        rows = ranked[: self.n_spin.value()]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for i, (value, r) in enumerate(rows):
            anode = QTableWidgetItem(anode_title(r.key))
            anode.setData(Qt.ItemDataRole.UserRole, r.key)
            status = r.status + (f" ({r.review})" if r.review else "")
            cells = [
                anode,
                QTableWidgetItem(status),
                _NumberItem(value, f"{value:.3g}{criterion.unit and ' ' + criterion.unit}"),
                _NumberItem(r.cv_gain, "" if r.cv_gain is None else f"{100 * r.cv_gain:.2f}"),
                _NumberItem(
                    r.fwhm_511_before,
                    "" if r.fwhm_511_before is None else f"{r.fwhm_511_before:.2f}",
                ),
                _NumberItem(
                    r.fwhm_511_after, "" if r.fwhm_511_after is None else f"{r.fwhm_511_after:.2f}"
                ),
                QTableWidgetItem(r.flags_text),
            ]
            for j, cell in enumerate(cells):
                self.table.setItem(i, j, cell)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def table_keys(self) -> list[AnodeKey]:
        """The anodes listed, in table order (for tests)."""
        keys = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None:
                keys.append(item.data(Qt.ItemDataRole.UserRole))
        return keys

    def _on_cell_clicked(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            key = item.data(Qt.ItemDataRole.UserRole)
            if key is not None:
                self.anode_activated.emit(AnodeKey(*key))
