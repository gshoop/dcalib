"""The Spectra tab: one anode's photopeaks before and after the correction (plan section 9).

Two panels, 511 keV (the Ge-68 events) and 662 keV (the Cs-137 events), each
with the anode energy spectrum before and after the correction (histograms in
keV, 0.25 % of E0 per bin), the smoothed spectra the resolution metrics are
measured on (:mod:`dcalib.metrics`), their FWHM and FWTM drawn where they are
measured (horizontal bars between the crossings, above the continuum level)
and the Gaussian-core fit of the corrected photopeak. The values are listed in
each panel's title. An anode that is not corrected shows its raw spectrum
only.

The events are those of the Depth tab (calibrated anode and cathode) with r in
the fit's range, as the CSV metrics use them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from dcalib.gui.session import AnodeData, anode_title
from dcalib.metrics import (
    KDE_BIN,
    WidthCrossings,
    fit_photopeak,
    kde_spectrum,
    width_crossings,
)
from dcalib.options import SOURCE_CS, SOURCE_GE, DepthOptions

__all__ = ["SpectraView"]

HIST_BIN = 0.0025
HIST_RANGE = (0.60, 1.30)
BEFORE_COLOR = "#8fa1b3"
AFTER_COLOR = "#ebcb8b"
FIT_COLOR = "#bf616a"
PANELS = ((SOURCE_GE, 511.0, "511 keV (Ge-68)"), (SOURCE_CS, 662.0, "662 keV (Cs-137)"))


class _Panel:
    """One energy's plot and its items."""

    def __init__(self, plot: pg.PlotItem) -> None:
        self.plot = plot
        self.before = pg.PlotDataItem(pen=pg.mkPen(BEFORE_COLOR, width=1), stepMode="center")
        self.after = pg.PlotDataItem(pen=pg.mkPen(AFTER_COLOR, width=1), stepMode="center")
        self.before_smooth = pg.PlotDataItem(pen=pg.mkPen(BEFORE_COLOR, width=2))
        self.after_smooth = pg.PlotDataItem(pen=pg.mkPen(AFTER_COLOR, width=2))
        self.fit = pg.PlotDataItem(pen=pg.mkPen(FIT_COLOR, width=2, style=Qt.PenStyle.DashLine))
        self.before_markers = pg.PlotDataItem(connect="pairs", pen=pg.mkPen(BEFORE_COLOR, width=3))
        self.after_markers = pg.PlotDataItem(connect="pairs", pen=pg.mkPen(AFTER_COLOR, width=3))
        for item in (
            self.before,
            self.after,
            self.before_smooth,
            self.after_smooth,
            self.fit,
            self.before_markers,
            self.after_markers,
        ):
            plot.addItem(item)
        self.legend = plot.addLegend(offset=(-5, 5))
        self.text = ""
        self.lines: list[str] = []

    def clear(self) -> None:
        empty = np.empty(0)
        for item in (
            self.before,
            self.after,
            self.before_smooth,
            self.after_smooth,
            self.fit,
            self.before_markers,
            self.after_markers,
        ):
            if item in (self.before, self.after):
                item.setData(np.array([0.0, 1.0]), np.array([0.0]))
            else:
                item.setData(empty, empty)
        self.legend.clear()
        self.text = ""
        self.lines = []


def _histogram(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_bins = int(round((HIST_RANGE[1] - HIST_RANGE[0]) / HIST_BIN))
    counts, edges = np.histogram(values, bins=n_bins, range=HIST_RANGE)
    return edges, counts.astype(np.float64)


def _width_bars(
    crossings: list[WidthCrossings | None], e0: float, scale: float
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for c in crossings:
        if c is None:
            continue
        xs += [e0 * c.left, e0 * c.right]
        ys += [scale * c.level, scale * c.level]
    return xs, ys


class SpectraView(QWidget):
    """The Spectra tab (see the module docstring).

    Signals:
        display_changed: The log-scale toggle changed.
    """

    display_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: AnodeData | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self.title_label = QLabel("No anode selected")
        top.addWidget(self.title_label, 1)
        self.log_check = QCheckBox("Log scale")
        top.addWidget(self.log_check)
        layout.addLayout(top)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.panels: dict[int, _Panel] = {}
        for col, (source, _, title) in enumerate(PANELS):
            plot = self.graphics.addPlot(row=0, col=col, title=title)
            plot.setLabel("bottom", "Anode energy", units="keV")
            plot.setLabel("left", "Events / bin")
            plot.showGrid(x=True, y=True, alpha=0.15)
            self.panels[source] = _Panel(plot)
        self.log_check.toggled.connect(self._on_log_toggled)
        self.clear()

    @property
    def data(self) -> AnodeData | None:
        return self._data

    def log_scale(self) -> bool:
        return self.log_check.isChecked()

    def set_log_scale(self, on: bool) -> None:
        self.log_check.blockSignals(True)
        self.log_check.setChecked(on)
        self.log_check.blockSignals(False)
        for panel in self.panels.values():
            panel.plot.setLogMode(y=on)

    def _on_log_toggled(self, on: bool) -> None:
        for panel in self.panels.values():
            panel.plot.setLogMode(y=on)
        self.display_changed.emit()

    def panel_text(self, source: int) -> str:
        """The values shown in a panel's title (for tests)."""
        return self.panels[source].text

    def clear(self, message: str = "No anode selected") -> None:
        self._data = None
        self.title_label.setText(message)
        for source, (_, _, title) in zip(self.panels, PANELS):
            self.panels[source].clear()
            self.panels[source].plot.setTitle(title)

    def set_data(self, data: AnodeData) -> None:
        self._data = data
        corrected = data.corrected
        self.title_label.setText(
            f"{anode_title(data.key)}: "
            + ("before and after the correction" if corrected else "not corrected (raw spectra)")
        )
        opts = data.options or DepthOptions()
        in_r = (data.r >= opts.r_lo) & (data.r <= opts.r_hi) & np.isfinite(data.x)
        after_all = data.corrected_x() if corrected else data.x
        for source, e0, title in PANELS:
            panel = self.panels[source]
            panel.clear()
            mask = in_r & (data.source == source)
            before = data.x[mask]
            after = after_all[mask]
            self._draw(panel, before, after if corrected else None, e0)
            panel.plot.setTitle("<br>".join([title, *panel.lines]), size="9pt")

    def _draw(
        self,
        panel: _Panel,
        before: npt.NDArray[np.float64],
        after: npt.NDArray[np.float64] | None,
        e0: float,
    ) -> None:
        if len(before) == 0:
            panel.text = "no events"
            return
        edges, counts = _histogram(before)
        panel.before.setData(e0 * edges, counts)
        panel.legend.addItem(panel.before, f"before ({len(before):,})")
        scale = HIST_BIN / KDE_BIN  # smoothed counts per grid step -> per histogram bin
        centres, smooth = kde_spectrum(before)
        panel.before_smooth.setData(e0 * centres, scale * smooth)
        crossings = [width_crossings(before, 0.5), width_crossings(before, 0.1)]
        parts = [self._describe("before", crossings)]
        panel.before_markers.setData(*_width_bars(crossings, e0, scale))
        if after is not None:
            edges_a, counts_a = _histogram(after)
            panel.after.setData(e0 * edges_a, counts_a)
            panel.legend.addItem(panel.after, "after")
            centres_a, smooth_a = kde_spectrum(after)
            panel.after_smooth.setData(e0 * centres_a, scale * smooth_a)
            after_crossings = [width_crossings(after, 0.5), width_crossings(after, 0.1)]
            parts.append(self._describe("after", after_crossings))
            panel.after_markers.setData(*_width_bars(after_crossings, e0, scale))
            fit = fit_photopeak(after)
            if fit.ok:
                lo, hi = fit.window
                xs = np.linspace(lo, hi, 200)
                panel.fit.setData(e0 * xs, fit.curve(xs))
                panel.legend.addItem(panel.fit, f"core fit (FWHM {fit.fwhm_pct:.2f} %)")
        panel.text = "; ".join(parts)
        panel.lines = parts
        panel.plot.setXRange(0.8 * e0, 1.1 * e0, padding=0)

    @staticmethod
    def _describe(label: str, crossings: list[WidthCrossings | None]) -> str:
        fwhm, fwtm = crossings
        text_fwhm = f"{fwhm.width_pct:.2f}" if fwhm is not None else "n/a"
        text_fwtm = f"{fwtm.width_pct:.2f}" if fwtm is not None else "n/a"
        return f"{label} FWHM {text_fwhm} % FWTM {text_fwtm} %"
