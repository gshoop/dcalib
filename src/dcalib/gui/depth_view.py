"""The Depth tab: one anode's photopeak against depth, before and after (plan section 9).

Two linked 2D histograms of r = C/A (horizontal) against the anode energy
(vertical; normalised ``x = A / E0`` or keV), drawn with a log colour scale:

- **Before**: the raw energies, with the fitted curve g(r) (solid over the r
  range the data cover, dotted where consumers extrapolate it), the pooled
  slice points +- their errors and, optionally, the Ge-only and Cs-only
  curves (from the stored per-source slices) and the selection window of the
  fit (``x_lo..x_hi`` by ``r_lo..r_hi``, in x units).
- **After**: the corrected energies ``x / g(r)``: a well corrected anode shows
  a flat ridge on the dashed line at 1. Without an accepted correction (the
  anode is not in the ``.dcc``) it shows the raw energies, and says so.
- **Residuals** (under the first plot, sharing its r axis): the slice points
  minus the curve, ``mu_i - g(r_i)``, +- their errors.

A cathode filter restricts the events to one cathode, and "Cathode ridges"
draws each cathode's photopeak ridge (the median energy of the photopeak
events in r bins) in its own colour: a miscalibrated cathode stretches or
shifts r, so its ridge stands apart (open item O8). The source selector shows
the Ge-68 (511 keV), Cs-137 (662 keV) or all events; in keV both photopeaks
appear, at 511 and 662 keV.

The widget draws an :class:`~dcalib.gui.session.AnodeData`; it never loads
data itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from dcalib.calib import source_e0
from dcalib.channels import electrode_label
from dcalib.gui.session import AnodeData, anode_title
from dcalib.options import SOURCE_CS, SOURCE_GE, DepthOptions

__all__ = ["DepthDisplay", "DepthView"]

R_RANGE = (-0.05, 1.45)
X_RANGE = (0.60, 1.25)
KEV_RANGE = (300.0, 850.0)
X_VIEW = (0.82, 1.12)
KEV_VIEW = (380.0, 760.0)
R_BINS = 150
E_BINS = 130
RIDGE_BINS = 14
RIDGE_MIN_EVENTS = 150
PHOTOPEAK_WINDOW = (0.88, 1.10)

SOURCES = {"both": None, "ge": SOURCE_GE, "cs": SOURCE_CS}
SOURCE_LABELS = {"both": "Ge + Cs", "ge": "Ge-68 (511 keV)", "cs": "Cs-137 (662 keV)"}
UNITS = {"x": "E/E0", "kev": "keV"}
CURVE_COLOR = "#ff5555"
GE_COLOR = "#4fc3f7"
CS_COLOR = "#f48fb1"
LEGEND_BRUSH = (0, 0, 0, 170)  # a dark, translucent legend background over the histograms
RIDGE_COLORS = (
    "#ffd54f",
    "#81c784",
    "#64b5f6",
    "#e57373",
    "#ba68c8",
    "#4db6ac",
    "#ff8a65",
    "#a1887f",
)


@dataclass(frozen=True)
class DepthDisplay:
    """The Depth tab's display settings (persisted by the main window)."""

    sources: str = "both"
    units: str = "x"
    cathode: int | None = None  # rena * 36 + channel, or None for all
    source_curves: bool = True
    ridges: bool = False
    window: bool = True


def _log_image(counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.log1p(counts)


class DepthView(QWidget):
    """The Depth tab (see the module docstring).

    Signals:
        display_changed: A display setting changed.
    """

    display_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: AnodeData | None = None
        self._display = DepthDisplay()
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        self.title_label = QLabel("No anode selected")
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.title_label, 1)
        self.source_combo = QComboBox()
        for key, label in SOURCE_LABELS.items():
            self.source_combo.addItem(label, key)
        self.unit_combo = QComboBox()
        for key, label in UNITS.items():
            self.unit_combo.addItem(label, key)
        self.cathode_combo = QComboBox()
        self.cathode_combo.addItem("All cathodes", None)
        self.curves_check = QCheckBox("Ge/Cs curves")
        self.ridges_check = QCheckBox("Cathode ridges")
        self.window_check = QCheckBox("Selection window")
        for widget in (
            QLabel("Source:"),
            self.source_combo,
            QLabel("Energy:"),
            self.unit_combo,
            self.cathode_combo,
            self.curves_check,
            self.ridges_check,
            self.window_check,
        ):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: #d08770;")
        layout.addWidget(self.message_label)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.before_plot = self.graphics.addPlot(row=0, col=0, title="Before correction")
        self.after_plot = self.graphics.addPlot(row=0, col=1, title="After correction")
        self.residual_plot = self.graphics.addPlot(row=1, col=0, title="Slice residuals")
        self.graphics.ci.layout.setRowStretchFactor(0, 3)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        for plot in (self.before_plot, self.after_plot, self.residual_plot):
            plot.setLabel("bottom", "C/A")
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.residual_plot.setLabel("left", "μ − g", units="")
        self.after_plot.setXLink(self.before_plot)
        self.residual_plot.setXLink(self.before_plot)
        self.after_plot.setYLink(self.before_plot)

        cmap = pg.colormap.get("inferno")
        self.before_image = pg.ImageItem()
        self.after_image = pg.ImageItem()
        for image, plot in (
            (self.before_image, self.before_plot),
            (self.after_image, self.after_plot),
        ):
            image.setColorMap(cmap)
            plot.addItem(image)
        self.curve_item = pg.PlotDataItem(pen=pg.mkPen(CURVE_COLOR, width=2))
        self.curve_extrapolated = pg.PlotDataItem(
            pen=pg.mkPen(CURVE_COLOR, width=2, style=Qt.PenStyle.DotLine)
        )
        self.ge_curve = pg.PlotDataItem(
            pen=pg.mkPen(GE_COLOR, width=1.5, style=Qt.PenStyle.DashLine)
        )
        self.cs_curve = pg.PlotDataItem(
            pen=pg.mkPen(CS_COLOR, width=1.5, style=Qt.PenStyle.DashLine)
        )
        self.slice_points = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#ffffff"), pen=None)
        self.slice_errors = pg.ErrorBarItem(pen=pg.mkPen("#ffffff"), beam=0.01)
        self.window_box = pg.PlotDataItem(
            pen=pg.mkPen("#88c0d0", width=1, style=Qt.PenStyle.DashLine)
        )
        self.unity_line = pg.InfiniteLine(
            angle=0, pos=1.0, pen=pg.mkPen("#ffffff", style=Qt.PenStyle.DashLine)
        )
        self.ridge_items: list[pg.PlotDataItem] = []
        for item in (
            self.curve_item,
            self.curve_extrapolated,
            self.ge_curve,
            self.cs_curve,
            self.slice_errors,
            self.slice_points,
            self.window_box,
        ):
            self.before_plot.addItem(item)
        self.after_plot.addItem(self.unity_line)
        self.residual_points = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#ffffff"), pen=None)
        self.residual_errors = pg.ErrorBarItem(pen=pg.mkPen("#ffffff"), beam=0.01)
        self.residual_plot.addItem(pg.InfiniteLine(angle=0, pos=0.0, pen=pg.mkPen("#888888")))
        self.residual_plot.addItem(self.residual_errors)
        self.residual_plot.addItem(self.residual_points)
        self.legend = self.before_plot.addLegend(offset=(-10, 10), brush=LEGEND_BRUSH)

        self.source_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.unit_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.cathode_combo.currentIndexChanged.connect(self._on_controls_changed)
        for check in (self.curves_check, self.ridges_check, self.window_check):
            check.toggled.connect(self._on_controls_changed)
        self.set_display(DepthDisplay())
        self.show_message("Select an anode on the System Map")

    # -- state -------------------------------------------------------------

    @property
    def data(self) -> AnodeData | None:
        return self._data

    def display(self) -> DepthDisplay:
        return self._display

    def set_display(self, display: DepthDisplay) -> None:
        """Apply display settings without emitting ``display_changed``."""
        self._updating = True
        try:
            self._display = display
            self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(display.sources)))
            self.unit_combo.setCurrentIndex(max(0, self.unit_combo.findData(display.units)))
            self.curves_check.setChecked(display.source_curves)
            self.ridges_check.setChecked(display.ridges)
            self.window_check.setChecked(display.window)
        finally:
            self._updating = False
        self._redraw()

    def _on_controls_changed(self, *_args: object) -> None:
        if self._updating:
            return
        cathode = self.cathode_combo.currentData()
        self._display = DepthDisplay(
            sources=str(self.source_combo.currentData()),
            units=str(self.unit_combo.currentData()),
            cathode=int(cathode) if cathode is not None else None,
            source_curves=self.curves_check.isChecked(),
            ridges=self.ridges_check.isChecked(),
            window=self.window_check.isChecked(),
        )
        self._redraw()
        self.display_changed.emit()

    def show_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def clear(self, message: str = "") -> None:
        """Forget the anode and blank the plots."""
        self._data = None
        self.title_label.setText("No anode selected")
        self._fill_cathodes()
        self._redraw()
        self.show_message(message)

    def set_loading(self, title: str) -> None:
        self.title_label.setText(f"{title} (loading...)")

    def set_data(self, data: AnodeData) -> None:
        """Draw an anode."""
        self._data = data
        self._fill_cathodes()
        result = data.result
        status = result.status if result is not None else "not fitted"
        if result is not None and result.review:
            status += f", {result.review}"
        if result is not None and result.options_source == "override":
            status += ", override"
        self.title_label.setText(
            f"{anode_title(data.key)}: {status}; {len(data.x):,} events with calibrated "
            f"cathodes of {data.n_events:,}"
        )
        messages = []
        if data.n_uncal_cathode:
            messages.append(
                f"{data.n_uncal_cathode:,} events dropped: their cathode has no keV calibration"
            )
        if result is not None and not data.corrected:
            messages.append("Not in the .dcc: the After plot shows the raw energies")
        self.show_message("; ".join(messages))
        self._redraw()

    def _fill_cathodes(self) -> None:
        self._updating = True
        try:
            wanted = self._display.cathode
            self.cathode_combo.clear()
            self.cathode_combo.addItem("All cathodes", None)
            data = self._data
            if data is not None:
                for rena, channel in data.cathodes():
                    code = rena * 36 + channel
                    try:
                        label = electrode_label(data.key.board, rena, channel)
                    except KeyError:
                        label = f"r{rena} ch{channel}"
                    count = int(np.count_nonzero(data.cathode == code))
                    self.cathode_combo.addItem(f"{label} ({count:,})", code)
            index = self.cathode_combo.findData(wanted) if wanted is not None else 0
            if index < 0:
                index = 0
                self._display = DepthDisplay(**{**self._display.__dict__, "cathode": None})
            self.cathode_combo.setCurrentIndex(index)
        finally:
            self._updating = False

    # -- drawing -----------------------------------------------------------

    def _selected(self, data: AnodeData) -> npt.NDArray[np.bool_]:
        mask = np.isfinite(data.x) & np.isfinite(data.r)
        source = SOURCES[self._display.sources]
        if source is not None:
            mask &= data.source == source
        if self._display.cathode is not None:
            mask &= data.cathode == self._display.cathode
        return np.asarray(mask)

    def _energy(
        self, data: AnodeData, x: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]:
        if self._display.units == "kev":
            return np.asarray(x[mask] * source_e0(data.source[mask]), dtype=np.float64)
        return np.asarray(x[mask], dtype=np.float64)

    def _energy_range(self) -> tuple[float, float]:
        return KEV_RANGE if self._display.units == "kev" else X_RANGE

    def _set_image(
        self, image: pg.ImageItem, r: npt.NDArray[np.float64], e: npt.NDArray[np.float64]
    ) -> None:
        lo, hi = self._energy_range()
        counts, _, _ = np.histogram2d(r, e, bins=(R_BINS, E_BINS), range=(R_RANGE, (lo, hi)))
        image.setImage(_log_image(counts), autoLevels=True)
        image.setRect(R_RANGE[0], lo, R_RANGE[1] - R_RANGE[0], hi - lo)

    def _redraw(self) -> None:
        data = self._data
        for item in self.ridge_items:
            self.before_plot.removeItem(item)
        self.ridge_items = []
        self.legend.clear()
        empty = np.empty(0)
        if data is None:
            for image in (self.before_image, self.after_image):
                image.clear()
            for item in (
                self.curve_item,
                self.curve_extrapolated,
                self.ge_curve,
                self.cs_curve,
                self.window_box,
            ):
                item.setData(empty, empty)
            self.slice_points.setData([], [])
            self.slice_errors.setData(x=empty, y=empty, height=empty)
            self.residual_points.setData([], [])
            self.residual_errors.setData(x=empty, y=empty, height=empty)
            return
        kev = self._display.units == "kev"
        unit = "keV" if kev else "E/E0"
        for plot in (self.before_plot, self.after_plot):
            plot.setLabel("left", "Anode energy", units=unit)
        mask = self._selected(data)
        r = data.r[mask]
        self._set_image(self.before_image, r, self._energy(data, data.x, mask))
        self._set_image(self.after_image, r, self._energy(data, data.corrected_x(), mask))
        self.unity_line.setVisible(not kev)
        self.after_plot.setTitle(
            "After correction" if data.corrected else "After correction (not corrected: raw)"
        )
        self._draw_curves(data, kev)
        self._draw_slices(data, kev)
        self._draw_window(data, kev)
        if self._display.ridges:
            self._draw_ridges(data, kev)
        self.before_plot.setRange(xRange=R_RANGE, yRange=KEV_VIEW if kev else X_VIEW, padding=0)

    def _scales(self, kev: bool) -> tuple[float, ...]:
        """Factors from E/E0 to the plot's unit: one per photopeak shown (two in keV for both)."""
        if not kev:
            return (1.0,)
        source = SOURCES[self._display.sources]
        if source is None:
            return (511.0, 662.0)
        return (662.0,) if source == SOURCE_CS else (511.0,)

    @staticmethod
    def _copies(
        r: npt.NDArray[np.float64], y: npt.NDArray[np.float64], scales: tuple[float, ...]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(r, y)`` once per scale, the copies separated by NaN (drawn ``connect="finite"``)."""
        rs: list[npt.NDArray[np.float64]] = []
        ys: list[npt.NDArray[np.float64]] = []
        for scale in scales:
            rs += [r, np.array([np.nan])]
            ys += [scale * y, np.array([np.nan])]
        return np.concatenate(rs[:-1]), np.concatenate(ys[:-1])

    def _draw_curves(self, data: AnodeData, kev: bool) -> None:
        empty = np.empty(0)
        scales = self._scales(kev)
        if data.curve is None:
            self.curve_item.setData(empty, empty)
            self.curve_extrapolated.setData(empty, empty)
        else:
            result = data.result
            lo = result.ca_p01 if result is not None and result.ca_p01 is not None else 0.0
            hi = result.ca_p99 if result is not None and result.ca_p99 is not None else 1.3
            inside = np.linspace(lo, hi, 100)
            outside = np.concatenate([np.linspace(0.0, lo, 20), [np.nan], np.linspace(hi, 1.3, 20)])
            self.curve_item.setData(
                *self._copies(inside, data.curve(inside), scales), connect="finite"
            )
            self.curve_extrapolated.setData(
                *self._copies(outside, data.curve(outside), scales), connect="finite"
            )
            self.legend.addItem(self.curve_item, "g(r)")
        show = self._display.source_curves
        for name, item in (("ge", self.ge_curve), ("cs", self.cs_curve)):
            curve = data.source_curves.get(name)
            if show and curve is not None:
                rr = np.linspace(0.0, 1.3, 100)
                item.setData(*self._copies(rr, curve(rr), scales), connect="finite")
                self.legend.addItem(item, f"{name.capitalize()} only")
            else:
                item.setData(empty, empty)

    def _draw_slices(self, data: AnodeData, kev: bool) -> None:
        empty = np.empty(0)
        scales = self._scales(kev)
        slices = data.slices.get("both")
        if slices is None or len(slices) == 0:
            self.slice_points.setData([], [])
            self.slice_errors.setData(x=empty, y=empty, height=empty)
            self.residual_points.setData([], [])
            self.residual_errors.setData(x=empty, y=empty, height=empty)
            return
        r = np.tile(slices.r_med, len(scales))
        mu = np.concatenate([scale * slices.mu for scale in scales])
        err = np.concatenate([scale * slices.mu_err for scale in scales])
        self.slice_points.setData(r, mu)
        self.slice_errors.setData(x=r, y=mu, height=2 * err)
        self.legend.addItem(self.slice_points, "slices")
        if data.curve is not None:
            residual = slices.mu - data.curve(slices.r_med)
            self.residual_points.setData(slices.r_med, residual)
            self.residual_errors.setData(x=slices.r_med, y=residual, height=2 * slices.mu_err)
        else:
            self.residual_points.setData([], [])
            self.residual_errors.setData(x=empty, y=empty, height=empty)

    def _draw_window(self, data: AnodeData, kev: bool) -> None:
        opts = data.options or DepthOptions()
        if not self._display.window or kev:
            self.window_box.setData(np.empty(0), np.empty(0))
            return
        x0, x1, r0, r1 = opts.x_lo, opts.x_hi, opts.r_lo, opts.r_hi
        self.window_box.setData([r0, r1, r1, r0, r0], [x0, x0, x1, x1, x0])

    def _ridge(
        self, r: npt.NDArray[np.float64], e: npt.NDArray[np.float64], edges: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Median energy per r bin (NaN with fewer than 15 events)."""
        which = np.digitize(r, edges) - 1
        out = np.full(len(edges) - 1, math.nan)
        for i in range(len(out)):
            values = e[which == i]
            if len(values) >= 15:
                out[i] = float(np.median(values))
        return out

    def _draw_ridges(self, data: AnodeData, kev: bool) -> None:
        base = self._selected(data)
        edges = np.linspace(0.0, 1.3, RIDGE_BINS + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        window = (data.x >= PHOTOPEAK_WINDOW[0]) & (data.x <= PHOTOPEAK_WINDOW[1])
        # In keV each photopeak has its own ridge; in E/E0 the sources are pooled.
        groups: list[tuple[npt.NDArray[np.bool_], float]] = [
            (
                (
                    data.source == (SOURCE_CS if scale == 662.0 else SOURCE_GE)
                    if kev
                    else np.ones(len(data.x), dtype=bool)
                ),
                scale,
            )
            for scale in self._scales(kev)
        ]
        for index, (rena, channel) in enumerate(data.cathodes()):
            mask_all = base & window & (data.cathode == rena * 36 + channel)
            if int(mask_all.sum()) < RIDGE_MIN_EVENTS:
                continue
            xs: list[npt.NDArray[np.float64]] = []
            ys: list[npt.NDArray[np.float64]] = []
            for group, scale in groups:
                mask = mask_all & group
                xs += [centres, np.array([np.nan])]
                ys += [self._ridge(data.r[mask], scale * data.x[mask], edges), np.array([np.nan])]
            color = RIDGE_COLORS[index % len(RIDGE_COLORS)]
            item = pg.PlotDataItem(
                np.concatenate(xs),
                np.concatenate(ys),
                pen=pg.mkPen(color, width=2),
                connect="finite",
            )
            self.before_plot.addItem(item)
            self.ridge_items.append(item)
            try:
                label = electrode_label(data.key.board, rena, channel)
            except KeyError:
                label = f"r{rena} ch{channel}"
            self.legend.addItem(item, label)
