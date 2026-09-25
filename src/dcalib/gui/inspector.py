"""Fit Inspector dock: every field of the selected anode (plan section 9).

The field list is driven by :data:`dcalib.analysis.RESULT_COLUMNS` (the
``depth_summary.csv`` columns), so a column added to ``AnodeResult`` appears
here without a change: it is placed in a group by its name (:func:`group_of`)
and formatted by its kind. Unavailable values (None) are shown empty.

Below the fields the inspector lists the anode's warning flags with a short
explanation (:data:`FLAG_DESCRIPTIONS`), the review decision and its note, and
the options the result was fitted with, with their source: the batch run or a
per-anode override.
"""

from __future__ import annotations

import math
from typing import Any

from PyQt6.QtGui import QBrush, QColor, QPalette
from PyQt6.QtWidgets import QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from dcalib.analysis import (
    KIND_COUNT,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_INT,
    OPTIONS_OVERRIDE,
    RESULT_COLUMNS,
    AnodeResult,
)
from dcalib.channels import AnodeKey
from dcalib.gui._contrast import readable_color
from dcalib.gui.map_colors import CATEGORY_COLORS, status_category
from dcalib.gui.session import anode_title
from dcalib.options import (
    FLAG_CONVEX_CURVE,
    FLAG_EXTRAPOLATION_RISK,
    FLAG_NARROW_CA_COVERAGE,
    FLAG_PARTIAL_CATHODE_COVERAGE,
    FLAG_SLICE_FIT_FAILED,
    FLAG_SOURCE_INCONSISTENT,
    DepthOptions,
)

__all__ = ["FLAG_DESCRIPTIONS", "RESULT_GROUPS", "FitInspector", "format_value", "group_of"]

FLAG_DESCRIPTIONS: dict[str, str] = {
    FLAG_SLICE_FIT_FAILED: "At least one r slice's photopeak fit failed; the slice was dropped.",
    FLAG_CONVEX_CURVE: "The chosen curve is quadratic and convex (the legacy fit forbade it).",
    FLAG_SOURCE_INCONSISTENT: (
        "The Ge-only and Cs-only curves differ by more than {consistency_min_diff:.1%} "
        "(and {consistency_nsigma:g} sigma) over the 10-90 % r range."
    ),
    FLAG_EXTRAPOLATION_RISK: (
        "g(r) leaves [{extrap_g_lo:g}, {extrap_g_hi:g}] somewhere on [0, {extrap_r_max:g}]: "
        "consumers evaluate it there unclamped."
    ),
    FLAG_NARROW_CA_COVERAGE: (
        "The 1-99 % C/A range of the data is narrower than "
        "[{coverage_r_lo:g}, {coverage_r_hi:g}]."
    ),
    FLAG_PARTIAL_CATHODE_COVERAGE: (
        "More than {partial_cathode_frac:.0%} of the anode's events have a cathode without "
        "a keV calibration."
    ),
}

RESULT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Anode",
        (
            "node",
            "board",
            "rena",
            "channel",
            "electrode",
            "status",
            "flags",
            "review",
            "options_source",
        ),
    ),
    ("Events", ("n_events", "n_ge", "n_cs", "n_uncal_cathode", "n_selected")),
    ("C/A coverage", ("ca_p01", "ca_p50", "ca_p99")),
    (
        "Curve (.dcc units)",
        ("degree", "p0", "p1", "p2", "err_p0", "err_p1", "err_p2", "chi2ndf", "n_slices"),
    ),
    ("Depth effect", ("peak_spread", "ge_cs_max_diff", "cv_gain")),
)
"""Groups of the result fields by name; the remaining columns go to "Resolution"."""

_PERCENT_FRACTIONS = frozenset({"peak_spread", "ge_cs_max_diff", "cv_gain"})
_PRECISE = frozenset({"p0", "p1", "p2", "err_p0", "err_p1", "err_p2"})


def group_of(name: str) -> str:
    for group, names in RESULT_GROUPS:
        if name in names:
            return group
    return "Resolution"


def format_value(name: str, kind: str, value: Any) -> str:
    """Format one field for display (empty for None)."""
    if value is None:
        return ""
    if kind == KIND_FLAGS:
        return ", ".join(value) if value else "none"
    if kind in (KIND_INT, KIND_COUNT):
        return f"{int(value):,}" if name.startswith("n_") else str(int(value))
    if kind == KIND_FLOAT:
        number = float(value)
        if not math.isfinite(number):
            return ""
        if name in _PERCENT_FRACTIONS:
            return f"{100 * number:.3g} %"
        if name.startswith(("fwhm", "fwtm")):
            return f"{number:.3g} %"
        return f"{number:.6g}" if name in _PRECISE else f"{number:.4g}"
    return str(value)


class FitInspector(QWidget):
    """The Fit Inspector dock's contents."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title_label = QLabel("No anode selected")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Field", "Value"])
        header = self.tree.header()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree, 1)
        self.setMinimumWidth(240)

    def rows(self) -> dict[str, str]:
        """``{field: shown text}`` of every leaf (for tests)."""
        out: dict[str, str] = {}
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            assert group is not None
            for j in range(group.childCount()):
                child = group.child(j)
                assert child is not None
                out[child.text(0)] = child.text(1)
        return out

    def clear(self, message: str = "No anode selected") -> None:
        self.tree.clear()
        self.title_label.setText(message)

    def show_anode(
        self,
        key: AnodeKey,
        result: AnodeResult | None,
        options: DepthOptions | None,
        review_note: str = "",
    ) -> None:
        """Show an anode's result (or that it has none)."""
        self.tree.clear()
        self.title_label.setText(anode_title(key))
        if result is None:
            item = QTreeWidgetItem(["Status", "not fitted (run Process > Fit All)"])
            self.tree.addTopLevelItem(item)
            return
        groups: dict[str, QTreeWidgetItem] = {}
        for name, _ in (*RESULT_GROUPS, ("Resolution", ())):
            group = QTreeWidgetItem([name, ""])
            group.setFirstColumnSpanned(True)
            self.tree.addTopLevelItem(group)
            groups[name] = group
        text_color = self.palette().color(QPalette.ColorRole.Base)
        for column in RESULT_COLUMNS:
            value = getattr(result, column.name)
            item = QTreeWidgetItem([column.name, format_value(column.name, column.kind, value)])
            if column.name == "status":
                category = status_category(result.status, result.flags, review=result.review)
                color = readable_color(QColor(CATEGORY_COLORS[category]), text_color)
                item.setForeground(1, QBrush(color))
            groups[group_of(column.name)].addChild(item)
        if result.flags:
            flags = QTreeWidgetItem(["Flags explained", ""])
            flags.setFirstColumnSpanned(True)
            self.tree.addTopLevelItem(flags)
            values = (options or DepthOptions()).to_dict()
            for flag in result.flags:
                text = FLAG_DESCRIPTIONS.get(flag, "").format(**values)
                child = QTreeWidgetItem([flag, text])
                child.setToolTip(1, text)
                flags.addChild(child)
        if result.review:
            review = QTreeWidgetItem(["Review", ""])
            review.setFirstColumnSpanned(True)
            self.tree.addTopLevelItem(review)
            review.addChild(QTreeWidgetItem(["state", result.review]))
            review.addChild(QTreeWidgetItem(["note", review_note]))
        if options is not None:
            source = (
                "override (own options)" if result.options_source == OPTIONS_OVERRIDE else "batch"
            )
            opts = QTreeWidgetItem([f"Options ({source})", ""])
            opts.setFirstColumnSpanned(True)
            self.tree.addTopLevelItem(opts)
            defaults = DepthOptions().to_dict()
            for name, value in options.to_dict().items():
                child = QTreeWidgetItem([name, str(value)])
                if value != defaults[name]:
                    font = child.font(1)
                    font.setBold(True)
                    child.setFont(1, font)
                    child.setToolTip(1, f"default: {defaults[name]}")
                opts.addChild(child)
        self.tree.expandAll()
        self.tree.scrollToTop()
