#!/usr/bin/env python3
"""Render the screenshots of ``docs/GUI.md`` with Qt's offscreen platform.

::

    venv/bin/python scripts/gui_screenshots.py CACHE --results RESULTS --out docs/images

Opens ``CACHE`` with its stored results in ``dcalib-gui`` on a virtual
1920 x 1200 screen (so the window takes its default 85 %, 1632 x 1020, with
the default dock sizes) and saves:

- ``main_window.png``: the whole window on the Depth tab (``--anode``).
- ``depth_kev.png``: the Depth tab in keV (both photopeaks).
- ``depth_ridges.png``: the Depth tab with the cathode ridges.
- ``spectra.png``, ``board_grid.png``, ``fleet_summary.png``: the other tabs.
- ``system_map_metric.png``: the System Map coloured by ``cv_gain``.
- ``system_map_cathodes.png``: the System Map's cathode view (keV calibration).
- ``review.png``: the whole window after a Ge-only *Fit Channel* override of
  ``--override`` and a rejection of ``--reject``.

The results file is copied to a temporary directory first (the override and
the rejection are stored in the copy; ``RESULTS`` is only read), and the GUI
settings go to a temporary directory as well, so neither the user's results
nor their layout change. Exit codes: 0, 1 (the GUI reported an error or timed
out), 2 (bad arguments or missing files).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

SCREEN = (1920, 1200)
TIMEOUT_S = 120.0
TAB_SHOTS = (("spectra.png", 1), ("board_grid.png", 2), ("fleet_summary.png", 3))


def _key(text: str) -> tuple[int, int, int, int]:
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected node,board,rena,channel, got {text!r}")
    return parts[0], parts[1], parts[2], parts[3]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path, help="adc2kev calibration cache (*.cache.h5)")
    parser.add_argument("--results", type=Path, required=True, help="stored results (.depth.h5)")
    parser.add_argument("--out", type=Path, default=Path("docs/images"), help="output directory")
    parser.add_argument(
        "--anode", type=_key, default=(8, 28, 0, 10), help="anode of the main screenshots"
    )
    parser.add_argument(
        "--override", type=_key, default=(8, 28, 1, 7), help="anode re-fitted with Ge only"
    )
    parser.add_argument("--reject", type=_key, default=(8, 28, 1, 21), help="anode rejected")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in (args.cache, args.results):
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
    with tempfile.TemporaryDirectory(prefix="dcalib-shots-") as tmp:
        work = Path(tmp)
        screen = work / "screen.json"
        screen.write_text(
            json.dumps(
                {
                    "synchronousWindowSystemEvents": False,
                    "windowFrameMargins": False,
                    "screens": [
                        {
                            "name": "shots",
                            "x": 0,
                            "y": 0,
                            "width": SCREEN[0],
                            "height": SCREEN[1],
                            "logicalDpi": 96,
                            "logicalBaseDpi": 96,
                            "dpr": 1,
                        }
                    ],
                }
            )
        )
        os.environ["QT_QPA_PLATFORM"] = f"offscreen:configfile={screen}"
        results = work / args.results.name
        shutil.copy2(args.results, results)
        return _render(args, results, work / "settings")


def _render(args: argparse.Namespace, results: Path, settings_dir: Path) -> int:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QWidget

    from dcalib.channels import AnodeKey
    from dcalib.gui.depth_view import DepthDisplay
    from dcalib.gui.system_map import VIEW_ANODES, VIEW_CATHODES

    app = QApplication(sys.argv[:1])
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(settings_dir))
    from dcalib.gui.window import MainWindow

    window = MainWindow(workers=1, results_path=results)
    errors: list[str] = []
    window._show_error = lambda title, text: errors.append(f"{title}: {text}")  # type: ignore[method-assign]
    window.show()

    def spin(done: Callable[[], bool], settle: float = 0.3) -> bool:
        start = time.monotonic()
        while not done():
            if errors or time.monotonic() - start > TIMEOUT_S:
                return False
            app.processEvents()
            time.sleep(0.02)
        end = time.monotonic() + settle
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.02)
        return True

    def idle() -> bool:
        return window.session.is_open and not window.is_busy and not window.data_pending

    def save(widget: QWidget, name: str) -> None:
        path = args.out / name
        widget.grab().save(str(path))
        print(f"wrote {path} ({widget.width()} x {widget.height()})")

    def show(key: AnodeKey) -> bool:
        window.select_anode(key)
        return spin(idle)

    args.out.mkdir(parents=True, exist_ok=True)
    window.open_cache(args.cache)
    anode = AnodeKey(*args.anode)
    ok = spin(idle) and show(anode)
    if ok:
        window.tabs.setCurrentIndex(0)
        window.depth_view.set_display(DepthDisplay())
        ok = spin(lambda: True)
        save(window, "main_window.png")
        for name, display in (
            ("depth_kev.png", DepthDisplay(units="kev", window=False)),
            ("depth_ridges.png", DepthDisplay(source_curves=False, ridges=True, window=False)),
        ):
            window.depth_view.set_display(display)
            spin(lambda: True)
            save(window.tabs, name)
        window.depth_view.set_display(DepthDisplay())
        for name, index in TAB_SHOTS:
            window.tabs.setCurrentIndex(index)
            spin(lambda: True)
            save(window.tabs, name)
        window.tabs.setCurrentIndex(0)
        window.system_map.set_color_mode("cv_gain")
        spin(lambda: True)
        save(window.system_map_dock, "system_map_metric.png")
        window.system_map.set_color_mode("status")
        window.system_map.set_view(VIEW_CATHODES)
        spin(lambda: True)
        save(window.system_map_dock, "system_map_cathodes.png")
        window.system_map.set_view(VIEW_ANODES)
        # Review: a Ge-only override and a rejection, stored in the copy.
        override = AnodeKey(*args.override)
        ok = show(override)
        batch_options = window.band.options
        window.band.set_options(replace(batch_options, sources="ge"))
        ok = ok and window.refit_channel(override) and spin(idle)
        window.band.set_options(batch_options)
        ok = ok and window.set_rejected(
            AnodeKey(*args.reject), True, "screenshot: example rejection"
        )
        ok = ok and show(override)
        if ok:
            save(window, "review.png")
    window.close()
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 0 if ok and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
