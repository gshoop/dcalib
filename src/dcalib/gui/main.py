"""Entry point of the ``dcalib-gui`` console script.

``dcalib-gui CACHE [--results PATH] [--kev KEV] [--workers N]`` (plan section 9).

This module is deliberately light: it imports only the standard library at
module level. Fit All runs the analysis in a ``spawn`` process pool, and every
spawned worker re-runs the parent's main script, so Qt, pyqtgraph and the
window are imported inside :func:`main` and the workers never pay for them.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]

_SIGNAL_POLL_MS = 250  # how often Python gets to run a pending SIGINT handler


def __getattr__(name: str) -> Any:
    """Lazy access to the window module's public names (``MainWindow``, ...)."""
    if name in ("MainWindow", "TAB_TITLES"):
        from dcalib.gui import window

        return getattr(window, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dcalib-gui",
        description="Inspect, re-fit and review CZT depth calibrations.",
    )
    parser.add_argument(
        "cache", nargs="?", type=Path, help="adc2kev calibration cache (*.cache.h5)"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="sidecar results file (default: <cache stem>.depth.h5 next to the cache)",
    )
    parser.add_argument(
        "--kev", type=Path, default=None, help="use this .kev file instead of the cache's"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes for Fit All (default: min(8, usable CPUs))",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="log progress to stderr (-v: info, -vv: debug)",
    )
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``dcalib-gui``. Returns a process exit code."""
    args = _parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from dcalib.gui.window import MainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    app.setOrganizationName("dcalib")
    app.setApplicationName("dcalib-gui")
    window = MainWindow(workers=args.workers, results_path=args.results, kev_path=args.kev)
    window.show()
    if args.cache is not None:
        cache: Path = args.cache
        QTimer.singleShot(0, lambda: window.open_cache(cache))

    # Ctrl+C in the terminal: Qt's event loop never returns to Python on its own, so a
    # timer wakes the interpreter to run the handler, which closes the window properly.
    previous = signal.signal(
        signal.SIGINT, lambda _signum, _frame: QTimer.singleShot(0, window.request_quit)
    )
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(_SIGNAL_POLL_MS)
    try:
        return int(app.exec())
    finally:
        wake.stop()
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    sys.exit(main())
