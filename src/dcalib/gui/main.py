"""Entry point of the ``dcalib-gui`` console script.

``dcalib-gui CACHE [--results PATH] [--kev KEV] [--workers N]`` (plan section 9).

This module is deliberately light: it imports only the standard library at
module level. Fit All runs the analysis in a ``spawn`` process pool, and every
spawned worker re-runs the parent's main script, so Qt, pyqtgraph and the
window are imported inside :func:`main`. The window arrives in phase 6.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

__all__ = ["main"]


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
    print("dcalib-gui: not implemented yet (planned for phase 6).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
