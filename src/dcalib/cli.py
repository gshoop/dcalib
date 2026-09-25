"""Command-line interface for dcalib (backs the ``dcalib`` console script).

Subcommands (plan section 8):

``dcalib process CACHE --output-dir DIR [...]``
    Depth-calibrate every anode of an adc2kev calibration cache and write
    ``<name>.dcc`` and ``depth_summary.csv``. Phase 4.

``dcalib legacy CACHE --output-dir DIR [...]``
    Run the replica of the legacy C++ Dcalib and write ``<name>_legacy.dcc``
    and ``legacy_summary.csv``. Phase 2.

Both subcommands are registered but not implemented yet; they exit with status 1.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from dcalib import __version__

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _positive_int(value: str) -> int:
    """argparse type: an int >= 1 (for ``--workers`` and ``--min-pairs``)."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _finite_float(value: str) -> float:
    """argparse type: a finite float (for ``--min-gain``)."""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"must be finite, got {value}")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dcalib",
        description="CZT depth (C/A) photopeak calibration from an adc2kev calibration cache.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_process_parser(subparsers)
    _add_legacy_parser(subparsers)
    return parser


def _add_common_arguments(p: argparse.ArgumentParser) -> None:
    """Arguments shared by ``process`` and ``legacy``."""
    p.add_argument("cache", type=Path, help="adc2kev calibration cache (*.cache.h5).")
    p.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for the .dcc and CSV files."
    )
    p.add_argument(
        "--kev",
        type=Path,
        default=None,
        help="Use this .kev calibration file instead of the calibrations stored in the cache.",
    )


def _add_process_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``process`` subcommand."""
    p = subparsers.add_parser(
        "process",
        help="Depth-calibrate every anode and export .dcc + depth_summary.csv.",
        description=(
            "Build 1-anode/1-cathode events from the cache, fit every anode's photopeak "
            "position against C/A, and write <name>.dcc and depth_summary.csv."
        ),
    )
    _add_common_arguments(p)
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Sidecar results file (default: <cache stem>.depth.h5 next to the cache).",
    )
    p.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="Worker processes (default: min(8, CPU count)).",
    )
    p.add_argument(
        "--sources",
        choices=("both", "ge", "cs"),
        default=None,
        help="Sources to fit: Ge-68 (511 keV), Cs-137 (662 keV) or both (default: both).",
    )
    p.add_argument(
        "--min-pairs",
        type=_positive_int,
        default=None,
        help="Minimum selected events for an anode to be fitted.",
    )
    p.add_argument(
        "--max-degree",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Highest polynomial degree of the depth curve (default: 2).",
    )
    p.add_argument(
        "--min-gain",
        type=_finite_float,
        default=None,
        help="Minimum cross-validated relative FWHM gain for a correction to be accepted.",
    )
    p.add_argument(
        "--concave-only",
        action="store_true",
        help="Constrain the quadratic term to p2 <= 0, as the legacy fit did.",
    )
    p.add_argument(
        "--discard-overrides",
        action="store_true",
        help="Drop the per-anode re-fits stored by the GUI.",
    )
    p.add_argument(
        "--discard-review",
        action="store_true",
        help="Drop the per-anode Reject decisions stored by the GUI.",
    )
    p.set_defaults(func=_run_process)


def _add_legacy_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``legacy`` subcommand."""
    p = subparsers.add_parser(
        "legacy",
        help="Run the replica of the legacy C++ Dcalib (for cross-checks).",
        description=(
            "Reproduce the legacy C++ Dcalib algorithm on the cache's 1-anode/1-cathode "
            "events and write <name>_legacy.dcc and legacy_summary.csv."
        ),
    )
    _add_common_arguments(p)
    p.add_argument(
        "--energy",
        type=int,
        choices=(511, 662),
        default=511,
        help="Photopeak energy constant set and source (511: Ge-68, 662: Cs-137).",
    )
    p.add_argument(
        "--no-eof-quirk",
        action="store_true",
        help="Do not reproduce the C++ end-of-file re-read quirk.",
    )
    p.set_defaults(func=_run_legacy)


def _not_implemented(command: str, phase: int) -> int:
    print(f"dcalib {command}: not implemented yet (planned for phase {phase}).", file=sys.stderr)
    return EXIT_ERROR


def _run_process(args: argparse.Namespace) -> int:
    """Run ``dcalib process`` (phase 4)."""
    return _not_implemented(args.command, phase=4)


def _run_legacy(args: argparse.Namespace) -> int:
    """Run ``dcalib legacy`` (phase 2)."""
    return _not_implemented(args.command, phase=2)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        exit_code: int = args.func(args)
        return exit_code
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
