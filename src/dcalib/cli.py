"""Command-line interface for dcalib (backs the ``dcalib`` console script).

Subcommands (plan section 8):

``dcalib process CACHE --output-dir DIR [...]``
    Depth-calibrate every anode of an adc2kev calibration cache and write
    ``<name>.dcc`` and ``depth_summary.csv``. Phase 4.

``dcalib legacy CACHE --output-dir DIR [--kev KEV] [--energy {511,662}]
[--no-eof-quirk]``
    Run the replica of the legacy C++ Dcalib (:mod:`dcalib.legacy`) on the
    cache's 1-anode/1-cathode events of one source (Ge-68 for 511, Cs-137 for
    662) and write ``<name>_legacy.dcc`` and ``legacy_summary.csv``. It never
    touches the sidecar results file.

Exit codes: 0 on success, 1 on an error, 2 for a missing input file or bad
arguments, 130 when stopped with Ctrl-C.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TextIO

import h5py
from adc2kev.cache import CalibrationCache

from dcalib import __version__
from dcalib.calib import CalibrationError, Calibrations, load_calibrations
from dcalib.io import cache_stem
from dcalib.io.dcc import write_dcc
from dcalib.io.export import prepare_output_dir
from dcalib.io.summary_csv import LEGACY_SUMMARY_NAME, write_legacy_summary
from dcalib.legacy import legacy_all, legacy_constants
from dcalib.options import STATUS_OK

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


class _ProgressPrinter:
    """Throttled progress callback that writes to a stream (stderr).

    On a terminal the line is redrawn in place at every whole percent; on a
    pipe or file a new line is written every 10 %.
    """

    def __init__(self, label: str, stream: TextIO | None = None) -> None:
        self._label = label
        self._stream = stream if stream is not None else sys.stderr
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._step = 1 if self._tty else 10
        self._last = -1
        self._t0 = time.perf_counter()

    def __call__(self, fraction: float) -> None:
        pct = max(0, min(100, int(fraction * 100)))
        bucket = pct // self._step
        if bucket == self._last:
            return
        self._last = bucket
        elapsed = time.perf_counter() - self._t0
        text = f"{self._label}: {pct:3d}% ({elapsed:5.1f} s)"
        if self._tty:
            print(f"\r{text}", end="", file=self._stream, flush=True)
        else:
            print(text, file=self._stream, flush=True)

    def close(self) -> None:
        """End the in-place progress line (terminal only)."""
        if self._tty and self._last >= 0:
            print(file=self._stream, flush=True)


def _check_inputs(args: argparse.Namespace) -> str | None:
    """Return an error message for a missing input or a bad output directory."""
    cache: Path = args.cache
    if not cache.is_file():
        return f"cache file not found: {cache}"
    if args.kev is not None and not args.kev.is_file():
        return f"--kev file not found: {args.kev}"
    output_dir: Path = args.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        return f"--output-dir {output_dir} exists and is not a directory"
    return None


def _check_cache(cache: Path) -> None:
    """Raise ValueError unless the cache is a structurally valid adc2kev cache."""
    if not h5py.is_hdf5(cache):
        raise ValueError(f"{cache} is not an HDF5 file")
    valid, message = CalibrationCache(cache).is_cache_structurally_valid()
    if not valid:
        raise ValueError(f"{cache} is not a usable adc2kev calibration cache: {message}")


def _load_calibrations(args: argparse.Namespace) -> Calibrations:
    calibrations = load_calibrations(args.cache, args.kev)
    source = "cache" if args.kev is None else str(args.kev)
    print(
        f"Calibrations: {len(calibrations):,} valid channels from {source} "
        f"(fingerprint {calibrations.fingerprint[:12]})",
        file=sys.stderr,
    )
    return calibrations


def _run_legacy(args: argparse.Namespace) -> int:
    """Run ``dcalib legacy``: the C++ replica on every board, then the two outputs."""
    error = _check_inputs(args)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    constants = legacy_constants(args.energy)
    eof_quirk = not args.no_eof_quirk
    name = cache_stem(args.cache)
    t_start = time.perf_counter()
    try:
        dcc_path, csv_path = prepare_output_dir(
            args.output_dir, [f"{name}_legacy.dcc", LEGACY_SUMMARY_NAME]
        )
        _check_cache(args.cache)
        calibrations = _load_calibrations(args)
        progress = _ProgressPrinter("legacy replica")
        try:
            results = legacy_all(
                args.cache,
                calibrations,
                constants,
                eof_quirk=eof_quirk,
                progress=lambda done, total: progress(done / total if total else 1.0),
            )
        finally:
            progress.close()
        written = {r.key: r.coefficients for r in results if r.coefficients is not None}
        write_dcc(dcc_path, written)
        metadata = [
            ("cache", args.cache.resolve()),
            ("calibration", "cache" if args.kev is None else args.kev.resolve()),
            ("calibration_fingerprint", calibrations.fingerprint),
            ("energy_kev", constants.energy),
            ("eof_quirk", "on" if eof_quirk else "off"),
            ("dcalib_version", __version__),
        ]
        write_legacy_summary(csv_path, results, metadata)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    statuses = Counter(r.status for r in results)
    print(
        f"Legacy replica ({constants.energy} keV, EOF quirk {'on' if eof_quirk else 'off'}): "
        f"{len(results):,} anodes with events"
    )
    print("  status: " + ", ".join(f"{s} {n:,}" for s, n in sorted(statuses.items())))
    print("Outputs:")
    print(f"  {dcc_path}  ({statuses.get(STATUS_OK, 0):,} lines)")
    print(f"  {csv_path}  ({len(results):,} rows)")
    print(f"Time: {time.perf_counter() - t_start:.1f} s")
    return EXIT_OK


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
