"""Command-line interface for dcalib (backs the ``dcalib`` console script).

Subcommands (plan section 8):

``dcalib process CACHE --output-dir DIR [--kev KEV] [--results PATH]
[--workers N] [--sources both|ge|cs] [--min-pairs N] [--max-degree {0,1,2}]
[--min-gain F] [--concave-only] [--discard-overrides] [--discard-review]``
    Depth-calibrate every anode of an adc2kev calibration cache (plan 8):
    write-test the output directory, check the cache, load and fingerprint the
    calibrations, read the stored overrides and review from the sidecar
    (unless discarded, or the overrides are stale), analyse every board in a
    process pool, write ``<name>.dcc`` and ``depth_summary.csv`` atomically
    (overrides applied, rejected anodes left out of the ``.dcc``), store the
    batch as ``/results/current`` of the sidecar, and print the census and
    timings. An override whose options fit like the new batch is dropped (the
    batch reproduces it).

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
import contextlib
import logging
import math
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

import h5py
import numpy as np
from adc2kev.cache import CalibrationCache

from dcalib import __version__
from dcalib.analysis import (
    AnalysisCancelled,
    AnalysisError,
    AnodeResult,
    analyze_all,
    default_workers,
    merge_results,
)
from dcalib.calib import CalibrationError, Calibrations, load_calibrations
from dcalib.events import list_boards
from dcalib.io import cache_stem
from dcalib.io.dcc import write_dcc
from dcalib.io.export import output_paths, prepare_output_dir, write_outputs
from dcalib.io.summary_csv import LEGACY_SUMMARY_NAME, write_legacy_summary
from dcalib.legacy import legacy_all, legacy_constants
from dcalib.options import FLAGS, STATUS_OK, STATUSES, DepthOptions
from dcalib.results import (
    ResultsError,
    ResultsFile,
    StoredResults,
    cache_identity,
    default_results_path,
)

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


def _depth_options(args: argparse.Namespace) -> DepthOptions:
    """``DepthOptions`` from the defaults plus the options given on the command line."""
    given: dict[str, Any] = {}
    if args.sources is not None:
        given["sources"] = args.sources
    if args.min_pairs is not None:
        given["min_pairs"] = args.min_pairs
    if args.max_degree is not None:
        given["max_degree"] = args.max_degree
    if args.min_gain is not None:
        given["min_gain"] = args.min_gain
    if args.concave_only:
        given["concave_only"] = True
    return DepthOptions(**given)


def _describe_options(options: DepthOptions) -> str:
    changed = options.changed_fields()
    return ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "defaults"


def _results_file(args: argparse.Namespace) -> ResultsFile:
    """The sidecar; an unwritable default location is an error suggesting ``--results``."""
    if args.results is not None:
        results = ResultsFile(args.results)
        results.check_writable()
        return results
    results = ResultsFile(default_results_path(args.cache))
    try:
        results.check_writable()
    except OSError as exc:
        raise OSError(f"{exc}; choose another location with --results PATH") from exc
    return results


def _run_process(args: argparse.Namespace) -> int:
    """Run ``dcalib process`` (see the module docstring)."""
    error = _check_inputs(args)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    try:
        options = _depth_options(args)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    name = cache_stem(args.cache)
    t_start = time.perf_counter()
    try:
        prepare_output_dir(args.output_dir, [p.name for p in output_paths(args.output_dir, name)])
        _check_cache(args.cache)
        calibrations = _load_calibrations(args)
        identity = cache_identity(args.cache)
        sidecar = _results_file(args)
        stored = sidecar.load() if sidecar.exists() else None
        t_setup = time.perf_counter()

        overrides, n_reproduced, stale_reasons = _usable_overrides(
            stored, identity, calibrations, options, discard=args.discard_overrides
        )
        review = {} if args.discard_review or stored is None else stored.review_states()
        if stored is None and not args.discard_review:
            review = {k: e.state for k, e in sidecar.load_review().items()}

        n_boards = len(list_boards(args.cache))
        workers = min(args.workers or default_workers(), max(n_boards, 1))
        print(
            f"Analysing {n_boards} boards with {workers} worker(s), options: "
            f"{_describe_options(options)}",
            file=sys.stderr,
        )
        analysis = _analyze_with_progress(args.cache, calibrations, options, workers)
        t_analysis = time.perf_counter()

        merged = merge_results(analysis.results, (o.result for o in overrides), review)
        metadata = _csv_metadata(args, calibrations, options, sidecar)
        dcc_path, csv_path = write_outputs(args.output_dir, name, merged, metadata)
        t_write = time.perf_counter()

        n_dropped = sidecar.save_batch(
            analysis.results,
            analysis.slices,
            options,
            identity,
            calibrations,
            keep_overrides=not args.discard_overrides,
        )
        if args.discard_review:
            sidecar.clear_review()
        t_store = time.perf_counter()
    except AnalysisCancelled:
        print("analysis cancelled; nothing was stored or written.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (AnalysisError, CalibrationError, ResultsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Results: {sidecar.path}")
    if stale_reasons:
        print("  the stored results were stale (" + "; ".join(stale_reasons) + ")")
    _print_census(merged, n_boards, workers, options)
    _print_review_census(args, overrides, n_reproduced, n_dropped, review, merged)
    n_lines = sum(1 for r in merged if r.exported)
    print("Outputs:")
    print(f"  {dcc_path}  ({n_lines:,} lines)")
    print(f"  {csv_path}  ({len(merged):,} rows)")
    print(
        f"Time: setup {t_setup - t_start:.1f} s, analysis {t_analysis - t_setup:.1f} s, "
        f"write {t_write - t_analysis:.1f} s, store {t_store - t_write:.1f} s, "
        f"total {t_store - t_start:.1f} s"
    )
    return EXIT_OK


def _usable_overrides(
    stored: StoredResults | None,
    identity: Any,
    calibrations: Calibrations,
    options: DepthOptions,
    *,
    discard: bool,
) -> tuple[list[Any], int, tuple[str, ...]]:
    """The stored overrides to apply, how many the batch reproduces, and stale reasons."""
    if stored is None:
        return [], 0, ()
    validity = stored.validity(identity, calibrations.fingerprint)
    if discard or validity.stale:
        return [], 0, validity.reasons
    kept, reproduced = [], 0
    for override in stored.overrides.values():
        if override.reproduced_by(options):
            reproduced += 1
        else:
            kept.append(override)
    return kept, reproduced, ()


def _csv_metadata(
    args: argparse.Namespace,
    calibrations: Calibrations,
    options: DepthOptions,
    sidecar: ResultsFile,
) -> list[tuple[str, object]]:
    return [
        ("cache", args.cache.resolve()),
        ("calibration", calibrations.source),
        ("calibration_fingerprint", calibrations.fingerprint),
        ("options", options.to_json()),
        ("results", sidecar.path.resolve()),
        ("dcalib_version", __version__),
    ]


def _analyze_with_progress(
    cache: Path, calibrations: Calibrations, options: DepthOptions, workers: int
) -> Any:
    stop = threading.Event()
    progress = _ProgressPrinter("analysing")

    def on_progress(done: int, total: int) -> None:
        progress(done / total if total else 1.0)

    try:
        with _sigint_sets(stop):
            return analyze_all(
                cache,
                calibrations,
                options,
                workers=workers,
                progress_cb=on_progress,
                stop_flag=stop,
            )
    finally:
        progress.close()


def _median(values: list[float]) -> str:
    return f"{float(np.median(values)):.2f}" if values else "-"


def _print_census(
    results: list[AnodeResult], n_boards: int, workers: int, options: DepthOptions
) -> None:
    """Print the status/flag census and the fleet resolution of the exported results."""
    n_events = sum(r.n_events for r in results)
    print(
        f"Analysis: {len(results):,} anodes ({n_events:,} 1A1C events) on {n_boards} boards, "
        f"{workers} worker(s), options: {_describe_options(options)}"
    )
    statuses = Counter(r.status for r in results)
    print("  status:     " + ", ".join(f"{s} {statuses.get(s, 0):,}" for s in STATUSES))
    flags = {flag: sum(1 for r in results if flag in r.flags) for flag in FLAGS}
    print("  flags:      " + (", ".join(f"{f} {n:,}" for f, n in flags.items() if n) or "none"))
    ok = [r for r in results if r.ok]
    for energy in (511, 662):
        for metric in ("fwhm", "fwtm"):
            before = [getattr(r, f"{metric}_{energy}_before") for r in ok]
            after = [getattr(r, f"{metric}_{energy}_after") for r in ok]
            pairs = [(b, a) for b, a in zip(before, after) if b is not None and a is not None]
            if pairs:
                b_values = [b for b, _ in pairs]
                a_values = [a for _, a in pairs]
                print(
                    f"  {metric.upper()} {energy} keV of ok anodes (median, %): "
                    f"{_median(b_values)} -> {_median(a_values)} ({len(pairs):,} anodes)"
                )
    gains = [r.cv_gain for r in ok if r.cv_gain is not None]
    if gains:
        print(f"  cv_gain of ok anodes: median {100 * float(np.median(gains)):.2f} %")
    for energy in (511, 662):
        exported = [getattr(r, f"peak_{energy}") for r in results if r.exported]
        omitted = [getattr(r, f"peak_{energy}") for r in results if not r.exported]
        e_vals = [v for v in exported if v is not None]
        o_vals = [v for v in omitted if v is not None]
        if e_vals or o_vals:
            e_text = f"{float(np.median(e_vals)):.4f}" if e_vals else "-"
            o_text = f"{float(np.median(o_vals)):.4f}" if o_vals else "-"
            print(
                f"  photopeak position {energy} keV (median, E/E0): corrected {e_text}, "
                f"omitted {o_text}"
            )


def _print_review_census(
    args: argparse.Namespace,
    overrides: list[Any],
    n_reproduced: int,
    n_dropped: int,
    review: dict[Any, str],
    merged: list[AnodeResult],
) -> None:
    if args.discard_overrides:
        print("  overrides:  discarded (--discard-overrides)")
    else:
        text = f"{len(overrides):,} applied"
        if n_reproduced:
            text += f", {n_reproduced:,} reproduced by the batch options (dropped)"
        elif n_dropped:
            text += f", {n_dropped:,} dropped"
        print(f"  overrides:  {text}")
    if args.discard_review:
        print("  review:     discarded (--discard-review)")
    else:
        rejected = sorted(k for k, state in review.items() if state)
        present = {r.key for r in merged}
        listed = ", ".join(str(k) for k in rejected[:10])
        more = f" and {len(rejected) - 10} more" if len(rejected) > 10 else ""
        orphans = sum(1 for k in rejected if k not in present)
        print(
            f"  rejected:   {len(rejected):,}"
            + (f" ({listed}{more})" if rejected else "")
            + (f"; {orphans} without a result" if orphans else "")
        )


@contextlib.contextmanager
def _sigint_sets(stop: threading.Event) -> Iterator[None]:
    """While active, Ctrl-C sets ``stop`` instead of raising KeyboardInterrupt."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(_signum: int, _frame: FrameType | None) -> None:
        if stop.is_set():
            print("\nstill stopping the analysis...", file=sys.stderr, flush=True)
            return
        stop.set()
        print("\nstopping the analysis...", file=sys.stderr, flush=True)

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


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
