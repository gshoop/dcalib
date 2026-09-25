#!/usr/bin/env python3
"""Held-out validation of a depth calibration on another day's data (plan section 10.5).

::

    venv/bin/python scripts/validate_holdout.py                 # the test data, 8 workers
    venv/bin/python scripts/validate_holdout.py --run out/ --work-dir /scratch/holdout

1. The in-sample run: ``<name>.dcc`` and ``depth_summary.csv`` of ``dcalib
   process`` on the calibration cache (``--run DIR``). Without ``--run`` the
   script runs ``dcalib process`` itself into ``<work-dir>/insample`` (its
   sidecar goes there too, not next to the cache).
2. adc2kev ``DiagnosticCache``\\ s of the two raw acquisitions of the held-out
   day (``--ge``, ``--cs``) are built in ``--work-dir`` (reused while valid).
   The work directory must not be inside the raw data's or the cache's
   directory tree.
3. The 1-anode/1-cathode events of both files are built with
   :mod:`dcalib.events` (a diagnostic cache is single-source: the Ge file's
   events are source 0, the Cs file's source 1) and converted with the
   *in-sample* calibrations (the cache's, or ``--kev``).
4. For every anode of the in-sample CSV: the held-out FWHM and FWTM before and
   after the in-sample ``.dcc`` correction at 511 and 662 keV
   (:mod:`dcalib.metrics`), the photopeak positions, and for the corrected
   anodes the held-out alignment gain (the ``cv_gain`` estimator of
   :mod:`dcalib.depth` with the in-sample curve on the held-out events).

Outputs in the work directory: ``holdout_summary.csv`` (one row per anode)
and ``holdout_report.md`` (the fleet summary, also printed). The comparison is
paired (the same held-out events before and after), so a day of gain drift
shifts both and largely cancels; the report shows the drift as the photopeak
position of the uncorrected held-out spectra.

Exit codes: 0 on success, 1 on an error, 2 for bad arguments or missing files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import multiprocessing
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from dcalib import cli
from dcalib.analysis import AnodeResult
from dcalib.calib import BoardLUT, Calibrations, event_energies, load_calibrations
from dcalib.channels import AnodeKey
from dcalib.depth import _peak, _seeds, alignment, width_ratio
from dcalib.events import BoardEvents, build_board_events, list_boards
from dcalib.io import cache_stem
from dcalib.io.dcc import read_dcc
from dcalib.io.summary_csv import SUMMARY_CSV_NAME, read_summary_csv
from dcalib.metrics import FWHM_PER_SIGMA, MIN_METRIC_EVENTS, fit_photopeak, fwhm_pct, fwtm_pct
from dcalib.options import DCC_ENERGY_KEV, SOURCE_CS, SOURCE_GE, DepthOptions

TEST_DATA = Path.home() / "adc2kev-test-data/full-system"
DEFAULT_CACHE = TEST_DATA / "data_20260911_124617.cache.h5"
DEFAULT_GE = TEST_DATA / "sources/ge/data_20260910_120628.dat"
DEFAULT_CS = TEST_DATA / "sources/cs/data_20260910_121135.dat"
DEFAULT_WORK_DIR = Path(".scratch/holdout")

METRICS = ("fwhm", "fwtm")
ENERGIES = {SOURCE_GE: 511, SOURCE_CS: 662}
Coefficients = tuple[float, float, float]


@dataclass(frozen=True)
class HoldoutRow:
    """One anode's held-out measurements (a ``holdout_summary.csv`` row)."""

    key: AnodeKey
    status: str
    exported: bool
    n_ge: int
    n_cs: int
    values: dict[str, float | None]
    holdout_gain: float | None
    cv_gain: float | None


# ---------------------------------------------------------------------------
# Per board (runs in the worker processes)
# ---------------------------------------------------------------------------


def _events(ge_cache: Path, cs_cache: Path, node: int, board: int, window: int) -> BoardEvents:
    """Both files' 1A1C events of a board, the Cs file's relabelled as source 1."""
    parts = []
    for path, source in ((ge_cache, SOURCE_GE), (cs_cache, SOURCE_CS)):
        ev = build_board_events(path, node, board, window)
        parts.append(replace(ev, source=np.full(len(ev), source, dtype=np.int8)))
    rows = {
        name: np.concatenate([getattr(p, name) for p in parts]) for name in BoardEvents.ROW_FIELDS
    }
    census = parts[0].census.copy()
    census[SOURCE_CS] = parts[1].census[SOURCE_GE]
    return BoardEvents(node, board, window, census=census, **rows)


def _g(coefficients: Coefficients, r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p0, p1, p2 = coefficients
    return np.asarray((p0 + r * (p1 + r * p2)) / DCC_ENERGY_KEV, dtype=np.float64)


def _none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def board_task(
    ge_cache: str,
    cs_cache: str,
    node: int,
    board: int,
    lut: BoardLUT,
    anodes: Mapping[tuple[int, int], tuple[str, bool, float | None]],
    curves: Mapping[tuple[int, int], Coefficients],
    options: DepthOptions,
) -> list[HoldoutRow]:
    """Held-out measurements of one board's anodes."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)
    events = _events(Path(ge_cache), Path(cs_cache), node, board, options.cts_window)
    energies = event_energies(events, lut)
    usable = energies.anode_calibrated & energies.cathode_calibrated
    rows: list[HoldoutRow] = []
    for (rena, channel), index in events.anode_rows().items():
        if (rena, channel) not in anodes:
            continue
        status, exported, cv_gain = anodes[(rena, channel)]
        index = index[usable[index]]
        x, r, source = energies.x[index], energies.r[index], events.source[index]
        in_r = (r >= options.r_lo) & (r <= options.r_hi) & np.isfinite(x) & np.isfinite(r)
        curve = curves.get((rena, channel)) if exported else None
        values: dict[str, float | None] = {}
        gains = []
        for source_id, energy in ENERGIES.items():
            mask = in_r & (source == source_id)
            before = x[mask]
            after = before / _g(curve, r[mask]) if curve is not None else before
            for metric, width in (("fwhm", fwhm_pct), ("fwtm", fwtm_pct)):
                values[f"{metric}_{energy}_before"] = _none(width(before))
                values[f"{metric}_{energy}_after"] = _none(width(after))
            peak = fit_photopeak(before) if len(before) >= MIN_METRIC_EVENTS else None
            values[f"peak_{energy}_before"] = peak.mu if peak is not None and peak.ok else None
            if curve is not None and len(before) >= MIN_METRIC_EVENTS:
                aligned = alignment(
                    before, after, r[mask], options, *_seeds(_peak(before, options, None))
                )
                sigma = fwhm_pct(after) / 100.0 / FWHM_PER_SIGMA
                ratio = width_ratio(aligned, sigma)
                if math.isfinite(ratio):
                    gains.append(1.0 - ratio)
        rows.append(
            HoldoutRow(
                key=AnodeKey(node, board, rena, channel),
                status=status,
                exported=exported,
                n_ge=int(np.count_nonzero(source == SOURCE_GE)),
                n_cs=int(np.count_nonzero(source == SOURCE_CS)),
                values=values,
                holdout_gain=float(np.mean(gains)) if gains else None,
                cv_gain=cv_gain,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _check_work_dir(work_dir: Path, protected: Iterable[Path]) -> str | None:
    resolved = work_dir.resolve()
    for path in protected:
        root = path.resolve().parent
        if resolved == root or root in resolved.parents:
            return f"--work-dir {work_dir} is inside {root}, which must not be written to"
    return None


def _build_diag_cache(dat: Path, work_dir: Path) -> Path:
    from adc2kev.cache import DiagnosticCache

    path = work_dir / f"{dat.stem}.diag.h5"
    cache = DiagnosticCache(path)
    if cache.is_valid_for(dat):
        print(f"Diagnostic cache {path}: valid, reused")
        return path
    t0 = time.perf_counter()
    printer = cli._ProgressPrinter(f"parsing {dat.name}")
    try:
        n = cache.build_from_dat(dat, progress_callback=printer)
    finally:
        printer.close()
    print(f"Diagnostic cache {path}: {n:,} events parsed in {time.perf_counter() - t0:.0f} s")
    return path


def _csv_options(path: Path) -> DepthOptions:
    """The ``options`` metadata line of a ``depth_summary.csv`` (defaults if absent)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# options: "):
                return DepthOptions.from_json(line[len("# options: ") :].strip(), strict=False)
    return DepthOptions()


def _csv_fingerprint(path: Path) -> str | None:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# calibration_fingerprint: "):
                return line.split(":", 1)[1].strip()
    return None


def _insample(args: argparse.Namespace) -> Path:
    if args.run is not None:
        return Path(args.run)
    run: Path = Path(args.work_dir) / "insample"
    name = cache_stem(args.cache)
    if (run / SUMMARY_CSV_NAME).is_file() and (run / f"{name}.dcc").is_file():
        print(f"In-sample run {run}: reused")
        return run
    argv = [
        "process",
        str(args.cache),
        "--output-dir",
        str(run),
        "--results",
        str(run / f"{name}.depth.h5"),
        "--workers",
        str(args.workers),
    ]
    if args.kev is not None:
        argv += ["--kev", str(args.kev)]
    print("Running dcalib " + " ".join(argv))
    if cli.main(argv) != 0:
        raise RuntimeError("the in-sample dcalib process failed")
    return run


def _median(values: Iterable[float | None]) -> float:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else math.nan


def _ratios(rows: list[HoldoutRow], metric: str, energy: int) -> npt.NDArray[np.float64]:
    out = []
    for row in rows:
        before = row.values.get(f"{metric}_{energy}_before")
        after = row.values.get(f"{metric}_{energy}_after")
        if before and after:
            out.append(after / before)
    return np.array(out)


def report(
    rows: list[HoldoutRow],
    insample: Mapping[AnodeKey, AnodeResult],
    meta: Mapping[str, Any],
) -> str:
    """The markdown fleet summary."""
    corrected = [r for r in rows if r.exported]
    buf = io.StringIO()
    w = buf.write
    w("# Held-out validation\n\n")
    for key, value in meta.items():
        w(f"- {key}: {value}\n")
    w(
        f"\nAnodes with held-out events: {len(rows):,}; corrected (in the `.dcc`): {len(corrected):,}.\n\n"
    )
    w("## Corrected anodes: held-out widths before and after the in-sample correction\n\n")
    w(
        "| Metric | Median before | Median after | Ratio after/before p10 / median / p90 | Worse by > 1 % | In-sample median ratio |\n"
    )
    w("|---|---|---|---|---|---|\n")
    for energy in ENERGIES.values():
        for metric in METRICS:
            ratios = _ratios(corrected, metric, energy)
            ins = []
            for r in corrected:
                s = insample.get(r.key)
                b = getattr(s, f"{metric}_{energy}_before", None) if s else None
                a = getattr(s, f"{metric}_{energy}_after", None) if s else None
                if b and a:
                    ins.append(a / b)
            if not len(ratios):
                continue
            w(
                f"| {metric.upper()} {energy} keV "
                f"| {_median(r.values.get(f'{metric}_{energy}_before') for r in corrected):.2f} % "
                f"| {_median(r.values.get(f'{metric}_{energy}_after') for r in corrected):.2f} % "
                f"| {np.percentile(ratios, 10):.3f} / {np.median(ratios):.3f} / {np.percentile(ratios, 90):.3f} "
                f"| {100 * np.mean(ratios > 1.01):.1f} % "
                f"| {np.median(ins):.3f} |\n"
            )
    gains = np.array([r.holdout_gain for r in corrected if r.holdout_gain is not None])
    cv = np.array(
        [r.cv_gain for r in corrected if r.holdout_gain is not None and r.cv_gain is not None]
    )
    both = [
        (r.cv_gain, r.holdout_gain)
        for r in corrected
        if r.holdout_gain is not None and r.cv_gain is not None
    ]
    w("\n## Alignment gain (the `cv_gain` estimator) on the held-out events\n\n")
    if len(gains):
        w(
            f"- held-out gain of the corrected anodes: p10 {100 * np.percentile(gains, 10):.2f} %, "
            f"median {100 * np.median(gains):.2f} %, p90 {100 * np.percentile(gains, 90):.2f} %; "
            f"negative for {100 * np.mean(gains < 0):.1f} %\n"
        )
    if len(cv):
        w(f"- in-sample `cv_gain` of the same anodes: median {100 * np.median(cv):.2f} %\n")
    if len(both) > 2:
        c, h = np.array(both).T
        w(f"- correlation of `cv_gain` and the held-out gain: {np.corrcoef(c, h)[0, 1]:.2f}\n")
        w(f"- median held-out minus cv gain: {100 * np.median(h - c):+.2f} points\n")
    w("\n## Gain drift between the days\n\n")
    for energy in ENERGIES.values():
        peaks = [r.values.get(f"peak_{energy}_before") for r in rows]
        w(
            f"- uncorrected held-out photopeak position at {energy} keV (median, E/E0, with the "
            f"in-sample calibration): {_median(peaks):.4f}\n"
        )
    return buf.getvalue()


def write_rows(path: Path, rows: list[HoldoutRow]) -> None:
    value_names = [
        f"{m}_{e}_{w}" for e in ENERGIES.values() for m in METRICS for w in ("before", "after")
    ] + [f"peak_{e}_before" for e in ENERGIES.values()]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["node", "board", "rena", "channel", "status", "exported", "n_ge", "n_cs"]
            + value_names
            + ["holdout_gain", "cv_gain"]
        )
        for row in sorted(rows, key=lambda r: r.key):
            cells: list[Any] = [*row.key, row.status, int(row.exported), row.n_ge, row.n_cs]
            cells += [
                "" if row.values.get(n) is None else f"{row.values[n]:.6g}" for n in value_names
            ]
            cells += ["" if v is None else f"{v:.6g}" for v in (row.holdout_gain, row.cv_gain)]
            writer.writerow(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_holdout.py", description="Held-out validation of a depth calibration."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="calibration cache")
    parser.add_argument("--kev", type=Path, default=None, help=".kev instead of the cache's")
    parser.add_argument("--run", type=Path, default=None, help="in-sample dcalib process output")
    parser.add_argument("--ge", type=Path, default=DEFAULT_GE, help="held-out Ge-68 .dat")
    parser.add_argument("--cs", type=Path, default=DEFAULT_CS, help="held-out Cs-137 .dat")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="scratch")
    parser.add_argument("--workers", type=int, default=8, help="worker processes (8)")
    args = parser.parse_args(argv)

    for path in (args.cache, args.ge, args.cs, *(p for p in [args.kev] if p is not None)):
        if not Path(path).is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    error = _check_work_dir(args.work_dir, (args.cache, args.ge, args.cs))
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        run = _insample(args)
        csv_path = run / SUMMARY_CSV_NAME
        insample = {r.key: r for r in read_summary_csv(csv_path)}
        curves = read_dcc(run / f"{cache_stem(args.cache)}.dcc")
        options = _csv_options(csv_path)
        calibrations: Calibrations = load_calibrations(args.cache, args.kev)
        expected = _csv_fingerprint(csv_path)
        if expected is not None and expected != calibrations.fingerprint:
            print(
                "warning: these calibrations are not the in-sample run's "
                f"(fingerprint {calibrations.fingerprint[:12]} vs {expected[:12]})",
                file=sys.stderr,
            )
        ge_cache = _build_diag_cache(args.ge, args.work_dir)
        cs_cache = _build_diag_cache(args.cs, args.work_dir)
        boards = sorted(set(list_boards(ge_cache)) | set(list_boards(cs_cache)))
        by_board: dict[tuple[int, int], dict[tuple[int, int], tuple[str, bool, float | None]]] = {}
        for key, row in insample.items():
            by_board.setdefault((key.node, key.board), {})[(key.rena, key.channel)] = (
                row.status,
                key in curves and row.exported,
                row.cv_gain,
            )
        tasks = [nb for nb in boards if nb in by_board]
        jobs = [
            (
                str(ge_cache),
                str(cs_cache),
                node,
                board,
                calibrations.board_lut(node, board),
                by_board[(node, board)],
                {
                    (k.rena, k.channel): c
                    for k, c in curves.items()
                    if (k.node, k.board) == (node, board)
                },
                options,
            )
            for node, board in tasks
        ]
        rows: list[HoldoutRow] = []
        printer = cli._ProgressPrinter("held-out boards")
        if args.workers <= 1:
            for done, job in enumerate(jobs, start=1):
                rows.extend(board_task(*job))
                printer(done / len(jobs))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                futures = [pool.submit(board_task, *job) for job in jobs]
                for done, future in enumerate(as_completed(futures), start=1):
                    rows.extend(future.result())
                    printer(done / len(futures))
        printer.close()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary_path = args.work_dir / "holdout_summary.csv"
    write_rows(summary_path, rows)
    meta = {
        "in-sample run": run.resolve(),
        "calibration": calibrations.source,
        "held-out Ge-68": args.ge,
        "held-out Cs-137": args.cs,
        "options": json.dumps(options.changed_fields()) if options.changed_fields() else "defaults",
        "run time": f"{time.perf_counter() - t0:.0f} s",
    }
    text = report(rows, insample, meta)
    (args.work_dir / "holdout_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {summary_path} and {args.work_dir / 'holdout_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
