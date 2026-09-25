#!/usr/bin/env python3
"""Compare the C++ Dcalib ``.dcc`` with dcalib's legacy replica (plan section 10.3).

::

    venv/bin/python scripts/compare_dcc.py CHD_DIR/Dcalib_output/chd.dcc \\
        --dcalib out/data_20260911_124617_legacy.dcc

Both files are read with :func:`dcalib.io.dcc.read_dcc` and compared per anode,
keyed by channel (the C++ file follows ``readdir`` order). By default the
comparison is restricted to the boards present in the C++ file, so a replica
run over the whole system can be compared with a C++ run on a few dumped
boards.

With fewer than 3 surviving peaks the C++ fit is undefined and still writes a
line, while the replica writes none (status ``fit_failed``). Those anodes are
read from the replica's ``legacy_summary.csv`` (``--summary``; by default the
file next to ``--dcalib``) and excluded from both sides.

A coefficient agrees when ``|a - b| <= rtol * max(|a|, |b|) + atol``. The
default ``rtol = 1e-3`` allows for Minuit's convergence tolerance and the
6-digit ``%g`` output; ``atol`` (default 1e-3 keV) covers coefficients that one
side writes as ``0`` (``|p| < 1e-4``) and the other as a tiny number. The
report also gives the largest difference of the curves f(r) over r in [0, 1],
in % of f.

Done when every compared anode agrees and the two channel sets are identical
(exit code 0; 1 otherwise, 2 for bad arguments or missing files).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dcalib.channels import AnodeKey
from dcalib.io.dcc import read_dcc
from dcalib.io.summary_csv import LEGACY_SUMMARY_NAME, read_csv_rows
from dcalib.options import STATUS_FIT_FAILED

Coefficients = tuple[float, float, float]
_R = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class Comparison:
    """The outcome of comparing two ``.dcc`` files."""

    compared: list[tuple[AnodeKey, Coefficients, Coefficients, bool, float]]
    only_cpp: list[AnodeKey]
    only_dcalib: list[AnodeKey]
    excluded: list[AnodeKey]

    @property
    def n_agree(self) -> int:
        return sum(1 for _, _, _, ok, _ in self.compared if ok)

    @property
    def ok(self) -> bool:
        return (
            self.n_agree == len(self.compared)
            and not self.only_cpp
            and not self.only_dcalib
            and bool(self.compared)
        )


def coefficients_agree(a: Coefficients, b: Coefficients, rtol: float, atol: float) -> bool:
    return all(abs(x - y) <= rtol * max(abs(x), abs(y)) + atol for x, y in zip(a, b))


def curve_difference_pct(a: Coefficients, b: Coefficients) -> float:
    """Largest ``|f_a(r) - f_b(r)| / |f_a(r)|`` over r in [0, 1], in %."""
    fa = a[0] + a[1] * _R + a[2] * _R**2
    fb = b[0] + b[1] * _R + b[2] * _R**2
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(fa - fb) / np.abs(fa)
    return float(100.0 * np.nanmax(rel))


def fit_failed_anodes(summary: Path) -> set[AnodeKey]:
    """Anodes whose replica status is ``fit_failed`` in a ``legacy_summary.csv``."""
    return {
        AnodeKey(int(row["node"]), int(row["board"]), int(row["rena"]), int(row["channel"]))
        for row in read_csv_rows(summary)
        if row["status"] == STATUS_FIT_FAILED
    }


def compare(
    cpp: dict[AnodeKey, Coefficients],
    dcalib: dict[AnodeKey, Coefficients],
    *,
    boards: set[tuple[int, int]] | None = None,
    exclude: Collection[AnodeKey] = (),
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Comparison:
    """Compare two ``.dcc`` contents (see the module docstring)."""
    if boards is None:
        boards = {(k.node, k.board) for k in cpp}
    cpp = {k: v for k, v in cpp.items() if (k.node, k.board) in boards}
    dcalib = {k: v for k, v in dcalib.items() if (k.node, k.board) in boards}
    excluded = sorted(
        k for k in exclude if (k.node, k.board) in boards and (k in cpp or k in dcalib)
    )
    for key in excluded:
        cpp.pop(key, None)
        dcalib.pop(key, None)
    compared = []
    for key in sorted(set(cpp) & set(dcalib)):
        a, b = cpp[key], dcalib[key]
        compared.append(
            (key, a, b, coefficients_agree(a, b, rtol, atol), curve_difference_pct(a, b))
        )
    return Comparison(
        compared=compared,
        only_cpp=sorted(set(cpp) - set(dcalib)),
        only_dcalib=sorted(set(dcalib) - set(cpp)),
        excluded=excluded,
    )


def _fmt(c: Coefficients) -> str:
    return " ".join(f"{p:11.6g}" for p in c)


def print_report(result: Comparison, worst: int = 10) -> None:
    n = len(result.compared)
    print(f"Compared {n} anodes: {result.n_agree} agree, {n - result.n_agree} differ")
    if result.excluded:
        print(f"Excluded {len(result.excluded)} anodes with < 3 peaks (fit_failed)")
    print(
        f"Only in the C++ file:  {len(result.only_cpp)}  {[str(k) for k in result.only_cpp[:10]]}"
    )
    print(
        f"Only in the dcalib file: {len(result.only_dcalib)}  "
        f"{[str(k) for k in result.only_dcalib[:10]]}"
    )
    if result.compared:
        diffs = np.array([d for *_, d in result.compared])
        print(
            f"Curve difference over r in [0, 1]: median {np.median(diffs):.2e} %, "
            f"max {diffs.max():.2e} %"
        )
        ranked = sorted(result.compared, key=lambda item: (item[3], -item[4]))[:worst]
        print(f"{'anode':>18}  {'C++ p0 p1 p2':>35}  {'dcalib p0 p1 p2':>35}  ok  max df %")
        for key, a, b, ok, d in ranked:
            print(f"{str(key):>18}  {_fmt(a)}  {_fmt(b)}  {'y' if ok else 'N'}  {d:.2e}")
    print("RESULT:", "PASS" if result.ok else "FAIL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_dcc.py", description="Compare the C++ Dcalib .dcc with dcalib legacy."
    )
    parser.add_argument("cpp", type=Path, help=".dcc written by the C++ Dcalib")
    parser.add_argument("--dcalib", type=Path, required=True, help=".dcc of dcalib legacy")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=f"replica's {LEGACY_SUMMARY_NAME} (default: next to --dcalib)",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance (1e-3)")
    parser.add_argument("--atol", type=float, default=1e-3, help="absolute tolerance (1e-3)")
    parser.add_argument("--worst", type=int, default=10, help="rows in the table (10)")
    args = parser.parse_args(argv)

    for path in (args.cpp, args.dcalib):
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    summary = args.summary if args.summary is not None else args.dcalib.parent / LEGACY_SUMMARY_NAME
    exclude = fit_failed_anodes(summary) if summary.is_file() else set()
    if not summary.is_file():
        print(f"warning: no {summary}: anodes with < 3 peaks are not excluded", file=sys.stderr)
    result = compare(
        read_dcc(args.cpp), read_dcc(args.dcalib), exclude=exclude, rtol=args.rtol, atol=args.atol
    )
    print_report(result, args.worst)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
