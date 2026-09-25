"""Summary CSVs: ``depth_summary.csv`` (plan 6.4) and ``legacy_summary.csv``.

Every CSV starts with ``#``-prefixed metadata lines (title, ``Generated``,
cache, calibration source, options) closed by a lone ``#``, as adc2kev's
resolution CSVs do, then the header row. Rows are sorted by (node, board,
rena, channel) and lines end with ``\\n``. Values never contain commas or
quotes.

``depth_summary.csv`` has one row per anode with any 1A1C event; its header is
:data:`dcalib.analysis.CSV_COLUMNS`. Cells: integers in decimal, an unavailable
value empty, the curve coefficients and their errors with ``%.9g`` and every
other float with ``%.6g``, ``flags`` joined with ``;``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from dcalib.analysis import (
    CSV_COLUMNS,
    KIND_COUNT,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_INT,
    RESULT_COLUMNS,
    AnodeResult,
)
from dcalib.channels import electrode_label
from dcalib.io._atomic import atomic_write_text
from dcalib.legacy import LegacyResult
from dcalib.options import FLAG_SEPARATOR

__all__ = [
    "LEGACY_COLUMNS",
    "LEGACY_SUMMARY_NAME",
    "SUMMARY_CSV_NAME",
    "format_legacy_summary",
    "format_summary_csv",
    "metadata_lines",
    "read_csv_rows",
    "read_summary_csv",
    "write_legacy_summary",
    "write_summary_csv",
]

SUMMARY_CSV_NAME = "depth_summary.csv"
PRECISE_COLUMNS = frozenset({"p0", "p1", "p2", "err_p0", "err_p1", "err_p2"})

LEGACY_SUMMARY_NAME = "legacy_summary.csv"

LEGACY_COLUMNS: tuple[str, ...] = (
    "node",
    "board",
    "rena",
    "channel",
    "electrode",
    "status",
    "n_events",
    "n_pairs",
    "n_peaks",
    "p0",
    "p1",
    "p2",
)
"""Columns of ``legacy_summary.csv``; p0-p2 are empty unless the status is ``ok``."""

FLOAT_FORMAT = ".9g"


def metadata_lines(title: str, items: Sequence[tuple[str, object]]) -> list[str]:
    """Return the ``#`` header lines: title, generation time, ``key: value`` items, ``#``."""
    lines = [f"# {title}", f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    lines += [f"# {key}: {value}" for key, value in items]
    lines.append("#")
    return lines


def _label(board: int, rena: int, channel: int) -> str:
    try:
        return electrode_label(board, rena, channel)
    except KeyError:
        return ""


def format_legacy_summary(
    results: Iterable[LegacyResult], metadata: Sequence[tuple[str, object]] = ()
) -> str:
    """Return the text of ``legacy_summary.csv``."""
    buffer = io.StringIO()
    for line in metadata_lines("DCALIB Legacy Replica Summary", metadata):
        buffer.write(line + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LEGACY_COLUMNS)
    for r in sorted(results, key=lambda r: r.key):
        coefficients = (
            [format(p, FLOAT_FORMAT) for p in r.coefficients]
            if r.coefficients is not None
            else ["", "", ""]
        )
        writer.writerow(
            [
                r.key.node,
                r.key.board,
                r.key.rena,
                r.key.channel,
                _label(r.key.board, r.key.rena, r.key.channel),
                r.status,
                r.n_events,
                r.n_pairs,
                r.n_peaks,
                *coefficients,
            ]
        )
    return buffer.getvalue()


def write_legacy_summary(
    path: str | Path,
    results: Iterable[LegacyResult],
    metadata: Sequence[tuple[str, object]] = (),
) -> Path:
    """Write ``legacy_summary.csv`` atomically."""
    return atomic_write_text(path, format_legacy_summary(results, metadata))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a dcalib CSV (skipping the ``#`` lines) as one dict per row."""
    with open(path, encoding="utf-8", newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines))


# ---------------------------------------------------------------------------
# depth_summary.csv
# ---------------------------------------------------------------------------


def _cell(name: str, kind: str, value: object) -> str:
    if value is None:
        return ""
    if kind == KIND_FLOAT and isinstance(value, (int, float)):
        return format(float(value), FLOAT_FORMAT if name in PRECISE_COLUMNS else ".6g")
    if kind == KIND_FLAGS and isinstance(value, tuple):
        return FLAG_SEPARATOR.join(value)
    return str(value)


def format_summary_csv(
    results: Iterable[AnodeResult], metadata: Sequence[tuple[str, object]] = ()
) -> str:
    """Return the text of ``depth_summary.csv`` (rows sorted by key)."""
    buffer = io.StringIO()
    for line in metadata_lines("DCALIB Depth Calibration Summary", metadata):
        buffer.write(line + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for result in sorted(results, key=lambda r: r.key):
        writer.writerow(
            _cell(column.name, column.kind, getattr(result, column.name))
            for column in RESULT_COLUMNS
        )
    return buffer.getvalue()


def write_summary_csv(
    path: str | Path,
    results: Iterable[AnodeResult],
    metadata: Sequence[tuple[str, object]] = (),
) -> Path:
    """Write ``depth_summary.csv`` atomically."""
    return atomic_write_text(path, format_summary_csv(results, metadata))


def read_summary_csv(path: str | Path) -> list[AnodeResult]:
    """Read a ``depth_summary.csv`` back (values at the written precision).

    Raises:
        ValueError: If the header is not :data:`CSV_COLUMNS` or a row is invalid.
    """
    rows = read_csv_rows(path)
    with open(path, encoding="utf-8") as f:
        header = next((line for line in f if not line.startswith("#")), "")
    if tuple(header.rstrip("\n").split(",")) != CSV_COLUMNS:
        raise ValueError(f"{path}: not a dcalib depth_summary.csv (unexpected header)")
    kinds = {column.name: column.kind for column in RESULT_COLUMNS}
    out = []
    for lineno, row in enumerate(rows, start=2):
        values: dict[str, object] = {}
        for name, text in row.items():
            kind = kinds[name]
            if kind in (KIND_INT, KIND_COUNT, KIND_FLOAT) and text == "":
                values[name] = None
            elif kind in (KIND_INT, KIND_COUNT):
                values[name] = int(text)
            elif kind == KIND_FLOAT:
                values[name] = float(text)
            else:
                values[name] = text
        try:
            out.append(AnodeResult.from_dict(values))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} row {lineno}: {exc}") from exc
    return out
