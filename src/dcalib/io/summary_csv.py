"""Summary CSV writers: ``legacy_summary.csv`` (phase 2) and ``depth_summary.csv`` (phase 4).

Every CSV starts with ``#``-prefixed metadata lines (title, ``Generated``,
cache, calibration source, options) closed by a lone ``#``, as adc2kev's
resolution CSVs do, then the header row. Rows are sorted by (node, board,
rena, channel) and lines end with ``\\n``. Values never contain commas or
quotes.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from dcalib.channels import electrode_label
from dcalib.io._atomic import atomic_write_text
from dcalib.legacy import LegacyResult

__all__ = [
    "LEGACY_COLUMNS",
    "LEGACY_SUMMARY_NAME",
    "format_legacy_summary",
    "metadata_lines",
    "read_csv_rows",
    "write_legacy_summary",
]

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
