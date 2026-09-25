"""``.dcc`` depth-correction files in the legacy Dcalib layout (plan 6.3, D9).

One line per corrected anode, no header::

    node board rena channel p0 p1 p2 \\n

Fields are separated by one space and the line ends with a **trailing space**
before the newline, exactly as ``Dcalib`` writes it (``outFile << p << " "``
per field, then ``endl``). The coefficients are those of
``f(r) = p0 + p1 r + p2 r^2`` in keV, with ``r = C/A``; consumers correct an
anode energy as ``A * 511 / f(r)`` (plan 4.3). Each coefficient uses the
default ``std::ostream`` formatting, which is C's ``%g`` (6 significant
digits); Python's ``format(x, "g")`` gives identical bytes (checked by uvcorr
against a compiled ``cout``). A coefficient with ``|p| < 1e-4`` is written as
``0``. MATLAB's ``importDcc.m`` reads the file with ``textscan('%d %d %d %d %f
%f %f')``, so a header line would break it: provenance goes to the CSV and the
sidecar instead.

:func:`read_dcc` reads the layout back, skipping ``#`` lines and lines with
fewer than 7 fields, as the time-calibration loader does.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from dcalib.channels import AnodeKey
from dcalib.io._atomic import atomic_write_text

__all__ = [
    "DCC_ZERO_THRESHOLD",
    "DccFormatError",
    "format_dcc",
    "format_dcc_coefficient",
    "format_dcc_line",
    "read_dcc",
    "write_dcc",
]

DCC_ZERO_THRESHOLD = 1e-4
"""Coefficients with an absolute value below this are written as ``0`` (legacy rule)."""

Coefficients = tuple[float, float, float]


class DccFormatError(ValueError):
    """A ``.dcc`` file or coefficient cannot be written or read."""


def format_dcc_coefficient(value: float) -> str:
    """Format one coefficient like ``Dcalib`` does.

    Raises:
        DccFormatError: If ``value`` is not finite (consumers cannot parse it).
    """
    number = float(value)
    if not math.isfinite(number):
        raise DccFormatError(f"cannot write a non-finite coefficient: {value!r}")
    if abs(number) < DCC_ZERO_THRESHOLD:
        return "0"
    return format(number, "g")


def format_dcc_line(key: AnodeKey | tuple[int, int, int, int], coefficients: Coefficients) -> str:
    """Return one ``.dcc`` line (with its trailing space and newline).

    Args:
        key: The anode ``(node, board, rena, channel)``.
        coefficients: ``(p0, p1, p2)`` in keV; unused higher terms are 0.
    """
    node, board, rena, channel = (int(v) for v in key)
    if len(coefficients) != 3:
        raise DccFormatError(f"expected 3 coefficients, got {len(coefficients)}")
    fields = [str(node), str(board), str(rena), str(channel)]
    fields += [format_dcc_coefficient(p) for p in coefficients]
    return " ".join(fields) + " \n"


def format_dcc(
    entries: Mapping[AnodeKey, Coefficients] | Iterable[tuple[AnodeKey, Coefficients]],
) -> str:
    """Return the text of a ``.dcc`` file, one line per entry sorted by key."""
    items = entries.items() if isinstance(entries, Mapping) else entries
    return "".join(format_dcc_line(key, c) for key, c in sorted(items, key=lambda kv: tuple(kv[0])))


def write_dcc(
    path: str | Path,
    entries: Mapping[AnodeKey, Coefficients] | Iterable[tuple[AnodeKey, Coefficients]],
) -> Path:
    """Write a ``.dcc`` file atomically (sorted by key; empty text for no entries)."""
    return atomic_write_text(path, format_dcc(entries), encoding="ascii")


def read_dcc(path: str | Path) -> dict[AnodeKey, Coefficients]:
    """Read a ``.dcc`` file.

    Lines starting with ``#`` and lines with fewer than 7 fields are skipped
    (the time-calibration loader's rule). A repeated anode keeps its last line.

    Raises:
        DccFormatError: If a line with 7 or more fields cannot be parsed.
    """
    entries: dict[AnodeKey, Coefficients] = {}
    with open(path, encoding="ascii") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.split()
            if not parts or parts[0].startswith("#") or len(parts) < 7:
                continue
            try:
                key = AnodeKey(*(int(v) for v in parts[:4]))
                p0, p1, p2 = (float(v) for v in parts[4:7])
            except ValueError as exc:
                raise DccFormatError(f"{path}:{line_num}: cannot parse {line.rstrip()!r}") from exc
            entries[key] = (p0, p1, p2)
    return entries
