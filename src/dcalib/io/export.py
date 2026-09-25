"""Output-directory checks and the export of ``<name>.dcc`` + ``depth_summary.csv``.

:func:`prepare_output_dir` checks the output location *before* an analysis, so
a long run does not end in a permission error. :func:`write_outputs` writes
both files through ``fsync``-ed temporary files renamed back to back, so a
failure leaves both previous files untouched.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from dcalib.analysis import AnodeResult
from dcalib.channels import AnodeKey
from dcalib.io._atomic import atomic_write_texts
from dcalib.io.dcc import format_dcc
from dcalib.io.summary_csv import SUMMARY_CSV_NAME, format_summary_csv

__all__ = ["dcc_entries", "output_paths", "prepare_output_dir", "write_outputs"]


def output_paths(output_dir: str | Path, name: str) -> tuple[Path, Path]:
    """The ``(<name>.dcc, depth_summary.csv)`` paths in ``output_dir``."""
    directory = Path(output_dir)
    return directory / f"{name}.dcc", directory / SUMMARY_CSV_NAME


def dcc_entries(results: Iterable[AnodeResult]) -> dict[AnodeKey, tuple[float, float, float]]:
    """The ``.dcc`` lines of the results: ``ok`` anodes that are not rejected (D7, D10)."""
    entries: dict[AnodeKey, tuple[float, float, float]] = {}
    for result in results:
        coefficients = result.coefficients
        if result.exported and coefficients is not None:
            entries[result.key] = coefficients
    return entries


def write_outputs(
    output_dir: str | Path,
    name: str,
    results: Iterable[AnodeResult],
    metadata: Sequence[tuple[str, object]] = (),
) -> tuple[Path, Path]:
    """Write ``<name>.dcc`` and ``depth_summary.csv`` of the (merged) results atomically."""
    rows = list(results)
    dcc_path, csv_path = output_paths(output_dir, name)
    atomic_write_texts(
        [
            (dcc_path, format_dcc(dcc_entries(rows)), "ascii"),
            (csv_path, format_summary_csv(rows, metadata), "utf-8"),
        ]
    )
    return dcc_path, csv_path


def prepare_output_dir(output_dir: str | Path, outputs: Iterable[str | Path]) -> list[Path]:
    """Create the output directory and check that the outputs can be written there.

    Args:
        output_dir: Output directory (created with its parents if missing).
        outputs: The output file names (relative to ``output_dir``).

    Returns:
        The output paths.

    Raises:
        OSError: If the directory cannot be created or written, or an output
            path is a directory (the message says which).
    """
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create the output directory {directory}: {exc}") from exc
    paths = [directory / name for name in outputs]
    for path in paths:
        if path.is_dir():
            raise IsADirectoryError(f"the output file {path} is a directory; remove or rename it")
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".dcalib-write-test-"):
            pass
    except OSError as exc:
        raise OSError(f"cannot write to the output directory {directory}: {exc}") from exc
    return paths
