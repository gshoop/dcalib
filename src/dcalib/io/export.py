"""Output-directory checks and paths shared by ``dcalib process`` and ``dcalib legacy``.

:func:`prepare_output_dir` checks the output location *before* an analysis, so
a long run does not end in a permission error.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

__all__ = ["prepare_output_dir"]


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
