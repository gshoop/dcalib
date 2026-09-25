"""Pytest configuration and shared fixtures for dcalib tests.

GUI tests (pytest-qt, ``qt_api = "pyqt6"`` in ``pyproject.toml``) run on Qt's
``offscreen`` platform by default so the suite works headless. Export
``QT_QPA_PLATFORM`` (e.g. ``xcb``) before running pytest to watch them on a
real display instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest

from tests import synthetic_depth as sd
from tests.synthetic_cache import AnodeRecipe, depth_board, depth_calibrations, write_cache

# Must happen before the first QApplication is created (pytest-qt creates it
# lazily in the ``qapp``/``qtbot`` fixtures).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Two boards of node 3 with anodes of known behaviour (board 16 is even: its
# cathodes are channels 25-28 of both RENAs; board 15 is odd: RENA 0 4-7 and
# RENA 1 7-10).
DEPTH_BOARDS: dict[tuple[int, int], dict[tuple[int, int], AnodeRecipe]] = {
    (3, 16): {
        (0, 10): AnodeRecipe(sd.steep, 4000, seed=1),  # ok (8 % curve)
        (0, 11): AnodeRecipe(sd.flat, 3000, seed=2),  # no_depth_dependence
        (0, 12): AnodeRecipe(sd.steep, 200, seed=3),  # too_few_events
        (1, 12): AnodeRecipe(sd.steep, 300, seed=4),  # no_anode_calibration
        (1, 13): AnodeRecipe(sd.steep, 300, seed=5, cathode=(1, 26)),  # no_calibrated_cathode
    },
    (3, 15): {
        (0, 9): AnodeRecipe(sd.concave, 5000, seed=6),  # ok (1.7 % curve)
    },
}
UNCALIBRATED = [(3, 16, 1, 12), (3, 16, 1, 26)]


@pytest.fixture(scope="session")
def _depth_cache_master(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("depth")
    boards = {nb: depth_board(*nb, anodes) for nb, anodes in DEPTH_BOARDS.items()}
    cals = depth_calibrations(DEPTH_BOARDS, UNCALIBRATED)
    return write_cache(directory / "data_20260911_124617.cache.h5", boards, cals)


@pytest.fixture
def depth_cache(tmp_path: Path, _depth_cache_master: Path) -> Path:
    """A private copy of a small adc2kev cache with known depth curves (no sidecar)."""
    path = tmp_path / _depth_cache_master.name
    shutil.copy2(_depth_cache_master, path)
    return path


_HOLD_SCRIPT = """
import sys, h5py
f = h5py.File(sys.argv[1], "a")
f.flush()
print("ready", flush=True)
sys.stdin.read()  # hold the file (and its HDF5 lock) until the parent closes stdin
f.close()
"""


@contextmanager
def _hold_h5_open(path: Path) -> Iterator[subprocess.Popen[str]]:
    """Keep ``path`` open for writing in another process (HDF5 locks are per process)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_SCRIPT, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
        yield proc
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait(timeout=30)


@pytest.fixture
def hold_h5_open() -> Callable[[Path], AbstractContextManager[subprocess.Popen[str]]]:
    """Factory: ``with hold_h5_open(path): ...`` keeps ``path`` locked by another process."""
    return _hold_h5_open
