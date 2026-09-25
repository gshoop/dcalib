"""Checks against the real full-system adc2kev cache (``pytest -m realdata``).

The cache is ``~/adc2kev-test-data/full-system/data_20260911_124617.cache.h5``
(override with ``DCALIB_TEST_CACHE``). It is only ever opened read-only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dcalib import calib, events
from dcalib.options import DepthOptions

pytestmark = pytest.mark.realdata

CACHE = Path(
    os.environ.get(
        "DCALIB_TEST_CACHE",
        Path.home() / "adc2kev-test-data/full-system/data_20260911_124617.cache.h5",
    )
)


@pytest.fixture(scope="module")
def cache_path() -> Path:
    if not CACHE.exists():
        pytest.skip(f"real test cache not found: {CACHE}")
    return CACHE


@pytest.fixture(scope="module")
def calibrations(cache_path: Path) -> calib.Calibrations:
    return calib.load_calibrations(cache_path)


def test_calibrations(calibrations: calib.Calibrations) -> None:
    # 5,492 valid channels in the cache (one more than calibration.kev).
    assert len(calibrations) == 5492


def test_board_count(cache_path: Path) -> None:
    assert len(events.list_boards(cache_path)) == 156


# (node, board): 1A1C share of the clusters per source, and the pooled Ge+Cs
# photopeak events per calibrated anode (min, median) in the default selection
# window (x in [0.75, 1.12], r in [0, 1.3], calibrated cathode). Plan section 11.
EXPECTED = {
    (1, 17): ((0.655, 0.710), (6124, 7788)),
    (5, 22): ((0.584, 0.661), (715, 1137)),
}


@pytest.mark.parametrize("board", sorted(EXPECTED))
def test_one_anode_one_cathode_counts(
    cache_path: Path, calibrations: calib.Calibrations, board: tuple[int, int]
) -> None:
    (share_ge, share_cs), (count_min, count_median) = EXPECTED[board]
    ev = events.build_board_events(cache_path, *board)
    assert ev.one_anode_one_cathode_share(0) == pytest.approx(share_ge, abs=5e-4)
    assert ev.one_anode_one_cathode_share(1) == pytest.approx(share_cs, abs=5e-4)
    assert np.abs(ev.dcts).max() <= ev.cts_window

    lut = calibrations.board_lut(*board)
    energies = calib.event_energies(ev, lut)
    opts = DepthOptions()
    selected = (
        energies.cathode_calibrated
        & (energies.x >= opts.x_lo)
        & (energies.x <= opts.x_hi)
        & (energies.r >= opts.r_lo)
        & (energies.r <= opts.r_hi)
    )
    counts = [
        int(selected[rows].sum())
        for (rena, channel), rows in ev.anode_rows().items()
        if lut.is_calibrated(rena, channel)
    ]
    assert min(counts) == count_min
    assert np.median(counts) == count_median
