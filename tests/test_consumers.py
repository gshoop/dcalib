"""The ``.dcc`` read back by the real time-calibration loader (plan 10.4, integration).

``timecalibration.calibration`` is imported from the installed package or, if
absent, from ``~/time-calibration/src`` (override with ``DCALIB_TIMECAL_SRC``);
the test is skipped when neither works.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from dcalib import cli
from dcalib.io.summary_csv import read_summary_csv


def _timecalibration() -> ModuleType:
    try:
        return importlib.import_module("timecalibration.calibration")
    except ImportError:
        pass
    src = Path(os.environ.get("DCALIB_TIMECAL_SRC", Path.home() / "time-calibration/src"))
    if not src.is_dir():
        pytest.skip("timecalibration is not importable")
    sys.path.insert(0, str(src))
    try:
        return importlib.import_module("timecalibration.calibration")
    except ImportError as exc:
        pytest.skip(f"timecalibration is not importable: {exc}")
    finally:
        sys.path.remove(str(src))


@pytest.mark.integration
def test_dcc_round_trip_with_time_calibration(depth_cache: Path, tmp_path: Path) -> None:
    timecal = _timecalibration()
    out = tmp_path / "out"
    assert cli.main(["process", str(depth_cache), "--output-dir", str(out), "--workers", "1"]) == 0
    loaded = timecal.load_dcc_calibration(str(out / "data_20260911_124617.dcc"))
    rows = {r.key: r for r in read_summary_csv(out / "depth_summary.csv") if r.exported}
    assert set(loaded) == set(rows) and rows
    for key, params in loaded.items():
        row = rows[key]
        assert row.p0 is not None
        # The consumer formula A * 511 / f(C/A) is the dcalib correction A / g(r).
        for r in (0.1, 0.5, 0.9):
            anode = 480.0
            corrected = timecal.apply_dcc_correction(anode, r * anode, **params)
            expected = anode / float(row.g(np.array([r]))[0])
            assert corrected == pytest.approx(expected, rel=2e-5)
