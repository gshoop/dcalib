"""Tests of calibration loading, LUTs, the fingerprint and the energy conversion."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dcalib import calib
from dcalib.channels import AnodeKey
from dcalib.events import build_board_events
from tests.synthetic_cache import BoardData, write_cache, write_kev

CALS = {
    (1, 16, 0, 10): (0.25, -10.0),  # anode
    (1, 16, 1, 12): (0.30, 5.0),  # anode
    (1, 16, 0, 25): (-0.2, 900.0),  # cathode, negative slope
    (2, 15, 0, 9): (0.5, 1.0),
}


class TestFingerprint:
    def test_order_independent_and_deterministic(self) -> None:
        a = calib.calibration_fingerprint(CALS)
        b = calib.calibration_fingerprint(dict(reversed(list(CALS.items()))))
        assert a == b and len(a) == 64

    def test_sensitive_to_one_ulp(self) -> None:
        changed = dict(CALS)
        slope, intercept = changed[(1, 16, 0, 10)]
        changed[(1, 16, 0, 10)] = (math.nextafter(slope, 1.0), intercept)
        assert calib.calibration_fingerprint(changed) != calib.calibration_fingerprint(CALS)

    def test_sensitive_to_channel_set(self) -> None:
        fewer = {k: v for k, v in CALS.items() if k != (2, 15, 0, 9)}
        assert calib.calibration_fingerprint(fewer) != calib.calibration_fingerprint(CALS)


class TestLoad:
    def test_from_cache_drops_failed_and_zero_slope(self, tmp_path: Path) -> None:
        cals = dict(CALS)
        cals[(1, 16, 1, 20)] = (0.0, 3.0)  # the .kev failed-channel marker
        path = write_cache(tmp_path / "x.cache.h5", {}, cals, failed=[(1, 16, 1, 21)])
        loaded = calib.load_calibrations(path)
        assert loaded.source == calib.CALIBRATION_SOURCE_CACHE
        assert loaded.source_sha256 is None
        assert set(loaded.table) == {AnodeKey(*k) for k in CALS}
        assert loaded.table[AnodeKey(1, 16, 0, 25)] == (-0.2, 900.0)
        assert loaded.fingerprint == calib.calibration_fingerprint(CALS)

    def test_kev_override(self, tmp_path: Path) -> None:
        path = write_cache(tmp_path / "x.cache.h5", {}, CALS)
        kev = write_kev(tmp_path / "cal.kev", {(1, 16, 0, 10): (0.125, -2.5)})
        loaded = calib.load_calibrations(path, kev)
        assert loaded.source == str(kev.resolve())
        assert loaded.source_sha256 == calib.file_sha256(kev)
        assert dict(loaded.table) == {AnodeKey(1, 16, 0, 10): (0.125, -2.5)}

    def test_errors(self, tmp_path: Path) -> None:
        path = write_cache(tmp_path / "x.cache.h5", {})
        with pytest.raises(calib.CalibrationError, match="No valid"):
            calib.load_calibrations(path)
        with pytest.raises(calib.CalibrationError, match="Cannot read"):
            calib.load_calibrations(path, tmp_path / "missing.kev")
        bad = tmp_path / "bad.kev"
        bad.write_text("1 16 0 10 0.5\n")
        with pytest.raises(calib.CalibrationError, match="Cannot read"):
            calib.load_calibrations(path, bad)


class TestLUTAndEnergies:
    def test_board_lut(self) -> None:
        cals = calib.Calibrations.from_table(CALS, "cache")
        lut = cals.board_lut(1, 16)
        assert lut.is_calibrated(0, 10) and lut.is_calibrated(0, 25)
        assert not lut.is_calibrated(0, 11)
        assert int(lut.calibrated.sum()) == 3
        energy = lut.energy([0, 1, 0], [10, 12, 11], [1000, 1000, 1000])
        np.testing.assert_allclose(energy[:2], [240.0, 305.0])
        assert np.isnan(energy[2])
        assert set(cals.board_luts()) == {(1, 16), (2, 15)}

    def test_event_energies(self, tmp_path: Path) -> None:
        board = BoardData()
        # Ge event: A = 0.25 * 2084 - 10 = 511, C = -0.2 * 3222.5 + 900 = 255.5
        board.anode.add(0, 10, 2084, 100, 0)
        board.cathode.add(0, 25, 3222, 110, 0)
        # Cs event on an uncalibrated cathode (1, 26)
        board.anode.add(1, 12, 2190, 5000, 1)
        board.cathode.add(1, 26, 1000, 5001, 1)
        path = write_cache(tmp_path / "x.cache.h5", {(1, 16): board}, CALS)
        cals = calib.load_calibrations(path)
        events = build_board_events(path, 1, 16)
        energies = calib.event_energies(events, cals.board_lut(1, 16))
        np.testing.assert_allclose(energies.anode_kev, [511.0, 0.3 * 2190 + 5.0])
        assert energies.cathode_kev[0] == pytest.approx(-0.2 * 3222 + 900.0)
        assert np.isnan(energies.cathode_kev[1])
        np.testing.assert_allclose(energies.x, [1.0, (0.3 * 2190 + 5.0) / 662.0])
        assert energies.r[0] == pytest.approx(energies.cathode_kev[0] / 511.0)
        assert np.isnan(energies.r[1])
        assert energies.anode_calibrated.tolist() == [True, True]
        assert energies.cathode_calibrated.tolist() == [True, False]
        with pytest.raises(ValueError, match="LUT of n2 b15"):
            calib.event_energies(events, cals.board_lut(2, 15))

    def test_source_e0(self) -> None:
        np.testing.assert_array_equal(calib.source_e0(np.array([0, 1, 0])), [511.0, 662.0, 511.0])
