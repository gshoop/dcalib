"""Tests of the legacy C++ replica (plan 5.6)."""

from __future__ import annotations

import numpy as np
import pytest

from dcalib import legacy
from dcalib.calib import Calibrations
from dcalib.channels import AnodeKey
from dcalib.events import BoardEvents
from dcalib.io.dcc import format_dcc_line
from dcalib.legacy import LEGACY_511, LEGACY_662
from tests.synthetic_cache import tie_pairs

KEY = AnodeKey(9, 16, 0, 10)

# Synthetic calibrations: identity (a doubly converted pair passes the gates)
# and slope 0.5 (it never does). The golden lines below are the output of the
# Linux build of ~/DataProcessing/Dcalib/main.cpp (ROOT 6.26/10) on .chd files
# with exactly these events, recorded 2026-09-24.
KEV = {
    (9, 16, 0, 10): (1.0, 0.0),
    (9, 16, 0, 25): (1.0, 0.0),
    (9, 16, 0, 11): (0.5, 0.0),
    (9, 16, 0, 26): (0.5, 0.0),
    (9, 16, 0, 12): (1.0, 0.0),
    (9, 16, 0, 13): (0.5, 0.0),
    (9, 16, 0, 14): (0.5, 0.0),
}
LUT = Calibrations.from_table(KEV, "test").board_lut(9, 16)


_tie_pairs = tie_pairs


def _run(
    channel: int, cathode_channel: int, pairs: list[tuple[int, int]], eof_quirk: bool = True
) -> legacy.LegacyResult:
    n = len(pairs)
    return legacy.legacy_anode(
        AnodeKey(9, 16, 0, channel),
        [a for a, _ in pairs],
        [0] * n,
        [cathode_channel] * n,
        [c for _, c in pairs],
        LUT,
        eof_quirk=eof_quirk,
    )


class TestGoldenCpp:
    """The C++ binary's output on files that make the EOF quirk visible."""

    def test_identity_calibration_keeps_last_real_pair(self) -> None:
        # The doubly converted pair passes and is the one popped: 4 peaks.
        result = _run(10, 25, _tie_pairs(1.0))
        assert (result.status, result.n_pairs, result.n_peaks) == ("ok", 105, 4)
        assert result.coefficients is not None
        cpp = (508.14, 10.6227, -36.1257)
        np.testing.assert_allclose(result.coefficients, cpp, rtol=1e-4)

    def test_real_calibration_loses_last_real_pair(self) -> None:
        # The last real pair (the tie breaker) is popped: 3 peaks, other curve.
        result = _run(11, 26, _tie_pairs(2.0))
        assert (result.status, result.n_pairs, result.n_peaks) == ("ok", 104, 3)
        assert result.coefficients is not None
        cpp = (507.763, -5.18198, -17.4061)
        np.testing.assert_allclose(result.coefficients, cpp, rtol=1e-3)
        no_quirk = _run(11, 26, _tie_pairs(2.0), eof_quirk=False)
        assert (no_quirk.n_pairs, no_quirk.n_peaks) == (105, 4)

    def test_min_data_is_checked_before_the_pop(self) -> None:
        # C++: 20 accepted pairs pass MIN_DATA (then 19 are fitted); 19 with a real
        # calibration are skipped ("Less than 20 data points"); 19 with an identity
        # calibration pass because the doubly converted pair is accepted too.
        assert _run(13, 26, [(1010, 500)] * 20).status == "fit_failed"  # passes, 1 peak
        assert _run(14, 26, [(1010, 500)] * 19).status == "too_few_events"
        assert _run(12, 25, [(505, 250)] * 19).status == "fit_failed"  # passes, 1 peak
        assert _run(12, 25, [(505, 250)] * 19, eof_quirk=False).status == "too_few_events"


class TestImport:
    def test_gates_are_inclusive(self) -> None:
        lo, hi = LEGACY_511.an_elow, LEGACY_511.an_ehigh
        # Identity calibration: PHA is keV. Pairs at the anode window edges and
        # at C/A = 0 and 1 are kept; just outside they are dropped.
        a = np.array([lo, hi, 500.0, 500.0, np.nextafter(lo, 0), 500.0, 500.0, 500.0])
        c = np.array([100.0, 100.0, 0.0, 500.0, 100.0, -1.0, 500.5, 100.0])
        slope = np.ones(len(a))
        slope[-1] = np.nan  # uncalibrated cathode
        kev_a, kev_c, n = legacy.import_pairs(
            a, c, (1.0, 0.0), slope, np.zeros(len(a)), eof_quirk=False
        )
        np.testing.assert_array_equal(kev_a, [lo, hi, 500.0, 500.0])
        np.testing.assert_array_equal(kev_c, [100.0, 100.0, 0.0, 500.0])
        assert n == 4

    def test_constants_match_the_macros(self) -> None:
        assert LEGACY_511.an_elow == 511 * 0.9 and LEGACY_511.an_ehigh == 511 * 1.1
        assert (LEGACY_511.min_anbin, LEGACY_511.max_anbin, LEGACY_511.source) == (300, 600, 0)
        assert LEGACY_662.an_elow == 662 * 0.8 and LEGACY_662.an_ehigh == 662 * 1.2
        assert (LEGACY_662.min_anbin, LEGACY_662.max_anbin, LEGACY_662.source) == (400, 750, 1)
        assert legacy.legacy_constants(662) is LEGACY_662
        with pytest.raises(ValueError):
            legacy.legacy_constants(600)


class TestRootAxis:
    def test_find_bin(self) -> None:
        bins = legacy.root_find_bin([-0.1, 0.0, 0.0239, 0.024, 1.19, 1.2, np.nan], 50, 0.0, 1.2)
        assert bins.tolist() == [0, 1, 1, 2, 50, 51, 51]

    def test_bin_center(self) -> None:
        np.testing.assert_allclose(legacy.root_bin_center([1, 50], 50, 300.0, 600.0), [303, 597])


class TestPeaks:
    def test_first_maximum_and_gate(self) -> None:
        # Column C/A 0.5: two y bins tie at 5 -> the lower energy wins.
        # Column C/A 0.9: 1 count < 25 % of 5 -> dropped; exactly 25 % would stay.
        a = [450.0] * 5 + [500.0] * 5 + [480.0] * 3 + [470.0]
        c = [0.5 * x for x in a[:10]] + [0.1 * 480.0] * 3 + [0.9 * 470.0]
        peaks = legacy.find_peaks(a, c)
        assert peaks.counts.tolist() == [3, 5]
        a4 = [450.0] * 4 + [500.0] * 4 + [480.0] * 3 + [470.0]
        c4 = [0.5 * x for x in a4[:8]] + [0.1 * 480.0] * 3 + [0.9 * 470.0]
        assert legacy.find_peaks(a4, c4).counts.tolist() == [3, 4, 1]
        np.testing.assert_allclose(peaks.ca, [0.108, 0.492])
        np.testing.assert_allclose(peaks.energy, [483.0, 453.0])

    def test_fit_interior_and_bound(self) -> None:
        x = np.array([0.1, 0.4, 0.7, 1.0])
        concave = 500 + 10 * x - 30 * x**2
        np.testing.assert_allclose(legacy.fit_concave_pol2(x, concave), (500, 10, -30), atol=1e-9)
        convex = 500 - 10 * x + 30 * x**2
        p0, p1, p2 = legacy.fit_concave_pol2(x, convex)
        assert p2 == 0.0
        np.testing.assert_allclose((p0, p1), np.polyfit(x, convex, 1)[::-1], atol=1e-9)
        with pytest.raises(ValueError):
            legacy.fit_concave_pol2(x[:2], concave[:2])


class TestBoard:
    def test_board_and_statuses(self) -> None:
        pairs = _tie_pairs(2.0)
        n = len(pairs)
        rows = {
            "source": np.zeros(n + 5, np.int8),
            "anode_rena": np.zeros(n + 5, np.int8),
            "anode_channel": np.array([11] * n + [15] * 5, np.int8),
            "anode_pha": np.array([a for a, _ in pairs] + [1000] * 5, np.int16),
            "cathode_rena": np.zeros(n + 5, np.int8),
            "cathode_channel": np.full(n + 5, 26, np.int8),
            "cathode_pha": np.array([c for _, c in pairs] + [400] * 5, np.int16),
            "cts": np.arange(n + 5, dtype=np.int64),
            "dcts": np.zeros(n + 5, np.int32),
        }
        census = np.zeros((2, 4, 4), np.int64)
        events = BoardEvents(9, 16, 48, census=census, **rows)
        results = legacy.legacy_board(events, LUT)
        assert [r.key.channel for r in results] == [11, 15]
        assert results[0].status == "ok" and results[0].n_events == n
        assert results[1].status == "no_anode_calibration"
        # Cs build: no Cs events on this board.
        assert legacy.legacy_board(events, LUT, LEGACY_662) == []

    def test_output_line_format(self) -> None:
        result = _run(10, 25, _tie_pairs(1.0))
        assert result.coefficients is not None
        line = format_dcc_line(result.key, result.coefficients)
        assert line == "9 16 0 10 508.14 10.6228 -36.1259 \n"
