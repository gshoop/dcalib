"""Tests of the sidecar results file (plan 6.2 and 10.4)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

from dcalib import results as res
from dcalib.analysis import AnodeResult, analyze_all
from dcalib.calib import Calibrations, load_calibrations
from dcalib.channels import AnodeKey
from dcalib.options import DepthOptions

Hold = Callable[[Path], AbstractContextManager[subprocess.Popen[str]]]

STEEP = AnodeKey(3, 16, 0, 10)
FLAT = AnodeKey(3, 16, 0, 11)


@pytest.fixture
def batch(depth_cache: Path) -> tuple[Path, Calibrations, list[AnodeResult], list]:  # type: ignore[type-arg]
    cals = load_calibrations(depth_cache)
    out = analyze_all(depth_cache, cals, workers=1)
    return depth_cache, cals, out.results, out.slices


def _save(
    sidecar: res.ResultsFile,
    batch: tuple[Path, Calibrations, list[AnodeResult], list],  # type: ignore[type-arg]
    options: DepthOptions | None = None,
    calibrations: Calibrations | None = None,
) -> int:
    cache, cals, rows, slices = batch
    return sidecar.save_batch(
        rows, slices, options or DepthOptions(), res.cache_identity(cache), calibrations or cals
    )


def test_default_path() -> None:
    assert res.default_results_path("/d/data_1.cache.h5") == Path("/d/data_1.depth.h5")


class TestBatch:
    def test_round_trip(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        assert sidecar.load() is None and sidecar.created_at() is None
        assert _save(sidecar, batch, DepthOptions(min_gain=0.02)) == 0
        stored = sidecar.load()
        assert stored is not None
        cache, cals, rows, slices = batch
        assert list(stored.results) == rows and list(stored.slices) == slices
        assert stored.options == DepthOptions(min_gain=0.02)
        assert stored.calibration_fingerprint == cals.fingerprint
        assert stored.calibration_source == "cache"
        assert stored.identity == res.cache_identity(cache)
        assert stored.created_at == sidecar.created_at()
        assert stored.merged() == rows
        assert stored.validity(res.cache_identity(cache), cals.fingerprint).valid
        with h5py.File(sidecar.path, "r") as h5f:
            assert h5f["metadata"].attrs["depth_results_version"] == res.DEPTH_RESULTS_VERSION
            assert "results/current/slices" in h5f

    def test_staleness(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        cache, cals, _, _ = batch
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        _save(sidecar, batch)
        stored = sidecar.load()
        assert stored is not None
        identity = res.cache_identity(cache)
        # A changed calibration value (one ulp) makes the results stale.
        table = dict(cals.table)
        key = next(iter(table))
        table[key] = (table[key][0] * (1 + 1e-12), table[key][1])
        changed = Calibrations.from_table(table, "cache")
        validity = stored.validity(identity, changed.fingerprint)
        assert validity.stale and "calibration" in validity.reasons[0]
        # Another cache (different created_at) too; a moved cache does not.
        other = replace(identity, created_at="2030-01-01")
        assert stored.validity(other, cals.fingerprint).stale
        moved = replace(identity, path="/elsewhere/x.cache.h5", mtime=0.0)
        assert stored.validity(moved, cals.fingerprint).valid

    def test_rejects_bad_rows(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        cache, cals, rows, slices = batch
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        identity = res.cache_identity(cache)
        with pytest.raises(ValueError, match="Duplicate"):
            sidecar.save_batch(rows + rows[:1], slices, DepthOptions(), identity, cals)
        with pytest.raises(ValueError, match="batch rows"):
            sidecar.save_batch(
                [replace(rows[0], options_source="override")], [], DepthOptions(), identity, cals
            )

    def test_interrupted_swap_is_recovered(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        _save(sidecar, batch)
        with h5py.File(sidecar.path, "a") as h5f:
            group = h5f["results"]
            group.move("current", "_old")  # a crash between the two moves
            group.create_group("_new")  # and a half-written new batch
        stored = sidecar.load()
        assert stored is not None and len(stored.results) == len(batch[2])
        _save(sidecar, batch)
        with h5py.File(sidecar.path, "r") as h5f:
            assert set(h5f["results"]) == {"current"}

    def test_busy(self, batch: tuple, tmp_path: Path, hold_h5_open: Hold) -> None:  # type: ignore[type-arg]
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        _save(sidecar, batch)
        with hold_h5_open(sidecar.path), pytest.raises(res.ResultsBusyError, match="in use"):
            _save(sidecar, batch)


class TestOverridesAndReview:
    def test_overrides(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        _, _, rows, slices = batch
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        with pytest.raises(res.ResultsError, match="no batch"):
            sidecar.replace_overrides([(rows[1], [], DepthOptions())])
        _save(sidecar, batch)
        created = sidecar.created_at()
        steep = next(r for r in rows if r.key == STEEP)
        steep_slices = [s for s in slices if s.key == STEEP]
        override = replace(steep, status="no_gain")
        options = DepthOptions(min_gain=0.5)
        assert sidecar.replace_overrides(
            [(override, steep_slices, options)], expected_created_at=created
        ) == (1, 0)
        stored = sidecar.load()
        assert stored is not None
        entry = stored.overrides[STEEP]
        assert entry.options == options and entry.result.options_source == "override"
        assert list(entry.slices) == steep_slices
        merged = {r.key: r for r in stored.merged()}
        assert merged[STEEP].status == "no_gain" and merged[STEEP].options_source == "override"
        assert stored.slices_by_anode()[STEEP] == steep_slices
        with pytest.raises(res.StaleResultsError):
            sidecar.replace_overrides(delete=[STEEP], expected_created_at="1999")
        assert sidecar.replace_overrides(delete=[STEEP, FLAT]) == (0, 1)
        assert sidecar.replace_overrides() == (0, 0)
        loaded = sidecar.load()
        assert loaded is not None and not loaded.overrides

    def test_overrides_across_batches(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        _, cals, rows, _ = batch
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        _save(sidecar, batch)
        steep = next(r for r in rows if r.key == STEEP)
        flat = next(r for r in rows if r.key == FLAT)
        sidecar.replace_overrides(
            [(steep, [], DepthOptions(min_gain=0.5)), (flat, [], DepthOptions(max_degree=1))]
        )
        # A batch with min_gain=0.5 reproduces the first override: it is dropped.
        assert _save(sidecar, batch, DepthOptions(min_gain=0.5)) == 1
        stored = sidecar.load()
        assert stored is not None and set(stored.overrides) == {FLAT}
        # A batch with another calibration makes every override stale: dropped.
        other = Calibrations.from_table(dict(cals.table), "elsewhere.kev")
        table = dict(cals.table)
        k = next(iter(table))
        table[k] = (table[k][0] + 1.0, table[k][1])
        other = Calibrations.from_table(table, "elsewhere.kev")
        assert _save(sidecar, batch, calibrations=other) == 1
        stored = sidecar.load()
        assert stored is not None and not stored.overrides
        assert sidecar.clear_overrides() == 0

    def test_review(self, batch: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        sidecar = res.ResultsFile(tmp_path / "x.depth.h5")
        sidecar.set_review(STEEP, "rejected", "odd curve")  # before any batch
        assert sidecar.load() is None and sidecar.load_review()[STEEP].note == "odd curve"
        _save(sidecar, batch)
        stored = sidecar.load()
        assert stored is not None and stored.review[STEEP].state == "rejected"
        merged = {r.key: r for r in stored.merged()}
        assert merged[STEEP].review == "rejected" and not merged[STEEP].exported
        _save(sidecar, batch, DepthOptions(max_degree=1))  # review survives a new batch
        assert STEEP in sidecar.load_review()
        sidecar.set_review(STEEP, "")
        assert sidecar.load_review() == {}
        sidecar.set_reviews({STEEP: ("rejected", ""), FLAT: ("rejected", "")})
        assert sidecar.clear_review() == 2 and sidecar.load_review() == {}
        with pytest.raises(ValueError):
            sidecar.set_review(STEEP, "maybe")
