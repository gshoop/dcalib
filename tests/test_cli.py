"""Tests of the ``dcalib`` command line: ``legacy`` and ``process``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dcalib import cli
from dcalib.channels import AnodeKey
from dcalib.io.dcc import read_dcc
from dcalib.io.summary_csv import read_csv_rows, read_summary_csv
from dcalib.options import DepthOptions
from dcalib.results import ResultsFile
from tests.conftest import DEPTH_BOARDS, UNCALIBRATED
from tests.synthetic_cache import (
    BoardData,
    add_pairs,
    depth_calibrations,
    tie_pairs,
    write_cache,
    write_kev,
)

CALS = {
    (9, 16, 0, 10): (1.0, 0.0),  # identity anode
    (9, 16, 0, 11): (0.5, 0.0),
    (9, 16, 0, 25): (1.0, 0.0),
    (9, 16, 0, 26): (0.5, 0.0),
}


@pytest.fixture
def legacy_cache(tmp_path: Path) -> Path:
    """One board: two fittable anodes (Ge), one uncalibrated anode, Cs events on ch10."""
    board = BoardData()
    add_pairs(board, (0, 10), (0, 25), tie_pairs(1.0), t0=1_000)
    add_pairs(board, (0, 11), (0, 26), tie_pairs(2.0), t0=1_000_500)
    add_pairs(board, (0, 12), (0, 25), [(500, 100)] * 30, t0=5_000_000)
    add_pairs(board, (0, 10), (0, 25), [(600, 300)] * 25, t0=100, source=1)
    return write_cache(tmp_path / "data_20260911_124617.cache.h5", {(9, 16): board}, CALS)


class TestLegacy:
    def test_outputs(
        self, legacy_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        assert cli.main(["legacy", str(legacy_cache), "--output-dir", str(out)]) == cli.EXIT_OK
        dcc = out / "data_20260911_124617_legacy.dcc"
        lines = dcc.read_text().splitlines(keepends=True)
        # Golden C++ values (see tests/test_legacy.py) at 6 significant digits.
        assert lines[0] == "9 16 0 10 508.14 10.6228 -36.1259 \n"
        assert lines[1].startswith("9 16 0 11 507.763 -5.180")
        assert len(lines) == 2
        rows = read_csv_rows(out / "legacy_summary.csv")
        assert [(r["channel"], r["status"]) for r in rows] == [
            ("10", "ok"),
            ("11", "ok"),
            ("12", "no_anode_calibration"),
        ]
        assert rows[0]["electrode"].startswith("A") and rows[0]["n_peaks"] == "4"
        assert rows[1]["n_pairs"] == "104" and rows[2]["p0"] == ""
        header = (out / "legacy_summary.csv").read_text().splitlines()[:3]
        assert header[0] == "# DCALIB Legacy Replica Summary"
        assert header[1].startswith("# Generated: ")
        stdout = capsys.readouterr().out
        assert "ok 2" in stdout and "no_anode_calibration 1" in stdout
        assert not list(tmp_path.glob("*.depth.h5"))  # never touches the sidecar

    def test_no_eof_quirk_and_kev(self, legacy_cache: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        kev = write_kev(tmp_path / "cal.kev", {k: v for k, v in CALS.items() if k[3] != 11})
        argv = ["legacy", str(legacy_cache), "--output-dir", str(out)]
        assert cli.main([*argv, "--no-eof-quirk", "--kev", str(kev)]) == cli.EXIT_OK
        entries = read_dcc(out / "data_20260911_124617_legacy.dcc")
        # ch11 is not in the .kev: skipped; ch10 identical with or without the quirk.
        assert [k.channel for k in entries] == [10]
        assert "eof_quirk: off" in (out / "legacy_summary.csv").read_text()

    def test_662_uses_cs_events(self, legacy_cache: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        argv = ["legacy", str(legacy_cache), "--output-dir", str(out), "--energy", "662"]
        assert cli.main(argv) == cli.EXIT_OK
        rows = read_csv_rows(out / "legacy_summary.csv")
        # 25 Cs events at C/A 0.5 and 600 keV: a single peak.
        assert [(r["channel"], r["status"], r["n_events"]) for r in rows] == [
            ("10", "fit_failed", "25")
        ]

    def test_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out = str(tmp_path / "out")
        assert cli.main(["legacy", str(tmp_path / "none.h5"), "--output-dir", out]) == 2
        assert "not found" in capsys.readouterr().err
        bogus = tmp_path / "bogus.cache.h5"
        bogus.write_text("not hdf5")
        assert cli.main(["legacy", str(bogus), "--output-dir", out]) == cli.EXIT_ERROR
        assert "not an HDF5 file" in capsys.readouterr().err
        missing_kev = ["--kev", str(tmp_path / "x.kev")]
        assert cli.main(["legacy", str(bogus), "--output-dir", out, *missing_kev]) == 2
        (tmp_path / "file").write_text("")
        assert cli.main(["legacy", str(bogus), "--output-dir", str(tmp_path / "file")]) == 2


# ---------------------------------------------------------------------------
# dcalib process
# ---------------------------------------------------------------------------

STEEP = AnodeKey(3, 16, 0, 10)
CONCAVE = AnodeKey(3, 15, 0, 9)


def _process(cache: Path, out: Path, *extra: str) -> int:
    return cli.main(["process", str(cache), "--output-dir", str(out), "--workers", "1", *extra])


class TestProcess:
    def test_outputs_and_sidecar(
        self, depth_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        assert _process(depth_cache, out) == cli.EXIT_OK
        dcc = read_dcc(out / "data_20260911_124617.dcc")
        assert set(dcc) == {STEEP, CONCAVE}
        rows = read_summary_csv(out / "depth_summary.csv")
        assert len(rows) == 6 and {r.status for r in rows} >= {"ok", "too_few_events"}
        sidecar = ResultsFile(depth_cache.with_name("data_20260911_124617.depth.h5"))
        stored = sidecar.load()
        assert stored is not None and len(stored.results) == 6
        stdout = capsys.readouterr().out
        assert "status:     ok 2," in stdout and "overrides:  0 applied" in stdout
        assert "Time:" in stdout

    def test_results_option(self, depth_cache: Path, tmp_path: Path) -> None:
        results = tmp_path / "elsewhere.depth.h5"
        assert _process(depth_cache, tmp_path / "out", "--results", str(results)) == 0
        assert results.is_file()
        assert not depth_cache.with_name("data_20260911_124617.depth.h5").exists()

    def test_overrides_and_rejections_survive(
        self, depth_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        assert _process(depth_cache, out) == 0
        sidecar = ResultsFile(depth_cache.with_name("data_20260911_124617.depth.h5"))
        stored = sidecar.load()
        assert stored is not None
        steep = next(r for r in stored.results if r.key == STEEP)
        sidecar.replace_overrides(
            [(replace(steep, status="no_gain"), [], DepthOptions(min_gain=0.9))]
        )
        sidecar.set_review(CONCAVE, "rejected", "test")
        capsys.readouterr()
        assert _process(depth_cache, out) == 0
        stdout = capsys.readouterr().out
        assert "1 applied" in stdout and "rejected:   1" in stdout
        assert read_dcc(out / "data_20260911_124617.dcc") == {}
        rows = {r.key: r for r in read_summary_csv(out / "depth_summary.csv")}
        assert rows[STEEP].options_source == "override" and rows[STEEP].status == "no_gain"
        assert rows[CONCAVE].review == "rejected"
        # A batch with the override's options reproduces it: dropped.
        assert _process(depth_cache, out, "--min-gain", "0.9") == 0
        assert "reproduced by the batch options" in capsys.readouterr().out
        stored = sidecar.load()
        assert stored is not None and not stored.overrides

    def test_discard_flags(
        self, depth_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        assert _process(depth_cache, out) == 0
        sidecar = ResultsFile(depth_cache.with_name("data_20260911_124617.depth.h5"))
        stored = sidecar.load()
        assert stored is not None
        steep = next(r for r in stored.results if r.key == STEEP)
        sidecar.replace_overrides([(steep, [], DepthOptions(max_degree=1))])
        sidecar.set_review(CONCAVE, "rejected")
        assert _process(depth_cache, out, "--discard-overrides", "--discard-review") == 0
        stdout = capsys.readouterr().out
        assert "discarded (--discard-overrides)" in stdout
        assert "discarded (--discard-review)" in stdout
        assert set(read_dcc(out / "data_20260911_124617.dcc")) == {STEEP, CONCAVE}
        stored = sidecar.load()
        assert stored is not None and not stored.overrides and not stored.review

    def test_stale_results_drop_overrides_keep_review(
        self, depth_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        assert _process(depth_cache, out) == 0
        sidecar = ResultsFile(depth_cache.with_name("data_20260911_124617.depth.h5"))
        stored = sidecar.load()
        assert stored is not None
        steep = next(r for r in stored.results if r.key == STEEP)
        sidecar.replace_overrides([(steep, [], DepthOptions(max_degree=1))])
        sidecar.set_review(CONCAVE, "rejected")
        # A .kev with (slightly) different values: another fingerprint.
        kev = write_kev(
            tmp_path / "cal.kev",
            {
                k: (v[0] + 1e-6, v[1])
                for k, v in depth_calibrations(DEPTH_BOARDS, UNCALIBRATED).items()
            },
        )
        capsys.readouterr()
        assert _process(depth_cache, out, "--kev", str(kev)) == 0
        stdout = capsys.readouterr().out
        assert "stale" in stdout and "overrides:  0 applied" in stdout
        assert "rejected:   1" in stdout
        stored = sidecar.load()
        assert stored is not None and not stored.overrides and CONCAVE in stored.review

    def test_unwritable_default_location(
        self, depth_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = depth_cache.parent
        directory.chmod(0o555)
        try:
            assert _process(depth_cache, tmp_path.parent / f"{tmp_path.name}-out") == 1
        finally:
            directory.chmod(0o755)
        assert "--results" in capsys.readouterr().err

    def test_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert _process(tmp_path / "none.cache.h5", tmp_path / "out") == cli.EXIT_USAGE
        with pytest.raises(SystemExit) as excinfo:
            _process(tmp_path / "none.cache.h5", tmp_path / "out", "--min-gain", "nan")
        assert excinfo.value.code == cli.EXIT_USAGE
        bogus = tmp_path / "bogus.cache.h5"
        bogus.write_text("x")
        assert _process(bogus, tmp_path / "out") == cli.EXIT_ERROR
