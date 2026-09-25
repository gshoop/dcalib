"""Tests of the scripts: the C++ cross-check (``dump_chd.py``, ``compare_dcc.py``),
``validate_holdout.py`` and the argument handling of ``gui_screenshots.py``.

``scripts/`` is not a package; the scripts are loaded from their files. Nothing
here runs the C++ Dcalib binary or parses a raw ``.dat`` file (the held-out
test uses synthetic diagnostic caches that adc2kev considers valid).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import h5py
import pytest

from dcalib import cli
from dcalib.channels import AnodeKey
from dcalib.io.dcc import write_dcc
from dcalib.io.summary_csv import read_csv_rows
from tests.synthetic_cache import BoardData, add_pairs, depth_board, tie_pairs, write_cache

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (registered so dataclasses resolve)."""
    module_name = f"_dcalib_script_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dump_chd = _load("dump_chd")
compare_dcc = _load("compare_dcc")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    board = BoardData()
    add_pairs(board, (0, 10), (0, 25), [(1000, 400), (1001, 401)], t0=1000)
    add_pairs(board, (1, 12), (1, 26), [(900, 300)], t0=9000)
    add_pairs(board, (0, 10), (0, 25), [(700, 200)], t0=500, source=1)
    return write_cache(tmp_path / "x.cache.h5", {(3, 16): board})


class TestDumpChd:
    def test_files_and_format(self, cache: Path, tmp_path: Path) -> None:
        out = tmp_path / "chd"
        argv = ["--cache", str(cache), "--boards", "3:16", "--out", str(out)]
        assert dump_chd.main(argv) == 0
        files = sorted(p.name for p in out.iterdir())
        assert files == ["node3board16rena00channel10.chd", "node3board16rena01channel12.chd"]
        assert (out / files[0]).read_text() == (
            "3 16 0 10 1 1000\n3 16 0 25 0 400\n3 16 0 10 1 1001\n3 16 0 25 0 401\n"
        )
        # The Cs event only with --source cs; existing files need --overwrite.
        assert dump_chd.main([*argv, "--source", "cs"]) == 2
        assert dump_chd.main([*argv, "--source", "cs", "--overwrite"]) == 0
        assert (out / files[0]).read_text() == "3 16 0 10 1 700\n3 16 0 25 0 200\n"
        assert not (out / files[1]).exists()

    def test_errors(self, cache: Path, tmp_path: Path) -> None:
        assert dump_chd.main(["--cache", str(tmp_path / "no.h5"), "--out", str(tmp_path)]) == 2
        dotted = tmp_path / "a.b"
        assert dump_chd.main(["--cache", str(cache), "--out", str(dotted)]) == 2
        inputs = tmp_path / "myinputs"
        assert dump_chd.main(["--cache", str(cache), "--out", str(inputs)]) == 2
        empty = tmp_path / "empty"
        assert dump_chd.main(["--cache", str(cache), "--boards", "4:20", "--out", str(empty)]) == 1
        with pytest.raises(SystemExit):
            dump_chd.main(["--cache", str(cache), "--boards", "4-20", "--out", str(empty)])

    def test_parse_boards(self) -> None:
        assert dump_chd.parse_boards("1:17, 5:22") == [(1, 17), (5, 22)]


class TestCompareDcc:
    def test_agreement_rules(self) -> None:
        a = {AnodeKey(1, 17, 0, 9): (500.0, 10.0, -20.0), AnodeKey(1, 17, 0, 10): (505.0, 3.0, 0)}
        b = {
            AnodeKey(1, 17, 0, 9): (500.1, 10.001, -20.002),
            AnodeKey(1, 17, 0, 10): (505.0, 3.0, 0.0009),
            AnodeKey(2, 20, 0, 9): (1.0, 2.0, 3.0),  # other board: ignored
        }
        result = compare_dcc.compare(a, b)
        assert result.ok and result.n_agree == 2 and not result.only_dcalib
        b[AnodeKey(1, 17, 0, 9)] = (501.0, 10.0, -20.0)
        assert not compare_dcc.compare(a, b).ok
        b[AnodeKey(1, 17, 0, 9)] = (500.0, 10.0, -20.0)
        del b[AnodeKey(1, 17, 0, 10)]
        result = compare_dcc.compare(a, b)
        assert not result.ok and result.only_cpp == [AnodeKey(1, 17, 0, 10)]
        result = compare_dcc.compare(a, b, exclude={AnodeKey(1, 17, 0, 10)})
        assert result.ok and result.excluded == [AnodeKey(1, 17, 0, 10)]

    def test_curve_difference(self) -> None:
        assert compare_dcc.curve_difference_pct((500.0, 0.0, 0.0), (500.0, 0.0, 0.0)) == 0.0
        assert compare_dcc.curve_difference_pct((500.0, 0.0, 0.0), (500.0, 5.0, 0.0)) == (
            pytest.approx(1.0)
        )

    def test_end_to_end_with_legacy_output(self, tmp_path: Path) -> None:
        board = BoardData()
        add_pairs(board, (0, 10), (0, 25), tie_pairs(1.0))
        add_pairs(board, (0, 11), (0, 26), [(1000, 400)] * 25, t0=10_000_000)  # 1 peak
        cals = {
            (9, 16, 0, 10): (1.0, 0.0),
            (9, 16, 0, 25): (1.0, 0.0),
            (9, 16, 0, 11): (0.5, 0.0),
            (9, 16, 0, 26): (1.0, 0.0),
        }
        path = write_cache(tmp_path / "d.cache.h5", {(9, 16): board}, cals)
        out = tmp_path / "out"
        assert cli.main(["legacy", str(path), "--output-dir", str(out)]) == 0
        cpp = tmp_path / "cpp.dcc"
        # C++ output: readdir order, and a garbage line for the < 3 peak anode.
        cpp.write_text("9 16 0 11 253.503 515.25 -0.023847 \n9 16 0 10 508.14 10.6227 -36.1257 \n")
        argv = [str(cpp), "--dcalib", str(out / "d_legacy.dcc")]
        assert compare_dcc.main(argv) == 0
        wrong = tmp_path / "wrong.dcc"
        write_dcc(wrong, {AnodeKey(9, 16, 0, 10): (508.14, 10.6227, -30.0)})
        assert compare_dcc.main([str(wrong), "--dcalib", str(out / "d_legacy.dcc")]) == 1
        assert compare_dcc.main([str(tmp_path / "none.dcc"), "--dcalib", str(wrong)]) == 2


# ---------------------------------------------------------------------------
# validate_holdout.py
# ---------------------------------------------------------------------------

validate_holdout = _load("validate_holdout")


def _fake_diag_cache(path: Path, dat: Path, boards: dict) -> Path:  # type: ignore[type-arg]
    """A diagnostic cache of ``boards`` that adc2kev considers valid for ``dat``."""
    from adc2kev.cache.diagnostic_cache import DIAG_CACHE_VERSION
    from adc2kev.cache.hdf5_cache import CalibrationCache

    write_cache(path, boards)
    with h5py.File(path, "a") as h5f:
        attrs = h5f["metadata"].attrs
        attrs["diag_cache_version"] = DIAG_CACHE_VERSION
        attrs["source_size"] = dat.stat().st_size
        attrs["source_mtime"] = dat.stat().st_mtime
        attrs["source_hash"] = CalibrationCache._compute_file_hash(dat)
    return path


def _split_by_source(board: BoardData) -> tuple[BoardData, BoardData]:
    """The Ge and the Cs hits of a board, each relabelled as source 0 (a diagnostic cache)."""
    out = (BoardData(), BoardData())
    for polarity in ("anode", "cathode"):
        hits = getattr(board, polarity)
        for i in range(len(hits)):
            target = getattr(out[hits.source[i]], polarity)
            target.add(hits.rena[i], hits.channel[i], hits.pha[i], hits.cts[i], 0)
    return out


class TestValidateHoldout:
    def test_end_to_end(self, depth_cache: Path, tmp_path: Path) -> None:
        from tests.conftest import DEPTH_BOARDS

        # The work directory must not be inside the cache's or the raw data's directory.
        (tmp_path / "cache").mkdir()
        depth_cache = depth_cache.rename(tmp_path / "cache" / depth_cache.name)
        run = tmp_path / "run"
        assert (
            cli.main(["process", str(depth_cache), "--output-dir", str(run), "--workers", "1"]) == 0
        )
        data = tmp_path / "raw"
        data.mkdir()
        ge_dat, cs_dat = data / "ge.dat", data / "cs.dat"
        ge_dat.write_bytes(b"ge" * 100)
        cs_dat.write_bytes(b"cs" * 100)
        work = tmp_path / "work"
        work.mkdir()
        # Another day: the same curves, other events.
        held_out = {
            nb: depth_board(*nb, {k: replace(v, seed=v.seed + 100) for k, v in anodes.items()})
            for nb, anodes in DEPTH_BOARDS.items()
        }
        split = {nb: _split_by_source(b) for nb, b in held_out.items()}
        _fake_diag_cache(work / "ge.diag.h5", ge_dat, {nb: s[0] for nb, s in split.items()})
        _fake_diag_cache(work / "cs.diag.h5", cs_dat, {nb: s[1] for nb, s in split.items()})
        argv = [
            "--cache",
            str(depth_cache),
            "--run",
            str(run),
            "--ge",
            str(ge_dat),
            "--cs",
            str(cs_dat),
            "--work-dir",
            str(work),
            "--workers",
            "1",
        ]
        assert validate_holdout.main(argv) == 0
        rows = {
            (int(r["board"]), int(r["rena"]), int(r["channel"])): r
            for r in read_csv_rows(work / "holdout_summary.csv")
        }
        steep = rows[(16, 0, 10)]
        assert steep["exported"] == "1" and float(steep["holdout_gain"]) > 0.1
        assert float(steep["fwhm_511_after"]) < float(steep["fwhm_511_before"])
        flat = rows[(16, 0, 11)]
        assert flat["exported"] == "0" and flat["fwhm_511_after"] == flat["fwhm_511_before"]
        assert int(steep["n_ge"]) > 0 and int(steep["n_cs"]) > 0
        text = (work / "holdout_report.md").read_text()
        assert "# Held-out validation" in text and "FWHM 511 keV" in text
        assert "correlation" not in text or "cv_gain" in text

    def test_work_dir_guard(self, depth_cache: Path, tmp_path: Path) -> None:
        dat = tmp_path / "x.dat"
        dat.write_bytes(b"x")
        argv = ["--cache", str(depth_cache), "--ge", str(dat), "--cs", str(dat)]
        assert validate_holdout.main([*argv, "--work-dir", str(tmp_path / "w")]) == 2
        assert validate_holdout.main([*argv[:2], "--ge", str(tmp_path / "none.dat")]) == 2


class TestGuiScreenshots:
    """Only the argument handling: rendering needs its own QApplication (run by hand)."""

    def test_arguments_and_missing_files(self, tmp_path: Path) -> None:
        shots = _load("gui_screenshots")
        args = shots._parse_args(["c.cache.h5", "--results", "r.depth.h5", "--anode", "1,17,0,9"])
        assert args.anode == (1, 17, 0, 9) and args.override == (8, 28, 1, 7)
        with pytest.raises(SystemExit):
            shots._parse_args(["c.cache.h5", "--results", "r.depth.h5", "--anode", "1,17"])
        missing = tmp_path / "missing.cache.h5"
        assert shots.main([str(missing), "--results", str(missing), "--out", str(tmp_path)]) == 2
