"""Tests of the ``.dcc`` format, the atomic writer and the output helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dcalib.channels import AnodeKey
from dcalib.io import cache_stem
from dcalib.io import dcc as dcc_io
from dcalib.io.export import prepare_output_dir


class TestDccFormat:
    def test_line_layout(self) -> None:
        line = dcc_io.format_dcc_line(AnodeKey(8, 28, 0, 6), (550.724, -50.8708, -8.2751))
        assert line == "8 28 0 6 550.724 -50.8708 -8.2751 \n"

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (0.0, "0"),
            (9.99e-5, "0"),
            (-9.99e-5, "0"),
            (1e-4, "0.0001"),
            (-0.00012345678, "-0.000123457"),
            (505.12345, "505.123"),
            (1234567.0, "1.23457e+06"),
            (511.0, "511"),
        ],
    )
    def test_coefficient_formatting(self, value: float, text: str) -> None:
        assert dcc_io.format_dcc_coefficient(value) == text

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(dcc_io.DccFormatError):
            dcc_io.format_dcc_coefficient(float("nan"))
        with pytest.raises(dcc_io.DccFormatError):
            dcc_io.format_dcc_line(AnodeKey(1, 15, 0, 9), (1.0, 2.0))  # type: ignore[arg-type]

    def test_round_trip_sorted(self, tmp_path: Path) -> None:
        entries = {
            AnodeKey(2, 15, 1, 12): (500.0, 3.0, -2.5),
            AnodeKey(1, 30, 0, 9): (511.0, 0.0, 0.0),
        }
        path = dcc_io.write_dcc(tmp_path / "x.dcc", entries)
        text = path.read_text()
        assert text.splitlines()[0].startswith("1 30 0 9 ")
        assert "#" not in text
        assert dcc_io.read_dcc(path) == entries
        assert dcc_io.write_dcc(tmp_path / "empty.dcc", {}).read_text() == ""

    def test_reader_rules(self, tmp_path: Path) -> None:
        path = tmp_path / "x.dcc"
        path.write_text("# header\n1 15 0 9 500 1\n\n1 15 0 9 500 1 -2 \n1 16 0 10 a b c\n")
        with pytest.raises(dcc_io.DccFormatError, match=r"x\.dcc:5: cannot parse"):
            dcc_io.read_dcc(path)
        path.write_text("# header\n1 15 0 9 500 1\n1 15 0 9 500 1 -2 \n")
        assert dcc_io.read_dcc(path) == {AnodeKey(1, 15, 0, 9): (500.0, 1.0, -2.0)}


def test_cache_stem() -> None:
    assert cache_stem("/x/data_20260911_124617.cache.h5") == "data_20260911_124617"
    assert cache_stem("a.diag.h5") == "a"
    assert cache_stem("a.h5") == "a"
    assert cache_stem("a.hdf5") == "a"
    assert cache_stem(".cache.h5") == ".cache"


def test_prepare_output_dir(tmp_path: Path) -> None:
    paths = prepare_output_dir(tmp_path / "a" / "b", ["x.dcc", "y.csv"])
    assert [p.name for p in paths] == ["x.dcc", "y.csv"] and paths[0].parent.is_dir()
    (tmp_path / "a" / "b" / "x.dcc").mkdir()
    with pytest.raises(IsADirectoryError):
        prepare_output_dir(tmp_path / "a" / "b", ["x.dcc"])
    (tmp_path / "file").write_text("")
    with pytest.raises(OSError, match="cannot create"):
        prepare_output_dir(tmp_path / "file" / "sub", ["x"])
