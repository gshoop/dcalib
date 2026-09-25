"""Smoke tests for the phase 0 package skeleton and CLI wiring."""

from __future__ import annotations

import importlib
import re

import pytest

import dcalib
from dcalib import cli
from dcalib.gui import main as gui_main

SKELETON_MODULES = [
    "dcalib.options",
    "dcalib.channels",
    "dcalib.calib",
    "dcalib.events",
    "dcalib.depth",
    "dcalib.metrics",
    "dcalib.legacy",
    "dcalib.analysis",
    "dcalib.results",
    "dcalib.io",
    "dcalib.io.dcc",
    "dcalib.io.summary_csv",
    "dcalib.cli",
    "dcalib.gui",
    "dcalib.gui.main",
]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", dcalib.__version__)


@pytest.mark.parametrize("name", SKELETON_MODULES)
def test_modules_import(name: str) -> None:
    importlib.import_module(name)


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"dcalib {dcalib.__version__}"


def test_cli_requires_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["process", "legacy"])
def test_cli_subcommands_are_registered(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([command, "--help"])
    assert excinfo.value.code == 0
    assert "--output-dir" in capsys.readouterr().out


def test_cli_rejects_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["process", "x.cache.h5", "--output-dir", "out", "--max-degree", "3"])
    assert excinfo.value.code == cli.EXIT_USAGE
    assert "--max-degree" in capsys.readouterr().err


def test_gui_entry_is_wired(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        gui_main.main(["--help"])
    assert excinfo.value.code == 0
    assert "dcalib-gui" in capsys.readouterr().out
