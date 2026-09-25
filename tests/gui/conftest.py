"""GUI test package configuration (adapted from uvcorr's ``tests/gui/conftest.py``).

- Deterministic destruction of Qt objects: widget trees holding pyqtgraph
  plots end each test inside a reference cycle; with the cyclic collector
  running at allocation-driven moments, Qt could walk a half-deleted tree in a
  later test. Automatic collection is off for the package and each test's
  garbage is collected once its widgets are closed and their deferred
  deletions flushed.
- ``QSettings`` (Ini, user and system scope) points at a per-test directory, so
  a window that persists its layout never touches the real ``~/.config``.
- ``make_window``: shown main windows (``workers=1``) with recorded dialogs;
  ``results_cache``: a private copy of the synthetic depth cache whose sidecar
  holds batch results.
"""

from __future__ import annotations

import gc
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QSettings
from pytestqt.qtbot import QtBot

from dcalib import cli
from dcalib.gui.window import MainWindow

WAIT_MS = 30_000


@pytest.fixture(scope="package", autouse=True)
def _no_automatic_gc() -> Iterator[None]:
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


@pytest.fixture(autouse=True)
def _collect_qt_garbage_at_teardown() -> Iterator[None]:
    yield
    app = QCoreApplication.instance()
    if app is not None:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
    gc.collect(0)


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "cfg"))
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.SystemScope, str(tmp_path / "cfg-system")
    )


class Dialogs:
    """Records the window's error dialogs."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, str]] = []

    def error(self, title: str, text: str) -> None:
        self.errors.append((title, text))

    def install(self, window: MainWindow) -> None:
        window._show_error = self.error  # type: ignore[method-assign]


MakeWindow = Callable[..., tuple[MainWindow, Dialogs]]


@pytest.fixture
def make_window(qtbot: QtBot) -> Iterator[MakeWindow]:
    windows: list[MainWindow] = []

    def factory(**kwargs: object) -> tuple[MainWindow, Dialogs]:
        window = MainWindow(workers=1, **kwargs)  # type: ignore[arg-type]
        qtbot.addWidget(window)
        dialogs = Dialogs()
        dialogs.install(window)
        window.show()
        windows.append(window)
        return window, dialogs

    yield factory
    for window in windows:
        window.close()


def wait_open(qtbot: QtBot, window: MainWindow) -> None:
    qtbot.waitUntil(lambda: window.session.is_open and not window.is_busy, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: not window.data_pending, timeout=WAIT_MS)


@pytest.fixture(scope="session")
def _results_master(tmp_path_factory: pytest.TempPathFactory, _depth_cache_master: Path) -> Path:
    directory = tmp_path_factory.mktemp("gui-results")
    cache = directory / _depth_cache_master.name
    shutil.copy2(_depth_cache_master, cache)
    out = directory / "out"
    assert cli.main(["process", str(cache), "--output-dir", str(out), "--workers", "1"]) == 0
    return cache


@pytest.fixture
def results_cache(tmp_path: Path, _results_master: Path) -> Path:
    """A private copy of the synthetic depth cache with its sidecar holding batch results."""
    directory = tmp_path / "res"
    directory.mkdir()
    for path in _results_master.parent.glob("data_*.h5"):
        shutil.copy2(path, directory / path.name)
    return directory / _results_master.name
