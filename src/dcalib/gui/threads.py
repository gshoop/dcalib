"""``QThread`` workers of the GUI (plan section 9).

Follows uvcorr's ``gui/threads.py`` (itself after adc2kev 2.2.7): each long
operation runs in its own ``QThread`` whose ``run`` calls a blocking, Qt-free
function of :mod:`dcalib.gui.session` and reports back through signals, which
Qt queues to the GUI thread:

- ``progress``: see each class;
- ``finished(object)``: the operation's result (it shadows ``QThread.finished``;
  the thread is still finishing ``run`` when the slot runs, so call ``wait()``
  before dropping the last reference);
- ``error(str)``: a message for the user;
- ``stopped()``: :meth:`stop` was honoured and nothing was changed.

:class:`AnodeDataThread` is the Depth tab's short-lived loader: it carries a
generation number so the main window can drop results that a newer selection
has superseded.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from dcalib.analysis import AnalysisCancelled, AnalysisError, WorkerCrashedError
from dcalib.gui.session import (
    AnodeData,
    BatchOutcome,
    DepthSession,
    OpenedCache,
    SessionError,
    open_cache,
)
from dcalib.options import DepthOptions
from dcalib.results import ResultsError

logger = logging.getLogger(__name__)

__all__ = [
    "WORKER_CRASH_MESSAGE",
    "AnodeDataThread",
    "FitAllThread",
    "OpenThread",
    "error_text",
]

_USER_ERRORS = (AnalysisError, SessionError, ResultsError, OSError, ValueError, KeyError)

WORKER_CRASH_MESSAGE = (
    "A Fit All worker process stopped abruptly (for example it ran out of memory), so "
    "nothing was stored. Run Fit All again with fewer workers: start dcalib-gui with "
    "--workers 2, or --workers 1 to fit in-process and name the board that fails."
)


def error_text(exc: BaseException) -> str:
    """A user-facing message for an exception raised by a worker."""
    if isinstance(exc, WorkerCrashedError):
        return WORKER_CRASH_MESSAGE
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    text = str(exc)
    if isinstance(exc, _USER_ERRORS):
        return text or type(exc).__name__
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class _WorkerThread(QThread):
    """Base: runs :meth:`_work` and emits ``finished``, ``error`` or ``stopped``."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stop = threading.Event()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        """Ask the operation to stop at its next check."""
        self._stop.set()

    def run(self) -> None:
        try:
            result = self._work()
        except AnalysisCancelled:
            self.stopped.emit()
            return
        except Exception as exc:  # reported to the user, never raised in Qt's thread
            if not isinstance(exc, _USER_ERRORS):
                logger.exception("%s failed", type(self).__name__)
            self.error.emit(error_text(exc))
            return
        if self._stop.is_set():
            self.stopped.emit()
        else:
            self.finished.emit(result)

    def _work(self) -> Any:
        raise NotImplementedError


class OpenThread(_WorkerThread):
    """Opens a cache (:func:`~dcalib.gui.session.open_cache`); ``finished(OpenedCache)``."""

    def __init__(
        self,
        cache_path: str | Path,
        results_path: str | Path | None = None,
        kev_path: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._args = (cache_path, results_path, kev_path)

    def _work(self) -> OpenedCache:
        return open_cache(*self._args)


class FitAllThread(_WorkerThread):
    """Runs :meth:`DepthSession.run_batch`; ``progress(done, total)`` in hits, then
    ``finished(BatchOutcome)``."""

    progress = pyqtSignal(object, object)

    def __init__(
        self,
        session: DepthSession,
        options: DepthOptions,
        workers: int | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._options = options
        self._workers = workers

    def _work(self) -> BatchOutcome:
        return self._session.run_batch(
            self._options,
            self._workers,
            progress=lambda done, total: self.progress.emit(done, total),
            stop_flag=self._stop,
        )


class AnodeDataThread(QThread):
    """Loads one anode's :class:`AnodeData` (reading its board when not cached).

    Signals:
        done(int, object): the generation and the :class:`AnodeData`.
        failed(int, str): the generation and a message.
    """

    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        session: DepthSession,
        key: tuple[int, int, int, int],
        generation: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._key = key
        self.generation = generation

    def run(self) -> None:
        try:
            data: AnodeData = self._session.anode_data(self._key)
        except Exception as exc:
            if not isinstance(exc, _USER_ERRORS):
                logger.exception("Loading %s failed", self._key)
            self.failed.emit(self.generation, error_text(exc))
            return
        self.done.emit(self.generation, data)
