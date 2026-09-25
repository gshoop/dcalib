"""Exporters and file names: the legacy ``.dcc`` file and the summary CSVs."""

from __future__ import annotations

from pathlib import Path

__all__ = ["cache_stem"]

_CACHE_SUFFIXES = (".cache.h5", ".diag.h5", ".h5")


def cache_stem(cache_path: str | Path) -> str:
    """Return the name outputs derive from: the cache file name without its suffix.

    ``data_20260911_124617.cache.h5`` gives ``data_20260911_124617``, from
    which ``<name>.dcc``, ``<name>_legacy.dcc`` and the sidecar
    ``<name>.depth.h5`` are named.
    """
    name = Path(cache_path).name
    for suffix in _CACHE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return Path(name).stem
