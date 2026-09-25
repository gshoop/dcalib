# dcalib

Depth-of-interaction photopeak calibration for the dual-panel cross-strip CZT PET system, run
directly on the adc2kev calibration cache (`*.cache.h5`). An anode's photopeak position depends
slightly on where in the crystal the gamma ray interacted; the cathode-to-anode energy ratio
r = C/A is a depth proxy. `dcalib` builds one-anode/one-cathode events from the cache, fits each
anode's photopeak position as a function of r, and writes a per-anode `.dcc` correction in the
legacy Dcalib layout plus a `depth_summary.csv`. A PyQt6/pyqtgraph GUI (`dcalib-gui`) shows every
anode's depth curve and before/after spectra, re-fits channels with other options and rejects bad
corrections. It is a separate package built on [adc2kev](../adc2kev-python), like `uvcorr`.

> Status: phases 0-6 of [`docs/planning/DEPTH_CALIBRATION_PLAN.md`](docs/planning/DEPTH_CALIBRATION_PLAN.md)
> are implemented: `dcalib process` and `dcalib legacy` work, and the GUI browses the results
> (System Map, Depth tab, Fit Inspector, Fit All); re-fits, review and export arrive in phase 7. The
> algorithm, its deviations from the plan and the full-system census are in
> [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

## Install

Requires Python 3.10 and an adc2kev checkout at `~/adc2kev-python` (not on PyPI).

```bash
python3.10 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel Cython
venv/bin/pip install -e ~/adc2kev-python
venv/bin/pip install -e ".[dev]"
```

or, equivalently, `make venv install-dev` (use `make ADC2KEV=/path/to/adc2kev-python ...` for
another checkout).

## CLI usage

```bash
dcalib process data.cache.h5 --output-dir out/ [--kev calibration.kev] [--results PATH] \
    [--workers N] [--sources both|ge|cs] [--min-pairs N] [--max-degree {0,1,2}] \
    [--min-gain F] [--concave-only] [--discard-overrides] [--discard-review]
dcalib legacy  data.cache.h5 --output-dir out/ [--kev calibration.kev] [--energy {511,662}] \
    [--no-eof-quirk]
```

`process` writes `out/<name>.dcc` (legacy layout, one line per corrected anode) and
`out/depth_summary.csv`, and stores the results in the sidecar `<name>.depth.h5` next to the cache
(or `--results PATH`). The full system (97.5M 1A1C events) takes about 30 s with 8 workers.
`legacy` reproduces the C++ Dcalib for cross-checks (`<name>_legacy.dcc`, `legacy_summary.csv`);
`scripts/dump_chd.py` and `scripts/compare_dcc.py` run the comparison with the ROOT binary.
Full documentation arrives in phase 8.

## GUI usage

```bash
dcalib-gui data.cache.h5 [--results PATH] [--kev calibration.kev] [--workers N]
```

Opens the cache and the results stored in its sidecar. Click an anode on the System Map (or use
the selectors, Prev/Next or Ctrl+Left/Right) to see its photopeak against C/A before and after the
correction in the Depth tab, and every result field in the Fit Inspector. *Process > Fit All*
(Ctrl+F) runs the whole analysis with the control band's options and stores it.

## Development

The Makefile uses `./venv` automatically when it exists.

| Target | Runs |
|--------|------|
| `make format` / `make format-check` | `black` (`--check`) |
| `make lint` | `ruff check` |
| `make type-check` | `mypy src/` |
| `make test` | `pytest -m "not realdata"` |
| `make test-realdata` | `pytest -m realdata` (needs the full-system test cache) |
| `make dev-check` | format, lint, type-check, test |
| `make check` | format-check, lint, type-check, test (no file changes) |

GUI tests use pytest-qt and run on Qt's `offscreen` platform by default; set `QT_QPA_PLATFORM`
to override.
