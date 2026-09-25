# dcalib

Depth-of-interaction photopeak calibration for the dual-panel cross-strip CZT PET system, run
directly on the adc2kev calibration cache (`*.cache.h5`). An anode's photopeak position depends
slightly on where in the crystal the gamma ray interacted; the cathode-to-anode energy ratio
r = C/A is a depth proxy. `dcalib` builds one-anode/one-cathode events from the cache, fits each
anode's photopeak position as a function of r, and writes a per-anode `.dcc` correction in the
legacy Dcalib layout plus a `depth_summary.csv`. A PyQt6/pyqtgraph GUI (`dcalib-gui`) shows every
anode's depth curve and before/after spectra, re-fits channels with other options and rejects bad
corrections. It is a separate package built on [adc2kev](../adc2kev-python), like `uvcorr`.

> Status: all phases (0-8) of
> [`docs/planning/DEPTH_CALIBRATION_PLAN.md`](docs/planning/DEPTH_CALIBRATION_PLAN.md) are
> implemented. User documentation: this README (CLI, files, outputs) and
> [`docs/GUI.md`](docs/GUI.md) (the GUI). The method, its deviations from the plan, the C++
> cross-check, the full-system census and the held-out validation are in
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

## Quick start

```bash
# Depth-calibrate every anode; write out/<name>.dcc and out/depth_summary.csv, and store the
# results in the sidecar data_20260911_124617.depth.h5 next to the cache
dcalib process ~/adc2kev-test-data/full-system/data_20260911_124617.cache.h5 --output-dir out/

# Look at them, re-fit or reject anodes, export again
dcalib-gui ~/adc2kev-test-data/full-system/data_20260911_124617.cache.h5
```

The `.dcc` goes wherever the legacy Dcalib output went: `timecalibration` and the MATLAB time
calibration read it unchanged (section [Outputs](#outputs)).

## How it works (short)

1. **Events.** For every board, each source's anode and cathode hits are clustered in CTS time
   (48-tick window from the first hit) and the clusters with exactly one anode and one cathode
   (1A1C) are kept, 97.5M events on the test cache.
2. **Energies.** Both hits are converted to keV with adc2kev's linear calibrations (the cache's, or
   a `.kev` file), giving x = A/E0 (E0 = 511 keV for Ge-68, 662 keV for Cs-137) and r = C/A.
   Events on an uncalibrated anode or cathode cannot be used.
3. **Depth curve.** The photopeak events (x in [0.75, 1.12], r in [0, 1.3]) are cut into 4-12
   equal-count r slices; each slice's photopeak position is fitted (Gaussian + linear background,
   Poisson likelihood), and a polynomial g(r) of degree 0, 1 or 2 is chosen by a nested χ² test.
   The events are re-gated on x/g(r) and the fit repeated once.
4. **Gates.** A degree-0 curve means no depth dependence. Otherwise a cross-fitted estimate of the
   resolution gain (`cv_gain`) must reach 1 % with no source losing more than 1 %; the Ge-only and
   Cs-only curves are compared, and the curve is checked for extrapolation and C/A coverage.
5. **Output.** Accepted anodes get `A_corr = A / g(r)`, written as `f = 511·g` in the legacy `.dcc`
   layout; the others are left out and stay uncorrected. The FWHM and FWTM of both photopeaks are
   measured before and after.

On the test cache, 2,668 of 5,269 anodes are corrected, and their median FWHM at 511 keV goes from
6.22 % to 5.90 % (5.21 → 4.87 % at 662 keV); details in `docs/ALGORITHM.md` section 4.

## CLI reference

```
dcalib [--version] [-v | -vv] {process,legacy} ...
```

`-v` logs progress (INFO) to stderr, `-vv` debug output.

### `dcalib process`

```
dcalib process CACHE --output-dir DIR [--kev KEV] [--results PATH] [--workers N]
               [--sources {both,ge,cs}] [--min-pairs N] [--max-degree {0,1,2}] [--min-gain F]
               [--concave-only] [--discard-overrides] [--discard-review]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `CACHE` | | adc2kev calibration cache (`*.cache.h5`); opened read-only |
| `--output-dir DIR` | required | Directory for `<name>.dcc` and `depth_summary.csv` (created if missing) |
| `--kev KEV` | the cache's calibrations | Use this `.kev` file's linear calibrations instead |
| `--results PATH` | `<cache stem>.depth.h5` next to the cache | Sidecar results file ([below](#the-sidecar-results-file)) |
| `--workers N` | min(8, usable CPUs) | Worker processes (one board per task) |
| `--sources` | `both` | Fit the Ge-68 (511 keV), the Cs-137 (662 keV) or both photopeaks |
| `--min-pairs N` | 500 | Fewer selected photopeak events: `too_few_events` |
| `--max-degree` | 2 | Highest degree of g(r) |
| `--min-gain F` | 0.01 | Minimum `cv_gain` for a correction to be accepted |
| `--concave-only` | off | Constrain p2 ≤ 0, as the legacy fit did |
| `--discard-overrides` | off | Drop the per-anode re-fits stored by the GUI |
| `--discard-review` | off | Drop the Reject decisions stored by the GUI |

The other options (`DepthOptions` in `src/dcalib/options.py`) keep their defaults; the options used
are stored with the results and written into the CSV header.

What `process` does, in order:

1. Checks that the output directory can be written and the cache is a valid adc2kev cache, then
   loads and fingerprints the calibrations (SHA-256 of the calibration table).
2. Reads the sidecar's overrides and review, unless discarded. Overrides of stale results (another
   cache or calibration) are not applied.
3. Analyses every board in a process pool.
4. Writes `<name>.dcc` and `depth_summary.csv` (each written to a temporary file and renamed), with
   the overrides applied and rejected anodes left out of the `.dcc`.
5. Stores the batch as the sidecar's current results. Overrides are kept, except those the new
   batch reproduces (same options that matter for the fit), and all of them when the old results
   were stale. Rejections are always kept unless `--discard-review`.
6. Prints the census and the timings:

```
Results: /home/you/data/data_20260911_124617.depth.h5
Analysis: 5,269 anodes (97,473,424 1A1C events) on 156 boards, 8 worker(s), options: defaults
  status:     ok 2,668, no_gain 672, no_depth_dependence 1,149, too_few_events 196, fit_failed 9, no_calibrated_cathode 84, no_anode_calibration 491
  flags:      slice_fit_failed 50, convex_curve 12, source_inconsistent 437, extrapolation_risk 19, narrow_ca_coverage 198, partial_cathode_coverage 1,045
  FWHM 511 keV of ok anodes (median, %): 6.22 -> 5.90 (2,665 anodes)
  FWTM 511 keV of ok anodes (median, %): 11.48 -> 11.05 (2,587 anodes)
  FWHM 662 keV of ok anodes (median, %): 5.21 -> 4.87 (2,667 anodes)
  FWTM 662 keV of ok anodes (median, %): 10.01 -> 9.54 (2,635 anodes)
  cv_gain of ok anodes: median 4.27 %
  photopeak position 511 keV (median, E/E0): corrected 0.9984, omitted 0.9976
  photopeak position 662 keV (median, E/E0): corrected 0.9990, omitted 0.9980
  overrides:  0 applied
  rejected:   0
Outputs:
  out/data_20260911_124617.dcc  (2,668 lines)
  out/depth_summary.csv  (5,269 rows)
Time: setup 1.1 s, analysis 25.4 s, write 0.1 s, store 0.1 s, total 26.7 s
```

### `dcalib legacy`

```
dcalib legacy CACHE --output-dir DIR [--kev KEV] [--energy {511,662}] [--no-eof-quirk]
```

A replica of the legacy C++ Dcalib (`~/DataProcessing/Dcalib/main.cpp`) on the same 1A1C events,
for cross-checks: `--energy 511` (default) uses the Ge-68 events and constants, `662` the Cs-137
ones. It reproduces the C++ binning, gates and concave-bounded fit, and by default its end-of-file
quirk (`--no-eof-quirk` turns that off). It writes `<name>_legacy.dcc` and `legacy_summary.csv`
(per anode: status, events, gated pairs, peaks found and the coefficients) and never touches the
sidecar.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | An error (the message says which), e.g. an unreadable cache or an unwritable output |
| 2 | Bad arguments or a missing input file |
| 130 | Stopped with Ctrl-C (nothing is written or stored) |

## Inputs

- **The adc2kev calibration cache** (`*.cache.h5`, adc2kev ≥ 2.2.7): dcalib reads the per-board
  coincidence hits (`/coincidences`) and the calibrations through adc2kev, and never opens the cache
  for writing.
- **Calibrations**: the cache's linear keV calibrations by default, or `--kev FILE`. Anodes whose
  calibration is missing get `no_anode_calibration`; events whose cathode is uncalibrated are
  dropped (`n_uncal_cathode`), and an anode left with none gets `no_calibrated_cathode`. On the
  test cache 692 of 1,248 cathodes have a keV calibration; the GUI's cathode view shows which.

## The sidecar results file

`dcalib process` and the GUI store their results in `<cache stem>.depth.h5`, next to the cache
(`data_20260911_124617.cache.h5` → `data_20260911_124617.depth.h5`), or at `--results PATH`. If the
cache's directory cannot be written, the error suggests `--results`.

| Group | Content |
|-------|---------|
| `/metadata` | dcalib version, results format version, the cache identity (path, size, mtime, adc2kev `created_at`, source hashes) and the calibration source and fingerprint |
| `/results/current` | One row per anode (the CSV columns), the options and `created_at`; its `slices` table holds every slice point (pooled, Ge-only, Cs-only), so the GUI draws curves without refitting |
| `/results/current/overrides` | Per-anode re-fits from the GUI with their own options and slices |
| `/review` | Rejected anodes: state, time stamp, note |

- **Stale results.** When the cache identity or the calibration fingerprint no longer matches, the
  results are stale: the GUI shows them with a banner and disables re-fits; the next Fit All or
  `process` replaces them and drops the overrides. Rejections are kept, since they are decisions
  about a channel.
- **Writes** build the new table completely and then swap it in; no file handle stays open. If
  another process holds the HDF5 lock, dcalib retries for 1 s and then reports the file as in use.

## Outputs

### `<name>.dcc`

`<name>` is the cache stem, e.g. `data_20260911_124617.dcc`. One line per corrected anode (status
`ok` and not rejected), sorted by (node, board, rena, channel), in the legacy Dcalib layout:

```
1 16 0 10 492.044 64.0274 -64.9368 
1 16 1 17 516.507 -14.568 0 
```

`node board rena channel p0 p1 p2`, space-separated with a trailing space and no header, `%g` with
6 significant digits, coefficients with |p| < 1e-4 written as `0` (a degree-1 curve has p2 = 0).
The curve is `f(r) = p0 + p1 r + p2 r² = 511·g(r)`, and consumers correct `A_corr = A·511/f(r)`
with r = C/A in keV; for a 662 keV event this is the same `A/g(r)`. An anode without a line is
left uncorrected. Checked against `timecalibration`'s reader in `tests/test_consumers.py`.

### `depth_summary.csv`

`#` metadata lines first (generation time, cache, calibration source and fingerprint, options as
JSON, results file, dcalib version), then one row per anode with at least one 1A1C event, sorted by
key:

| Columns | Meaning |
|---------|---------|
| `node`, `board`, `rena`, `channel`, `electrode` | Address; electrode label `A01`-`A39` |
| `status` | See [Statuses and flags](#statuses-and-flags) |
| `flags` | `;`-separated warning flags |
| `review` | `rejected` or empty |
| `options_source` | `batch`, or `override` for a GUI re-fit |
| `n_events`, `n_ge`, `n_cs` | 1A1C events, and per source |
| `n_uncal_cathode` | Events dropped because their cathode has no keV calibration |
| `n_selected` | Events in the final photopeak selection |
| `ca_p01`, `ca_p50`, `ca_p99` | 1st, 50th and 99th percentile of r = C/A |
| `degree`, `p0`, `p1`, `p2` | The curve, in `.dcc` units (511·g) |
| `err_p0`, `err_p1`, `err_p2` | Their standard errors |
| `chi2ndf`, `n_slices` | Reduced χ² of the curve fit, number of slices |
| `peak_spread` | Max − min of the slice photopeak positions (fraction of E0) |
| `ge_cs_max_diff` | Largest difference of the Ge-only and Cs-only curves over the 10-90 % r range |
| `cv_gain` | Cross-fitted relative resolution gain (the acceptance criterion) |
| `fwhm_511_before`, `fwhm_511_after`, `fwhm_662_before`, `fwhm_662_after` | FWHM of the net photopeak, % of its position |
| `fwtm_511_before`, `fwtm_511_after`, `fwtm_662_before`, `fwtm_662_after` | The same at a tenth of the height |
| `peak_511`, `peak_662` | Photopeak position (E/E0) of the exported spectrum |

"After" is the corrected spectrum for an `ok` anode and the raw one otherwise. The widths are
measured on a smoothed spectrum with the continuum below the peak subtracted (empty where the peak
does not stand out of the continuum); `docs/ALGORITHM.md` section 4.1 explains why.

### Statuses and flags

| Status | Meaning | In the `.dcc` |
|--------|---------|---------------|
| `ok` | Corrected | Yes, unless rejected |
| `no_gain` | A depth curve, but `cv_gain` below `--min-gain` or a source losing more than 1 % | No |
| `no_depth_dependence` | The flat curve (degree 0) was chosen | No |
| `too_few_events` | Fewer than `--min-pairs` selected events | No |
| `fit_failed` | Fewer than 3 slice fits succeeded | No |
| `no_calibrated_cathode` | Every event's cathode lacks a keV calibration | No |
| `no_anode_calibration` | The anode has no keV calibration | No |

| Flag | Meaning |
|------|---------|
| `slice_fit_failed` | At least one slice fit failed and was dropped |
| `convex_curve` | Quadratic and convex (the legacy fit forbade it) |
| `source_inconsistent` | The Ge-only and Cs-only curves differ by more than max(1 %, 3σ) |
| `extrapolation_risk` | g(r) leaves [0.85, 1.10] somewhere on r ∈ [0, 1.3], where consumers evaluate it unclamped |
| `narrow_ca_coverage` | The 1-99 % r range is narrower than [0.25, 0.95] |
| `partial_cathode_coverage` | More than 50 % of the events had an uncalibrated cathode |

Flags never change the status: a flagged `ok` anode is in the `.dcc`. The GUI shows an `ok` anode
as *flagged* when it has `convex_curve`, `source_inconsistent` or `extrapolation_risk`, so that it
can be reviewed and rejected; the other three flags are informational.

## GUI

```bash
dcalib-gui [CACHE] [--results PATH] [--kev KEV] [--workers N] [-v | -vv]
```

![Main window](docs/images/main_window.png)

The System Map (every anode of both panels, coloured by status or a metric; cathodes by keV
calibration) selects the anode; the Depth tab shows its photopeak against C/A before and after the
correction, the Spectra tab its 511 and 662 keV spectra, the Board grid its board's curves side by
side and the Fleet summary the whole detector; the Fit Inspector lists every result field. *Fit
Channel* / *Fit Board* store per-anode overrides with other options, *Reject* keeps an anode out of
the `.dcc`, *File > Export* writes the `.dcc` and CSV. Everything is stored in the sidecar at once.
See [`docs/GUI.md`](docs/GUI.md).

## Performance (test cache, 8 cores)

| Step | Time |
|------|------|
| `dcalib process`, full system (5,269 anodes, 97.5M 1A1C events) | 27-31 s with 8 workers, 48 s with 4, 173 s with 1 |
| GUI: open the cache with stored results | 1.6-2.2 s |
| GUI: show an anode of a board not loaded yet / of a loaded board | 0.3-1.0 s / < 0.1 s |
| GUI: Fit Channel / Fit Board on a loaded board, stored and redrawn | 0.4-0.5 s / 0.7-1.4 s |

A board's events take 20-120 MB in memory; the GUI keeps the last 4 boards loaded.

## Validation

- **Legacy cross-check** (plan 10.3): `scripts/dump_chd.py` writes `.chd` files of chosen boards
  from the cache, the C++ Dcalib (built outside `~/DataProcessing`) runs on them, and
  `scripts/compare_dcc.py` compares its `.dcc` with `dcalib legacy`'s: 100 of 100 anodes agree on
  three boards. See `docs/ALGORITHM.md` section 2.
- **Synthetic study** (plan 10.2): flat anodes are never corrected, 2 % curves are accepted in
  95-99 % of the cases (`tests/test_depth.py`, `docs/ALGORITHM.md` section 3).
- **Held-out validation** (plan 10.5): `scripts/validate_holdout.py` applies the `.dcc` to two
  other runs' events (built from their raw `.dat` files, needs adc2kev's Cython parser: `make
  cython-check`): median held-out gain 3.6 % against a `cv_gain` of 4.3 %. See `docs/ALGORITHM.md`
  section 5.
- **GUI screenshots**: `scripts/gui_screenshots.py CACHE --results RESULTS --out docs/images`
  re-renders the images of `docs/GUI.md` offscreen, on a copy of the results.

## Development

The Makefile uses `./venv` automatically when it exists.

| Target | Runs |
|--------|------|
| `make venv` / `make install-dev` | Create `./venv`; install adc2kev (editable) and dcalib with the dev tools |
| `make format` / `make format-check` | `black` (`--check`) |
| `make lint` | `ruff check` |
| `make type-check` | `mypy src/ scripts/` |
| `make test` | `pytest -m "not realdata"` |
| `make test-fast` | Also skip the slow tests |
| `make test-realdata` | `pytest -m realdata` (needs the full-system test cache) |
| `make test-cov` | Tests with coverage |
| `make dev-check` | format, lint, type-check, test |
| `make check` | format-check, lint, type-check, test (no file changes; the gate of every phase) |

GUI tests use pytest-qt and run on Qt's `offscreen` platform by default; set `QT_QPA_PLATFORM`
to override. The real-data tests expect the cache at
`~/adc2kev-test-data/full-system/data_20260911_124617.cache.h5` and are skipped without it.

### Project layout

```
src/dcalib/
├── options.py        DepthOptions (defaults, JSON round trip), statuses and flags
├── channels.py       anode/cathode channels, electrode labels, strip order
├── calib.py          calibrations from the cache or a .kev, fingerprint, energies, C/A
├── events.py         per-board 1A1C event building (numba)
├── depth.py          slices, photopeak fits, curve choice, gain gate, consistency checks
├── metrics.py        Poisson peak fitter (numba), FWHM/FWTM of the net photopeak
├── analysis.py       one anode / one board / all boards (process pool), AnodeResult
├── results.py        the sidecar results file (batch, overrides, review, validity)
├── legacy.py         replica of the C++ Dcalib
├── io/               .dcc, the summary CSVs, atomic writes, exports
├── cli.py            dcalib process | legacy
└── gui/              dcalib-gui: window, session, threads, tabs, System Map, inspector
scripts/              dump_chd.py, compare_dcc.py, validate_holdout.py, gui_screenshots.py
tests/                pytest (synthetic caches in tests/synthetic_cache.py), tests/gui/ (pytest-qt)
docs/                 ALGORITHM.md, GUI.md, images/, planning/DEPTH_CALIBRATION_PLAN.md
```

## Documentation

- [`docs/GUI.md`](docs/GUI.md): the GUI, window by window.
- [`docs/ALGORITHM.md`](docs/ALGORITHM.md): events, the legacy replica and its cross-check, the
  default fit and why it differs from the plan, the full-system census, the held-out validation.
- [`docs/planning/DEPTH_CALIBRATION_PLAN.md`](docs/planning/DEPTH_CALIBRATION_PLAN.md): the plan,
  its decisions D1-D13 and the measured facts about the test data.
