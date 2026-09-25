# Depth Calibration from the adc2kev Cache: Plan

**Status:** Plan drafted 2026-09-24 from a brainstorming session; decisions D1-D13 confirmed by the
user. Phases 0-4 are implemented (skeleton; options, channels, calibrations and event building;
the legacy replica, `dcalib legacy` and the C++ cross-check; the default depth fit and metrics;
the analysis driver, sidecar results, exports, `dcalib process` and the full-system census, see
`docs/ALGORITHM.md`).
**Repository:** `/home/swuupii/dcalib` (git, created in phase 0)
**Package name:** `dcalib`. Console scripts: `dcalib` (CLI) and `dcalib-gui`.

This document is self-contained: an implementer with no prior context should be able to carry out
the whole plan from it. Read sections 1-5 first, then work through the phases in section 12.
Section 14 lists every external file referenced.

---

## 1. Problem

The detector is a dual-panel cross-strip CZT PET system read out by RENA-3 ASICs. An anode's
photopeak position depends slightly on the depth of interaction (DOI), mainly through electron
trapping near the cathode. The cathode-to-anode energy ratio **r = C/A** is a DOI proxy: r ≈ 0 near
the anode, r ≈ 1 near the cathode. A depth calibration measures each anode's photopeak position as a
function of r and writes a per-anode correction.

The existing tool is **Dcalib** (C++/ROOT, `~/DataProcessing/Dcalib/main.cpp`, section 4). Its
input is a folder of thousands of per-anode `.chd` text files, which a separate program
(`extractData -d`) writes from an unpacked text dump of the raw data. We want to:

1. Run the depth calibration **directly on the adc2kev calibration cache** that the energy
   calibration already built, using the same events and the same keV calibration.
2. Provide a **GUI** to inspect every anode's depth curve and its before/after spectra, re-fit
   channels with other options, reject bad corrections, and export the results.

The design follows **uvcorr** (`~/uv-ellipse-correction`), the rewrite of the legacy
EllipseCorrection on top of adc2kev. It is a separate package that depends on adc2kev. It has a
`process` CLI and a PyQt6/pyqtgraph GUI with a forked System Map, and it stores results plus GUI
overrides in HDF5. dcalib has the same relationship to adc2kev and follows the same conventions.

**Test data:** `~/adc2kev-test-data/full-system/`. Do not modify or move anything in it (the
sidecar results file of section 6.2 is the one exception, and only when written by `dcalib`):

| File | Content |
|------|---------|
| `data_20260911_124617.cache.h5` | adc2kev 2.2.7 calibration cache (3.57 GB): Ge-68 (file_id 0) and Cs-137 (file_id 1) events of the full system (525.8M events), stored calibrations and fit results |
| `calibration.kev` | The same calibrations exported (identical values; the cache has one extra valid channel) |
| `sources/ge/data_20260910_120628.dat`, `sources/cs/data_20260910_121135.dat` | Raw Ge-68 and Cs-137 acquisitions from the **previous day**, used only for the held-out validation (section 10.5) |

## 2. Decisions (all confirmed by the user)

| # | Decision |
|---|----------|
| D1 | A **separate package `dcalib`** in its own git repo at `~/dcalib` with its own venv. It **depends on adc2kev ≥ 2.2.7** (installed editable from `~/adc2kev-python`) and mirrors uvcorr's layout, tooling and GUI conventions. Reuse adc2kev's loaders, calibration I/O and geometry; don't copy them. |
| D2 | **Input: the adc2kev calibration cache (`*.cache.h5`) only.** It is opened read-only and never written. dcalib uses both sources (file_id 0 = Ge-68/511 keV, 1 = Cs-137/662 keV). Calibrations come from the cache by default; `--kev` overrides them. Raw `.dat` input is not a product feature. Only the held-out validation script reads a raw acquisition, through adc2kev's `DiagnosticCache`. |
| D3 | **Uncalibrated cathodes:** an anode whose events only have cathodes without a keV calibration gets the status `no_calibrated_cathode`. Fixing adc2kev's cathode-calibration coverage (692/1248 cathodes; 24/156 boards have none) is a **separate work item outside this plan**. |
| D4 | **Best method by default, plus a legacy-replica mode.** The replica reproduces the C++ algorithm exactly on the same events and exists to cross-check against the ROOT binary. The default method has no parity requirement. |
| D5 | **Events: exactly 1 anode + 1 cathode per board** within a greedy 48-CTS window anchored on the first hit. These are the `extractData -d` semantics (section 4.1) and the 1A1C events the consumers correct. Multi-anode and multi-cathode clusters are dropped; cathodes are never summed. |
| D6 | **Sources pooled** in normalised units x = A_keV / E0. The Ge and Cs curves are also fitted separately and compared, and a flag is raised when they disagree. The GUI shows the per-source curves. |
| D7 | **An anode whose correction fails, or doesn't improve held-out resolution, is omitted from the `.dcc`.** Every consumer already leaves missing anodes uncorrected. The CSV records the reason. |
| D8 | **Model: the pol2 family, degree 0-2 chosen per anode** by a significance test. It is fitted over the C/A range the data covers. Because consumers evaluate the polynomial at any r, the fit is not clamped (a clamp can't be encoded). The C/A coverage is reported in the CSV. |
| D9 | **`.dcc` in the legacy layout, headerless** (MATLAB `importDcc.m` would break on a header). The coefficients are expressed so that every consumer formula `A·511/(p0+p1 r+p2 r²)` works unchanged (section 5.4). Provenance goes in the CSV and the sidecar. |
| D10 | **GUI review:** a channel, board or everything can be re-fitted with other options (stored as per-anode overrides), and a per-anode **Reject** toggle drops an anode from the `.dcc`. Both are persisted in the sidecar results file, and the exports reflect both. |
| D11 | **GUI views:** System Map and Fit Inspector docks; tabs **Depth** (channel depth view), **Spectra** (before/after), **Board grid** and **Fleet summary** (section 9). |
| D12 | **Results live in a sidecar HDF5** (`<cache stem>.depth.h5` next to the cache by default, `--results PATH` otherwise): batch results, per-slice points, overrides and rejections (section 6.2). |
| D13 | **Extras in scope:** the C++ cross-check scripts (section 10.3) and the held-out validation on the 2026-09-10 data (section 10.5). **Out of scope:** a C/A-to-depth (mm) mapping, fixes to adc2kev's cathode fitter, exporting corrected per-event energies, and batch PDF/PNG figure export. |

Git: initialise the repo in phase 0. Commit messages carry **no** `Co-Authored-By` or
generated-by trailers (standing user rule).

## 3. Environment

| Item | Value |
|------|-------|
| OS | Linux (WSL2), 24 cores, 15 GB RAM |
| Python | 3.10 (`python3.10`) |
| adc2kev | `~/adc2kev-python`, version 2.2.7, installed editable in `~/adc2kev-python/venv`. It pins `numpy>=1.24,<2.0` and depends on scipy, h5py, PyQt6, pyqtgraph and **numba** (used here for the event clustering, section 5.1) |
| ROOT | `~/root-install` (6.26/10; `source ~/root-install/bin/thisroot.sh`). Only needed for the C++ cross-check. |
| Legacy binary | The checked-in `~/DataProcessing/Dcalib/Dcalib` is a **macOS arm64** build and does not run here. Build a Linux binary **outside** the DataProcessing tree (it compiles cleanly, about 1 s): `g++ -std=c++17 -O2 $(root-config --cflags) ~/DataProcessing/Dcalib/main.cpp -o <scratch>/Dcalib_linux $(root-config --libs --glibs)` |

**Venv (phase 0), as for uvcorr:** create `./venv` with `python3.10 -m venv venv`, then run
`pip install --upgrade pip setuptools wheel Cython`, `pip install -e ~/adc2kev-python` and
`pip install -e ".[dev]"`. Fallback: install into `~/adc2kev-python/venv`. The Cython parser isn't
needed (dcalib never parses `.dat`), except by the held-out validation script, which builds adc2kev
`DiagnosticCache`s.

## 4. Reference: the legacy pipeline

### 4.1 `.chd` production (`extractData -d`)

Source: `~/DataProcessing/extractData/data_kimia_edit/main.cpp` (the current version;
`data_original/` is older, and `compare_chd.sh` there documents the differences).

- Input: an unpacked text dump, `node board rena channel polarity PHA U V CTS`, where polarity 1 =
  anode and 0 = cathode. The hits are sorted by CTS.
- Greedy windows of `NUMCTS = 48` ticks (1 µs at 48 MHz), anchored on the first hit, inclusive
  (`<= startCTS + 48`, `main.cpp:24, 318-334`). Hits in a window are grouped by (node, board)
  (`:352-357`).
- A (node, board) group is written only if it has **exactly 1 anode and 1 cathode** (`:377-383`).
  There is no energy threshold and no C/A gate, and cathodes are never summed.
- Values are the **raw PHA** (not pedestal-subtracted, not keV). The per-channel linear `.kev`
  absorbs the pedestal and the sign, and some slopes are negative.
- Format: two lines per event, **anode first**, then the cathode, each `node board rena channel
  polarity PHA`, no header. One file per anode, named `node%dboard%02drena%02dchannel%02d.chd`
  (`channelwriter.cpp:15`), in `{basename}channelData/`. Example (`…/data_20250730_114926channelData/node10board15rena00channel08.chd`):
  ```
  10 15 0 8 1 2640
  10 15 1 10 0 2079
  ```

### 4.2 Dcalib (`~/DataProcessing/Dcalib/main.cpp`, 750 lines)

Constants (511 keV build; a 662 keV set is commented out at lines 46-57): `AN_ELOW/AN_EHIGH =
511·0.9/511·1.1`, `C2A_LOW/HIGH = 0/1`, `AN_BINS = CA_BINS = 50`, `MIN/MAX_ANBIN = 300/600`,
`MIN/MAX_CABIN = 0/1.2`, `MIN_DATA = 20`, `EN_GOAL = 511`, `MIN_MIN2MAX_PEAKRATIO = 0.25`.

Per `.chd` file (one anode):

1. **Import (`importCHD`, lines 67-124).** Read (anode, cathode) line pairs:
   - Anode without `.kev` parameters: **the whole file is skipped**.
   - Convert the anode to keV and keep it only if it is in `[AN_ELOW, AN_EHIGH]` (inclusive).
   - Cathode without parameters: skip the pair. Otherwise convert it to keV and keep the pair only if
     `C/A` is in `[0, 1]` (inclusive).
   - If fewer than `MIN_DATA` pairs are kept, **the file is skipped**. The count is checked
     *before* the next step.
   - **EOF quirk:** the `while(!eof)` loop runs its body once more after the last line pair: every
     `>>` fails and leaves its variable unchanged, and then `pop_back()` removes the last accepted
     pair. Because `aEn` and `cEn` were converted to keV *in place*, that extra pass converts the
     last anode energy a second time (`A' = A·slope + intercept`), and the cathode energy too if the
     last pair got past the anode gate. The doubled conversion almost never passes the gates, so
     in practice the last *accepted real* pair is lost, and a file with exactly 20 accepted pairs
     is fitted with 19 (one with 19 is skipped). Only a near-identity calibration lets the extra
     pair through, and then it is the one removed. (Phase 2 finding, confirmed against the ROOT
     binary on hand-made `.chd` files; the first draft of this plan assumed the re-read pair was
     an exact duplicate.)
2. **Histogram.** `TH2D` with x = C/A (50 bins on [0, 1.2]) and y = anode keV (50 bins on [300, 600],
   6 keV wide). Filled with every kept pair.
3. **Peaks.** For each x bin 1..50, take the y bin with the maximum content. The scan starts at y bin 1
   and uses strict `>`, so the first maximum wins. Columns whose maximum is 0 are skipped. Record
   (x bin centre, y bin centre, content). Then drop the columns whose content is < 0.25 × the largest
   content.
4. **Fit.** `TGraph` (unweighted) fitted with `pol2`, option `"SB"`, `SetParLimits(2, -1e5, 0)`
   (concave only). With fewer than 3 peaks the behaviour is undefined.
5. **Export.** The line `node board rena channel` (parsed from the filename) is followed by
   `p0 p1 p2`, where |p| < 1e-4 is written as `0`. It uses ostream default formatting (6 significant
   digits) and ends with a trailing space. With `-s` or `-d` it also fits Gaussians before and after
   correction and draws a 6-pad canvas (the FWHM is never written to the `.dcc`).

Known defects fixed in the default method: the EOF quirk; `MIN_DATA = 20` is far too low; a
fit with < 3 peaks writes garbage; the output name is derived from an `"inputs"` substring of
the path; no quality metrics are written; failed anodes vanish silently; the ±10 % gate on
**uncorrected** energy and the C/A ≤ 1 gate remove exactly the most depth-affected events
(section 11); and the argmax peak finder's 6 keV bins (1.2 % of E0) are as large as the effect it
measures.

### 4.3 `.dcc` consumers (what the output must stay compatible with)

| Consumer | Reads | Applies |
|----------|-------|---------|
| `~/time-calibration/src/timecalibration/calibration.py:113-192`, and the identical copy `~/tp-processing/rilpetlib/src/rilpetlib/timecalibration.py:33-106` | Skips `#` lines and lines with < 7 fields; looks up by the anode's channel key; a missing anode stays uncorrected | `r = C_keV/A_keV`, `A_corr = A_keV·511/(p0+p1 r+p2 r²)` (A = 0 gives 0; f = 0 leaves A unchanged). `core.py:1056-1080` then recomputes C/A from the corrected anode energy for the `c_a` regressors |
| `~/DataProcessing/TimeCalibration/MATLAB_Code/helper_functions/importDcc.m` | `textscan` with `%d %d %d %d %f %f %f`: **a header line would break it** | Correction commented out; only the raw C/A is used |
| `~/ril_software/TimeCalibration/TimeCalibration(MATLAB)/TimeCalibration_W14.m:522-538` (legacy) | Same format; missing anodes default to -1 (unchecked) | The same formula, then gates C/A to [0, 1.2] and the anode energy to 511 ± 10 % |

No consumer converts C/A to a physical depth. `DAQ_processing/doi_ca_mapping.py` builds
depth(mm) = f(C/A) from a collimator sweep, but only as an analysis.

## 5. Algorithms in `dcalib`

### 5.1 Event building (`dcalib.events`, D5)

Per board and per source (file_id 0 or 1):

1. Read `/coincidences/node_N/board_B/{anode_,cathode_}{triggers,renas,channels,phas,timestamps,file_ids}`
   in **one pass** (a thin h5py reader modelled on `adc2kev.analysis.ca_ratio._read_filtered`). A
   missing polarity group means no events of that polarity. `timestamps` is the CTS.
   `adc2kev.analysis.load_board_event_arrays` would need one call per source because it doesn't
   return `file_ids`. Adding that field upstream is an optional adc2kev change.
2. Concatenate the anode and cathode hits, keep a polarity flag and the index into the source array,
   and **stable-sort by CTS**. The cache stores the hits in file order, which is not monotonic in
   CTS.
3. **Greedy clustering** (a numba loop): a hit starts a new cluster when `cts > anchor_cts + 48`;
   otherwise it joins the current cluster (inclusive, like extractData). The window is
   `DepthOptions.cts_window` (default 48). Unlike extractData, which anchors its windows on the
   time-sorted hits of the whole system and then groups each window by (node, board), the windows
   are anchored per board, since the cache stores every board separately. The two agree unless
   another board's hit anchors a window between a board's anode and cathode hits.
4. Keep the clusters with exactly one anode hit and one cathode hit. The output is a
   `BoardEvents` structure of parallel arrays, one row per event: `source` (int8), anode `rena`,
   `channel`, `pha`, cathode `rena`, `channel`, `pha`, and `dcts`. Energies are **not** stored, so
   the calibration can change without rebuilding.

Measured: ≤ 0.2 s per board after loading, and 58-71 % of the clusters are 1A1C on the tested boards
(section 11). No event cache is persisted. The GUI keeps an in-memory LRU of the last few boards'
`BoardEvents`.

### 5.2 Energies and C/A (`dcalib.calib`)

- The calibration source is the cache (`CalibrationCache.load_all_calibrations()`, 1.2 s) or the
  `--kev` file (`KEVExporter().read`), reduced with `adc2kev.analysis.valid_linear_calibrations`
  (success and slope ≠ 0).
- Per-board LUTs as in `adc2kev.analysis.ca_ratio._build_luts`. `A_keV = slope·pha + intercept`,
  and the same for the cathode.
- Events whose anode is uncalibrated: the anode's status is `no_anode_calibration`.
- Events whose cathode is uncalibrated are dropped and counted per anode (`n_uncal_cathode`). An
  anode left with no events: `no_calibrated_cathode` (D3).
- `x = A_keV / E0` with E0 = 511 (Ge) or 662 (Cs); `r = C_keV / A_keV`, both **uncorrected**, as
  the consumers compute them.
- **Calibration fingerprint:** the SHA-256 of the sorted `(node, board, rena, channel, slope,
  intercept)` table, stored with the results. It detects results made stale by a changed adc2kev
  calibration.

### 5.3 Depth fit, default method (`dcalib.depth`)

All thresholds are `DepthOptions` fields. The values below are **initial defaults, to be retuned by
the phase-4 census**.

1. **Selection.** Events with `x ∈ [x_lo, x_hi] = [0.75, 1.12]` and `r ∈ [r_lo, r_hi] = [0.0, 1.3]`
   from the chosen sources (`sources = both | ge | cs`, default `both`). The x window is asymmetric
   so that the trapping tail is kept. The r range keeps the ~11 % of photopeak events with C/A > 1
   that the legacy gate dropped. Fewer than `min_pairs` (800) selected events: `too_few_events`.
2. **Slices.** Cut into `n_slices = clamp(N // per_slice, 4, 12)` equal-count bins in r
   (`per_slice = 400`). Record each slice's `r_lo`, `r_hi` and median r.
3. **Slice photopeak fit.** Histogram x on [0.80, 1.15] in 0.0025 bins, seed μ at the mode and σ
   from the anode's pooled fit, then fit a Gaussian plus a linear background by binned Poisson
   likelihood (`scipy.optimize.least_squares` on Pearson residuals, like adc2kev's fitters) over
   `[μ − 1.5σ, μ + 2.5σ]`, iterating the window up to 3 times. The window is asymmetric to keep the
   low-energy tail from pulling the centroid. Output: μ ± err, σ, n and status. A failed slice is
   dropped and counted (flag `slice_fit_failed`); fewer than 3 good slices gives `fit_failed`.
   **Phase 3:** the window is `[μ − 2σ, μ + 3σ]` and keeps the pooled σ as its width (it is only
   re-centred), which halves the slice position error; the likelihood is minimised with the
   deviance residuals by a numba Levenberg-Marquardt loop (25× faster than scipy). See
   `docs/ALGORITHM.md` section 3.
4. **Curve.** Weighted least squares of μ_i against the slice median r_i for degrees 0, 1 and 2.
   Choose the lowest degree that a nested Δχ² test doesn't reject at `p_degree = 0.01`
   (`max_degree = 2`). The legacy concave-only bound (p2 ≤ 0) is `concave_only`, default off; a
   significantly convex curve gets the flag `convex_curve`.
5. **Re-gate.** Correct `x_c = x / g(r)`, re-select the events with `x_c ∈ [x_lo, x_hi]`, and repeat
   steps 2-4 once (`gate_iterations = 1`). Legacy gated on the uncorrected energy only.
6. **Degree 0 chosen:** the status is `no_depth_dependence` and the anode is omitted (D7).
7. **Gain gate (D7).** 2-fold cross-validation with a deterministic split (even and odd events of
   the anode in CTS order). Fit on one half, apply it to the other, and compare the Gaussian-core
   FWHM of `x_c` with that of `x` on **the same events** (a paired comparison), per source.
   - `cv_gain` = 1 − FWHM_corr/FWHM_raw, averaged over folds and sources.
   - Accept if `cv_gain ≥ min_gain` (0.01) and no source worsens by more than `max_source_loss`
     (0.005). Otherwise the status is `no_gain`.
   - An accepted anode gets its final coefficients from the fit on **all** its events.
   - **Phase 3:** implemented as cross-fitting at the degree chosen on all events, and the width
     ratio is predicted from the alignment of the slice positions of the raw and the out-of-fold
     corrected energies (`sqrt((σ² + V_corr)/(σ² + V_raw))`), because the Gaussian-core FWHM does
     not see depth broadening and widths on a few hundred events are too noisy to confirm a 2 %
     curve (`docs/ALGORITHM.md` section 3.2).
8. **Source consistency (D6).** Fit the Ge-only and Cs-only curves at the chosen degree (when each
   has ≥ 3 × `per_slice` events). Compare them over the anode's 10-90 % r range; flag
   `source_inconsistent` if max |Δg| > max(0.01, 3σ_Δ). Fleet-level measurement: ≈ 0.4 % mid-range,
   ≈ 1 % at the ends (section 11).
9. **Extrapolation check (D8).** Evaluate g on `[0, 1.3]`. Flag `extrapolation_risk` if it leaves
   `[0.85, 1.10]`, and `narrow_ca_coverage` if the 1-99 % r range of the data is narrower than
   `[0.15, 0.95]`.

**Statuses** (one per anode): `ok` (written to the `.dcc`), `no_gain`, `no_depth_dependence`,
`too_few_events`, `fit_failed`, `no_calibrated_cathode`, `no_anode_calibration`. **Flags** (zero or
more): `slice_fit_failed`, `convex_curve`, `source_inconsistent`, `extrapolation_risk`,
`narrow_ca_coverage`, and `partial_cathode_coverage` (> 50 % of the anode's events had an
uncalibrated cathode). **Review** (D10) is a separate column, `rejected` or empty.

### 5.4 Correction and `.dcc` coefficients

g(r) is the photopeak position in E/E0 units, and the correction is `A_corr = A / g(r)`. The
consumers apply `A·511/f(r)`, so the `.dcc` stores `f = 511·g`, that is `p_k = 511·c_k`. For a
662 keV event, `A·511/(511·g) = A/g`, the same correction. An accepted correction therefore also
re-centres the photopeak at E0, as the legacy `EN_GOAL` normalisation did. Omitted anodes keep the
`.kev` scale (a residual offset of ≲ 0.5 %, open item O3).

### 5.5 Metrics (`dcalib.metrics`)

For every anode and per source, before (x) and after (x_c; x itself for anodes that aren't `ok`):

- `fwhm_*`: the Gaussian-core FWHM in % of μ (fitter of step 3 on the whole anode). **Phase 3:**
  the FWHM of the smoothed spectrum instead, like `fwtm_*` (`docs/ALGORITHM.md` section 3.3).
- `fwtm_*`: the non-parametric full width at 1/10 of the maximum of a lightly smoothed histogram, in
  %. It captures the tail that the depth correction is expected to improve most.
- The slice points and the curve, with the reduced χ² (`chi2ndf`), `peak_spread` (max − min of the
  slice μ) and `ge_cs_max_diff`.

### 5.6 Legacy replica (`dcalib.legacy`, D4)

The replica runs on the same `BoardEvents`, **Ge only** (E0 = 511; the 662 constant set is
optional). Each step of section 4.2 is reproduced exactly:

- the inclusive gates, the file-level skips and the `MIN_DATA` check before the pop;
- the **EOF quirk**, as an option that is on in legacy mode;
- `TH2D` binning semantics (`[low, high)` bins, under- and overflow ignored by the peak scan);
- the first-maximum argmax and the 25 % gate;
- the concave-bounded fit, done as bounded linear least squares (`scipy.optimize.lsq_linear`,
  `p2 ∈ [−1e5, 0]`). At an interior optimum this equals ROOT's `SB` fit, and at the bound it is the
  pol1 fit with p2 = 0;
- the |p| < 1e-4 → 0 rule and the same output formatting.

With fewer than 3 peaks the replica records `fit_failed` and writes no line. The cross-check
excludes those anodes. The C++ order of the `.chd` files is `readdir` order, so comparisons are
keyed by channel. The replica's results are written only to `<name>_legacy.dcc` and never stored
in the sidecar.

## 6. Data formats

### 6.1 Input: adc2kev calibration cache (read-only)

`/coincidences/node_N/board_B/` holds `{anode_,cathode_}{channels, renas (int8), phas (int16),
timestamps (int64 CTS), triggers (int32), file_ids (int8)}`, in file order. `/calibrations`,
`/metadata` (`CacheMetadata`: `created_at`, `ge68_path/hash/size/mtime`, `cs137_…`,
`total_events`) and `/fit_results` also exist. dcalib reads `/coincidences` and `/calibrations`
(through adc2kev) and records the metadata for provenance. The cache is never opened for writing.

### 6.2 Sidecar results `<cache stem>.depth.h5` (D12)

The default path is next to the cache: `data_20260911_124617.cache.h5` →
`data_20260911_124617.depth.h5`. `--results PATH` overrides it, and an unwritable default location
is an error that suggests `--results`.

```
/metadata              dcalib_version, depth_results_version (1.0.0), created_at;
                       cache identity: path, size, mtime, adc2kev created_at, ge68_hash, cs137_hash;
                       calibration source ("cache" | kev path + SHA-256), calibration fingerprint
/results/current       table, one row per anode (= the CSV columns, section 6.4);
                       attrs options_json, created_at, dcalib_version
/results/current/slices   table: anode id, curve (both | ge | cs), slice idx, r_lo, r_hi, r_med,
                          mu, mu_err, sigma, n. The GUI draws the curves and the board grid from
                          it without refitting
/results/current/overrides  per-anode re-fits: the same columns plus options_json per row, with
                            their own slices table
/review                table: anode id, state (rejected), timestamp, note
```

- Writes follow uvcorr: build the new group or table completely, then swap it in. No handle stays
  open between calls, and an HDF5 file lock gets a 1 s retry and then a clear "in use" error.
- **Validity.** The results are *stale* when the cache identity or the calibration fingerprint no
  longer matches. The GUI shows a banner and offers Fit All; `process` replaces the results.
  Overrides are dropped when stale. **Rejections are kept**, because they are human decisions keyed
  by channel, but they are listed in the census so that they can be re-examined.

### 6.3 `<name>.dcc` (D9)

`<name>` is the cache stem (`data_20260911_124617.dcc`). The file has one line per anode whose
status is `ok` and which isn't rejected, sorted by (node, board, rena, channel):
`node board rena channel p0 p1 p2 ` (space-separated, **trailing space**, no header, ostream-style
`%g` with 6 significant digits, |p| < 1e-4 written as `0`, a degree < 2 writes 0 for the higher
terms). It is written atomically.

### 6.4 `depth_summary.csv`

The file starts with `#`-prefixed metadata lines (Generated, cache, calibration source, options), as
in adc2kev's CSVs. It has one row per active anode with any 1A1C event, sorted by key.

| Columns | Meaning |
|---------|---------|
| `node`, `board`, `rena`, `channel`, `electrode` | Address; `ElectrodeMap` label `A01`-`A39` |
| `status`, `flags`, `review`, `options_source` | Section 5.3; `batch` or `override` |
| `n_events`, `n_ge`, `n_cs`, `n_uncal_cathode`, `n_selected` | 1A1C events; per source; dropped for an uncalibrated cathode; in the final selection |
| `ca_p01`, `ca_p50`, `ca_p99` | r coverage |
| `degree`, `p0`, `p1`, `p2`, `err_p0`, `err_p1`, `err_p2`, `chi2ndf`, `n_slices` | Curve in `.dcc` units (511·g) |
| `peak_spread`, `ge_cs_max_diff`, `cv_gain` | Section 5.3 |
| `fwhm_511_before`, `fwhm_511_after`, `fwhm_662_before`, `fwhm_662_after`, `fwtm_*` (the same four) | % |
| `peak_511`, `peak_662` | Photopeak position (E/E0) of the exported spectrum (added in phase 4, for open item O3) |

## 7. Package layout and APIs

```
src/dcalib/
├── options.py        DepthOptions (defaults, JSON round-trip, same_fit), status/flag names
├── channels.py       anode/cathode sets, labels, strip positions (adc2kev ElectrodeMap,
│                     PANEL_NODES, ACTIVE_BOARDS)
├── calib.py          calibration loading (cache | .kev), LUTs, fingerprint
├── events.py         board reader, numba greedy clustering, 1A1C → BoardEvents
├── depth.py          slice fits, curve + degree selection, re-gate, CV gain gate, consistency,
│                     g(r) / correct
├── metrics.py        Gaussian-core FWHM, FWTM, spectra helpers
├── legacy.py         C++ replica (section 5.6)
├── analysis.py       AnodeResult (= CSV row), analyze_anode / board / all (spawn pool, one task
│                     per board, largest first; threadpoolctl limits BLAS to 1 thread per worker)
├── results.py        sidecar HDF5: metadata, results, slices, overrides, review, validity
├── cli.py            dcalib process | legacy
├── io/               dcc.py, summary_csv.py, _atomic.py
└── gui/              main.py, window.py, session.py (data layer), threads.py, controls.py,
                      depth_view.py, spectra.py, board_grid.py, fleet.py, inspector.py,
                      system_map.py + _system_map_model.py (forked from uvcorr, O4),
                      map_colors.py, _layout.py, _flow_layout.py
scripts/              dump_chd.py, compare_dcc.py (C++ cross-check), validate_holdout.py
tests/                unit, CLI and script tests; gui/ (pytest-qt, offscreen); test_realdata.py
docs/                 ALGORITHM.md, GUI.md, images/, planning/ (this file)
```

Tooling is identical to uvcorr and adc2kev: black and ruff (line length 100), mypy with adc2kev's
strict flags, pytest with `--strict-markers` and the markers `gui`, `realdata`, `slow`,
`integration`, and a Makefile with `check`, `dev-check`, `test`, `test-fast`, `test-realdata`,
`lint`, `format`, `type-check` and `cython-check`.

## 8. CLI

```
dcalib [--version] [-v | -vv] {process,legacy} ...

dcalib process CACHE --output-dir DIR [--kev KEV] [--results PATH] [--workers N]
               [--sources both|ge|cs] [--min-pairs N] [--max-degree {0,1,2}] [--min-gain F]
               [--concave-only] [--discard-overrides] [--discard-review]
dcalib legacy  CACHE --output-dir DIR [--kev KEV] [--energy {511,662}] [--no-eof-quirk]
```

**`process`** runs these steps in order:

1. Write-test the output directory.
2. Check the cache is structurally valid (`CalibrationCache.is_cache_structurally_valid`).
3. Load the calibrations and fingerprint them.
4. Open or create the sidecar and read the overrides and review, unless discarded or stale.
5. Analyse every board.
6. Write `<name>.dcc` and `depth_summary.csv` atomically, with the overrides applied and the
   rejections omitted.
7. Store `/results/current`.
8. Print the census (statuses, flags, overrides, rejections, the fleet median FWHM/FWTM before and
   after) and the timings.

The override rule is uvcorr's: an override whose options fit exactly like the new batch is dropped.
Exit codes are uvcorr's: 0, 1 (error), 2 (bad arguments or missing input), 130 (Ctrl-C, with the same
stop-flag behaviour).

**`legacy`** writes `<name>_legacy.dcc` and `legacy_summary.csv` (status, number of pairs, number of
peaks, p0-p2). It never touches the sidecar.

## 9. GUI (`dcalib-gui CACHE [--results PATH] [--kev KEV] [--workers N]`)

The shell follows uvcorr and adc2kev. A control band sits above the tabs:
- **Row 1:** Prev/Next, node/board/channel selectors, the status and flags, **Reject** (a checkable
  button with an optional note).
- **Row 2:** option fields (sources, windows, max degree, min gain), **Fit Channel**, **Fit Board**,
  **Revert to batch**.

Opening a cache with stored, valid results shows them at once. *Process > Fit All* runs
`analyze_all` in a `spawn` pool. *File > Export* writes the `.dcc` and the CSV.

| View | Content |
|------|---------|
| **Depth** tab | Two linked 2D histograms of r against energy (x or keV), before and after the correction. The first overlays the curve g(r), the slice points ± err and optional per-source (Ge, Cs) curves; the second shows the flattened ridge. A residual row shows (μ_i − g(r_i)). Per-cathode colouring or filter, as in specview's C/A tab, exposes a miscalibrated cathode. Toggles for sources and for the selection window. |
| **Spectra** tab | The 511 keV (Ge) and 662 keV (Cs) panels, each with the before and after spectra overlaid, FWHM/FWTM markers and values, and the Gaussian-core fit. |
| **Board grid** tab | The anodes of one board as small multiples in physical strip order (position 1 = low-node side, via `ElectrodeMap.physical_strip_position`): slice points and curve, framed by status colour. Click one to open that anode. Drawn from the stored slices (no refit). |
| **Fleet summary** tab | System-wide histograms of `cv_gain`, `peak_spread`, the FWHM/FWTM change, the degree chosen, `ge_cs_max_diff`; status and flag counts; a sortable worst-N table (click to open the anode). |
| **Fit Inspector** dock | Every CSV field of the anode, the options used, `batch` or `override`, and the review note. |
| **System Map** dock | Both panels at electrode granularity. Anode cells are coloured by status (ok, no_gain, …, rejected, override) or by a metric (`cv_gain`, `peak_spread`, FWHM after). **Cathode cells are coloured by keV-calibration status**, which makes the D3 coverage gap visible. |

Performance targets: selecting an anode on a board not yet loaded takes ≤ 1.5 s (0.2-1 s to load
the board plus the clustering); on a loaded board, ≤ 0.3 s. The board grid and the fleet summary
read only the stored results.

## 10. Validation

- **10.1 Events.**
  - Synthetic hit streams: the inclusive window edge at +48; the anchor rule (a hit at +49 starts a
    new cluster even when it is within 48 of the previous hit); 2A1C, 1A2C and single-polarity
    clusters dropped; sources clustered separately; unsorted input.
  - Real data: the 1A1C counts of section 11 reproduced for n1 b17 and n5 b22.
- **10.2 Depth (synthetic).** Inject known g(r):
  - flat;
  - linear 2 % drop;
  - concave pol2 with a 1.7 % spread;
  - an 8 % spread.

  Add a realistic photopeak (Gaussian core, low-energy tail, flat background, per-source
  normalisation). Check that:
  - the coefficients are recovered within their errors;
  - the degree selection chooses 0 on flat inputs in ≥ 95 % of seeds, so the gain gate's false
    acceptance is ≤ 5 %;
  - 2 % curves are accepted at N ≥ 2,000;
  - a 1 % Ge/Cs offset injected at the ends raises `source_inconsistent`.
- **10.3 Legacy cross-check** (`scripts/`, needs ROOT):
  1. `dump_chd.py --cache CACHE --boards 1:17,5:22,3:20 --out DIR` writes the Ge 1A1C events of
     those boards as extractData `.chd` files (section 4.1: anode line first, `polarity` 1/0, raw
     PHA, no header, the filename pattern above).
  2. Run the Linux-built legacy binary: `Dcalib_linux -p DIR calibration.kev`, which writes
     `DIR/Dcalib_output/<dir name>.dcc`.
  3. `compare_dcc.py CPP_DCC --dcalib FILE` compares the coefficients per anode (default
     `rtol = 1e-3`, since Minuit's convergence tolerance limits the agreement).

  **Done when** every anode with ≥ 3 surviving peaks agrees, and the channel sets are identical.
  This also validates the event layer and the keV conversion against an independent
  implementation.

  **Result (phase 2, 2026-09-24):** boards n1 b17, n5 b22 and n3 b20 (108 `.chd` files, 848,419
  Ge events; `calibration.kev` on both sides). The C++ binary skipped 8 files (uncalibrated anodes)
  and wrote 100 lines; `dcalib legacy --kev calibration.kev` (full system, 58 s in one process)
  gives the same 100 anodes, all with ≥ 3 peaks, and all 100 agree at `rtol = 1e-3`
  (`atol = 1e-3` keV); the largest difference of the curves over r ∈ [0, 1] is 2.1e-4 % of f. The
  EOF quirk (section 4.2) was confirmed separately on hand-made `.chd` files where the last pair
  decides a peak and the `MIN_DATA` check; those C++ outputs are golden values in
  `tests/test_legacy.py`.
- **10.4 Pipeline.**
  - Sidecar atomicity and staleness (change a calibration value, and the fingerprint mismatches).
  - Overrides and rejections survive `process`.
  - `.dcc` round-trip: in a test marked `integration`, the `.dcc` is loaded with the real
    time-calibration loader (`timecalibration.calibration`) when that package is importable, and
    skipped otherwise.
  - The CSV schema.
- **10.5 Held-out validation** (`scripts/validate_holdout.py`):
  1. Build adc2kev `DiagnosticCache`s for the two 2026-09-10 `.dat` files, in a `--work-dir`
     (default scratch, never inside the test-data tree).
  2. Build 1A1C events with `dcalib.events` (identical layout; the source is given per file).
  3. Apply the `.dcc` from the 2026-09-11 cache run, together with that run's calibrations.
  4. Report per anode and fleet-wide FWHM and FWTM, before and after, at 511 and 662 keV, and compare
     with the in-sample and CV numbers.

  **Done when** the report exists and the accepted anodes show a non-negative held-out median gain
  consistent with `cv_gain`, or the gate defaults are retuned until they do. A day of gain drift
  can shift both the before and after numbers; the comparison is paired, so drift largely cancels.
- **10.6 GUI.** pytest-qt, offscreen: open the stored results, navigate from the map, Reject/unreject,
  Fit Channel override and Revert, export contents, and the stale-results banner.

## 11. Measured facts about the test data (2026-09-24, exploratory scripts)

Measured on the full-system cache and `calibration.kev` with ad-hoc versions of sections 5.1-5.3.
The rows marked **(P1)** were re-measured in phase 1 with `dcalib.events` and `dcalib.calib` (cache
calibrations, one process, page cache warm unless stated; `tests/test_realdata.py` pins the n1 b17
and n5 b22 values).

| Quantity | Value |
|----------|-------|
| Boards with events **(P1)** | 156 groups under `/coincidences` (524.7M hits); 146 have at least one 1A1C event, 141 at least 1,000. Some groups are nearly empty (1-4 hits). 97.5M 1A1C events in total |
| Cathodes with a valid keV calibration | 692 / 1248. adc2kev's 511 keV `smart_cathode_edge` fit fails on 230 (228 "No linear leading-edge region found", 2 "Leading flank too short"), and 302 cathodes have no fit at all |
| Ge cathode events on calibrated cathodes | 67.3 %; **24 boards 0 %**, 43 boards < 50 % |
| Board load (`read_board_hits`) **(P1)** | Boards with > 10⁵ hits: median 0.18 s, p90 0.71 s; n1 b17 (2.9M hits) 0.12-0.15 s warm. A cold page cache adds up to ≈ 5 s on the first read of a board. All 156 boards: 49 s in one process |
| Clustering + 1A1C selection **(P1)** | Both sources of a board: median 0.08 s, p90 0.21 s, max 0.36 s (n2 b16, 12.2M hits); 14.6 s for all boards (numba, compiled once and cached) |
| 1A1C share of the clusters **(P1)** | n1 b17: 65.5 % (Ge), 71.0 % (Cs); n5 b22: 58.4 %, 66.1 %. Fleet median 46 % (Ge), 58 % (Cs), p10 10-14 %: some boards have many 1A2C clusters (n1 b15: 41 % of its Ge clusters). The 2A1C share is 14.3 % (Ge) on n1 b17, fleet median 16 % |
| Anode-cathode \|ΔCTS\| within 1A1C **(P1)** | median 0, p99 23 (n1 b17) / 20 (n5 b22), max 48; 31-58 % of the pairs have different trigger numbers, so CTS clustering (not trigger numbers) is needed |
| Pooled Ge+Cs photopeak 1A1C events per calibrated anode **(P1)** | x ∈ [0.75, 1.12], r ∈ [0, 1.3], calibrated cathode: n1 b17: min 6,124, median 7,788 (25 calibrated anodes); n5 b22: min 715, median 1,137 (37) |
| Depth effect (198 anodes, 8 equal-count slices) | photopeak spread across r: median **1.7 %** of E0, p90 8 %; lowest-r slice at 0.996 E0, highest at 0.984 E0 (median); per-slice fit error ≈ 0.2 % |
| Photopeak pairs with C/A > 1 | ≈ 11 %, with the peak at 0.964 (Ge) / 0.975 (Cs) E0. The legacy gate drops them |
| Ge vs Cs, board-pooled, in E/E0 | agree to ≤ 0.4 % for r in [0.2, 0.9]; ≈ 1 % in the lowest and highest bins |
| Uncorrected Gaussian-core FWHM | median 5.7 % (p10 4.6 %, p90 8.2 %) |
| Legacy-style argmax + pol2 on one well-populated anode | FWHM 5.2 % → 6.3 % (worse): fitting noise at the 6 keV bin scale |
| Calibrations: cache vs `calibration.kev` **(P1)** | 5,492 vs 5,491 valid channels (692 cathodes, 4,800 anodes in the cache); the extra one is n5 b21 r0 ch25. The values agree to the 6 decimals `.kev` prints, so the two sources have different fingerprints. Loading the cache's calibrations takes 1.4-1.6 s |
| Legacy C++ on Linux | Builds against `~/root-install` in about 1 s and runs |

## 12. Phases

Commit at the end of every phase (conventional message, no co-author trailer). Each phase must leave
`ruff check`, `black --check`, `mypy src/` and `pytest -m "not realdata"` passing.

| Phase | Scope | Done when |
|-------|-------|-----------|
| 0 | `git init`, `.gitignore` (venv, caches, `*.h5`, build, htmlcov, `.scratch`), venv (section 3), `pyproject.toml`, package skeleton, README stub, Makefile | Install works; the tooling passes on the skeleton |
| 1 | `options.py`, `channels.py`, `calib.py`, `events.py` | 10.1 passes; the section 11 rows for loading, clustering and 1A1C counts re-measured and updated here |
| 2 | `legacy.py`, `dcalib legacy`, `scripts/dump_chd.py`, `scripts/compare_dcc.py` | 10.3 done on three boards |
| 3 | `depth.py`, `metrics.py` | 10.2 passes |
| 4 | `analysis.py`, `results.py`, `io/`, `dcalib process`, full-system census, retuning of the section 5.3 defaults, `docs/ALGORITHM.md` | 10.4 passes; full-system `process` in < 2 min with 8 workers; census and retuned defaults recorded |
| 5 | `scripts/validate_holdout.py` | 10.5 done; the report is in `docs/ALGORITHM.md` |
| 6 | GUI shell: session layer, window, System Map fork, Depth tab, Fit Inspector, Fit All thread | Any anode's depth view can be browsed from the map |
| 7 | Spectra, Board grid and Fleet summary tabs, re-fit overrides, Reject, persistence, export, stale banner | 10.6 passes |
| 8 | README (install, CLI, GUI), `docs/GUI.md` with screenshots, status update of this plan | Reviewed by the user |

## 13. Risks and open items

| # | Item | Default until decided |
|---|------|-----------------------|
| O1 | The numeric defaults in section 5.3 (windows, slice sizes, `min_pairs`, `min_gain`, `p_degree`) are informed guesses | **Retuned** (phases 3-4, `docs/ALGORITHM.md` section 4.2): slice window `[μ − 2σ, μ + 3σ]`, `min_pairs` 500, `max_source_loss` 0.01, `coverage_r_lo` 0.25; the others kept |
| O2 | The legacy concave-only constraint | Off; `convex_curve` flag; revisit after the census |
| O3 | Accepted anodes are re-centred at E0 by p0 while omitted anodes keep the `.kev` scale (≲ 0.5 % offset) | Accept; report the fleet offset in the census. **Measured:** about 0.1 % (median peak 0.9984 vs 0.9976 E/E0 at 511 keV); the census prints it |
| O4 | The System Map would become a third fork (adc2kev → uvcorr → dcalib) | Fork uvcorr's (it already has metric colouring and overrides). Upstreaming a generic map into adc2kev is a follow-up for the user to decide |
| O5 | Consumers evaluate the pol2 unclamped (D8) | `extrapolation_risk` flag. W14 gates C/A to [0, 1.2]; time-calibration does not |
| O6 | The 48-tick window is the legacy constant, while the measured \|ΔCTS\| p99 is ≈ 23 | Keep 48 (parity with extractData); `cts_window` option |
| O7 | Coverage is capped by D3: anodes on the 24 boards without calibrated cathodes get no correction | The census reports it; the adc2kev work item is separate |
| O8 | Pairs from all cathodes crossing an anode are pooled (as legacy). A miscalibrated cathode smears r | The GUI per-cathode colouring; possible later flag from per-cathode r-edge outliers |
| O9 | Held-out data is from another day (gain drift) | Paired before/after comparison; report both days |

## 14. Reference files

| Path | What |
|------|------|
| `~/DataProcessing/Dcalib/main.cpp`, `README.md`, `CHANGELOG.md`, `BUG_FIXES.md` | Legacy Dcalib |
| `~/DataProcessing/extractData/data_kimia_edit/main.cpp`, `channelwriter.cpp`, `data_original/`, `compare_chd.sh` | `.chd` production |
| `~/DataProcessing/extractData/data_kimia_edit/data_20250730_114926channelData/` | Sample `.chd` set (4,923 files) |
| `~/time-calibration/src/timecalibration/calibration.py`, `core.py` | Main `.dcc` consumer |
| `~/tp-processing/rilpetlib/src/rilpetlib/timecalibration.py` | Identical consumer copy |
| `~/DataProcessing/TimeCalibration/MATLAB_Code/helper_functions/importDcc.m` | MATLAB reader (no header allowed) |
| `~/tp-processing/rilpetlib/scripts/timecalibration/calibration_data/data_20250917_134552channelData.dcc` | Example legacy `.dcc` (78 lines) |
| `~/adc2kev-python/src/adc2kev/analysis/ca_ratio.py`, `input_resolution.py` | Board loading, LUTs, C/A pairing |
| `~/adc2kev-python/src/adc2kev/cache/hdf5_cache.py`, `diagnostic_cache.py` | Cache API (`load_all_calibrations`, `get_metadata`, `is_cache_structurally_valid`) |
| `~/adc2kev-python/src/adc2kev/tools/electrode_map.py`, `geometry.py` | Labels, polarity, strip order, panel grid |
| `~/adc2kev-python/docs/technical/PAIRING_APPROACH.md`, `HARDWARE_CONSTRAINTS.md` | Matcher and channel ranges |
| `~/uv-ellipse-correction/` (README, `docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md`, `src/uvcorr/gui/`) | Template: package layout, sidecar results and overrides, CLI, GUI, System Map fork |
| `~/DAQ_processing/doi_analysis.py`, `doi_ca_mapping.py` | Related DOI analyses (not consumers) |
