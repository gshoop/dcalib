# dcalib algorithm notes

This document records how dcalib's depth calibration works where it goes beyond, or departs from,
the plan ([`planning/DEPTH_CALIBRATION_PLAN.md`](planning/DEPTH_CALIBRATION_PLAN.md)), and the
studies behind those choices. The module docstrings of `dcalib.events`, `dcalib.legacy`,
`dcalib.metrics` and `dcalib.depth` describe each step in full.

## 1. Events (phase 1)

One-anode/one-cathode (1A1C) events are built per board and per source with a greedy 48-tick
inclusive CTS window anchored on the first hit (plan 5.1). extractData anchors its windows on the
time-sorted hits of the whole system; the cache stores each board separately, so dcalib anchors them
per board. Real-data checks are in the plan's section 11 (rows marked P1).

## 2. Legacy replica (phase 2)

`dcalib legacy` reproduces the C++ Dcalib step by step (`dcalib.legacy`). One detail differs from
the plan's first description: the end-of-file re-read converts the last pair's keV values a
*second* time (the C++ converts `aEn`/`cEn` in place), so the re-read pair almost never passes the
gates and `pop_back()` drops the last accepted *real* pair. A file with 19 accepted pairs is
therefore skipped (`MIN_DATA = 20`). Hand-made `.chd` files confirmed this against the ROOT binary;
their outputs are golden values in `tests/test_legacy.py`.

**Cross-check (plan 10.3).** Boards n1 b17, n5 b22, n3 b20 (108 `.chd` files, 848,419 Ge events),
`calibration.kev` on both sides: the Linux build of the ROOT binary and `dcalib legacy` write the same
100 anodes, all 100 agree at `rtol = 1e-3` and the largest curve difference over r ∈ [0, 1] is
2.1e-4 % of f.

```bash
venv/bin/python scripts/dump_chd.py --boards 1:17,5:22,3:20 --out $SCRATCH/chd
Dcalib_linux -p $SCRATCH/chd ~/adc2kev-test-data/full-system/calibration.kev
venv/bin/dcalib legacy CACHE --kev calibration.kev --output-dir $SCRATCH/py
venv/bin/python scripts/compare_dcc.py $SCRATCH/chd/Dcalib_output/chd.dcc \
    --dcalib $SCRATCH/py/data_20260911_124617_legacy.dcc
```

## 3. Default depth fit: the phase 3 synthetic study

Plan 10.2 asks that, on synthetic anodes with a realistic photopeak, a flat curve chooses degree 0
in at least 95 % of the seeds, 2 % curves are accepted at N ≥ 2,000, the coefficients are recovered
within their errors, and a 1 % Ge/Cs offset raises `source_inconsistent`. The synthetic anodes
(`tests/synthetic_depth.py`) have r uniform on [0.02, 1.15], half Ge and half Cs, a Gaussian core
of σ = 0.024 in x (FWHM 5.7 %, 10 % narrower for Cs), a 15 % exponential low-energy tail (scale
0.03) and a 10 % flat background on [0.6, 1.3]. The curves are flat, a 2 % linear drop, a concave
pol2 with a 1.7 % spread (the fleet median) and an 8 % spread (the fleet p90).

The plan's first version of step 3 and step 7 did not meet these targets. Three changes did.

### 3.1 Slice fit window: `[μ − 2σ, μ + 3σ]`, width fixed at the pooled σ

The slice positions (Gaussian core + linear background, binned Poisson likelihood) set the
sensitivity of the degree test. With the plan's window `[μ − 1.5σ, μ + 2.5σ]` re-sized to each
fit's own σ, a slice of 400 events had a position error of 0.0047 (in x), four times the ideal
σ/√n. Two causes: the narrow window leaves the Gaussian and the free background slope strongly
correlated, and re-sizing the window to a noisy σ shrinks it further (on a few hundred events the
width can collapse over the iterations). Measured on 400 events, 400 seeds (position standard
deviation, bias from the tail):

| Window | Width | Std of μ | Bias |
|---|---|---|---|
| [−1.5σ, +2.5σ] | re-sized to each fit | 0.0047 | +0.0009 |
| [−1.5σ, +2.5σ] | fixed at the seed σ | 0.0039 | +0.0006 |
| [−2.0σ, +3.0σ] | fixed at the seed σ | 0.0023 | −0.0009 |
| [−2.5σ, +3.0σ] | fixed at the seed σ | 0.0019 | −0.0015 |

`DepthOptions.window_lo_sigma` / `window_hi_sigma` are now 2.0 / 3.0 and every fit keeps its
window width at its seed (the anode's pooled σ, or the smoothed half maximum) and only re-centres
it on the fitted mean. The asymmetry still keeps most of the tail out; the remaining bias is the
same for every equal-count slice and only shifts the curve's constant term. The phase 4 census
checks this choice on real data, where the tail may depend on r.

### 3.2 Gain gate: cross-fitted alignment of the photopeak positions

Plan step 7 compared the Gaussian-core FWHM of `x / g(r)` with that of `x`. That fails in two ways.

1. **The core fit does not see depth broadening.** Depth smearing mostly adds a low-side
   shoulder, and the free linear background in the asymmetric window absorbs it. With the *true*
   curve of the 8 % case the core FWHM went from 5.09 % to 5.52 % (worse), while the actual peak
   (FWHM of the smoothed spectrum) narrowed by 37 %.
2. **Widths are too noisy.** Even measured non-parametrically, a width on ~1,000 events per source
   has 3-6 % relative noise, as large as the 2-3 % gain of a 2 % curve. With the true curve, the
   per-source gain of the 2 % case was 0.018 ± 0.055 (FWHM) and 0.027 ± 0.037 (FWTM).

Paired width gains on ~500 events per source (40 seeds; the truth is from 400,000 events):

| Estimator | 2 % linear | 1.7 % concave | 8 % | Truth (2 % / 1.7 % / 8 %) |
|---|---|---|---|---|
| Gaussian-core FWHM | 0.080 ± 0.193 | 0.031 ± 0.297 | −0.49 ± 1.56 | 0.042 / 0.031 / 0.008 |
| Smoothed-spectrum FWHM | 0.035 ± 0.062 | 0.034 ± 0.061 | 0.317 ± 0.091 | 0.034 / 0.035 / 0.368 |
| Smoothed-spectrum FWTM | 0.032 ± 0.038 | 0.032 ± 0.038 | 0.330 ± 0.039 | 0.033 / 0.033 / 0.330 |

What the correction does is line up the photopeak positions across r, and positions are measured
about ten times more precisely than widths. The gate therefore measures that alignment on held-out
events:

- The anode's rows are split into even and odd halves. Each half is fitted (steps 1-5) at the
  degree chosen on all events, and its curve corrects the *other* half (cross-fitting). Refitting
  the chosen degree, rather than repeating the degree test on half the statistics, matters: at
  N = 2,000 the halves often chose degree 0 for a real 2 % curve, and a constant correction has
  exactly zero gain.
- Per source, the events are sliced in r (as in step 2) and the slice positions of the raw and of
  the out-of-fold corrected energies are fitted on the same slices. The spread of the positions
  minus their mean squared fit error is the depth variance `V`.
- The predicted width ratio is `sqrt((σ² + V_corr) / (σ² + V_raw))` with the slices' median core
  σ; `cv_gain = 1 − ratio`, averaged over the sources. The acceptance rule is the plan's
  (`cv_gain ≥ min_gain`, no source loss above `max_source_loss`).

The gain noise at N = 2,000 is ±0.012, against ±0.025 for the best width-based variant. (Phase 4
pairs the slices and scales by the real peak width; section 4.1.)

### 3.3 Resolution metrics

`fwhm_*` and `fwtm_*` are widths of the spectrum smoothed with a Gaussian kernel, in % of the peak
position. The plan defined `fwtm_*` this way ("a lightly smoothed histogram") but `fwhm_*` as the
Gaussian-core FWHM, which section 3.2 shows is blind to depth broadening. Phase 4 refined the
definition on real spectra (section 4.1).

### 3.4 Results

100 seeds each, default options (`tests/test_depth.py` asserts the plan's thresholds on these):

| Curve | N | ok | no_gain | no_depth_dependence | cv_gain |
|---|---|---|---|---|---|
| flat | 2,000 | 0 | 1 | 99 | 0.000 ± 0.000 |
| flat | 8,000 | 0 | 1 | 99 | 0.001 ± 0.000 |
| 2 % linear | 2,000 | 97 | 1 | 2 | 0.033 ± 0.013 |
| 2 % linear | 4,000 | 99 | 1 | 0 | 0.036 ± 0.009 |
| 1.7 % concave | 2,000 | 95 | 3 | 2 | 0.029 ± 0.012 |
| 1.7 % concave | 4,000 | 100 | 0 | 0 | 0.035 ± 0.009 |
| 8 % | 2,000 | 100 | 0 | 0 | 0.369 ± 0.025 |

The degree test alone chooses degree 0 on 98-99 % of flat anodes (and on 294 of 300 in a separate
run without the re-gate), so the gate's false acceptance stays well under 5 %. On 20,000-event
anodes the coefficient pulls have a standard deviation of 0.9 (the errors are honest); c0 is biased
by −0.75σ (about −0.1 %) from the tail. A Ge/Cs tilt of 1.25 % at the 10 % and 90 % r quantiles
raises `source_inconsistent` on 30 of 30 seeds, and never without the tilt.

### 3.5 Peak fitter implementation

The binned Poisson likelihood is minimised as least squares on the signed root-deviance residuals,
like `scipy.optimize.least_squares(method="lm")`, but with a Levenberg-Marquardt loop compiled with
numba (with an active set for the non-negative background). At about a hundred peak fits per anode
scipy's per-call overhead would take 3.4 ms per fit (`trf`); the compiled loop takes 0.15 ms and
reaches the same optimum (`tests/test_metrics.py::test_lm_matches_scipy_least_squares`). A whole
anode fit, including the cross-fitting and the per-source curves, takes about 10 ms.

## 4. Phase 4: real data, retuning and the full-system census

### 4.1 What the real spectra changed

The first run on the full-system cache showed four problems the synthetic anodes did not have.

1. **Slices locked onto the continuum.** At r > 1 the Compton continuum near x ≈ 0.8 can rival the
   photopeak within a slice. A slice seeded at its global mode then fitted a "peak" at 0.82, and one
   such slice made an anode's cross-validated gain −72 %. Slice fits now seed at the slice's mode
   *within 0.15 of the anode's pooled peak* (`MAX_SLICE_SHIFT`) and reject positions further away.
   Seeding at the pooled peak itself failed on the far slices of an 8 % curve instead.
2. **Raw and corrected slices were compared out of step.** When a slice fit failed on one side
   only, the remaining slices no longer corresponded. The gate now fits both on the same slices and
   compares only slices that succeeded on both sides.
3. **The gain was judged against the Gaussian core.** The alignment ratio
   `sqrt((σ² + V_corr)/(σ² + V_raw))` used the slices' core σ, which is narrower than a real
   peak and overstated the gain (+6 % on an anode whose spectrum barely changed). σ is now the
   corrected spectrum's FWHM / 2.355 (below).
4. **The widths were undefined or unstable.** At 511 keV the CZT Compton continuum stays above a
   tenth of the photopeak down to x = 0.6, so the tenth-maximum crossing did not exist (every
   anode of n1 b17 had no FWTM). A 0.004 kernel also resolved fine structure at the top of real
   peaks, making the half-maximum crossings jump (6.81 % → 7.33 % on an anode whose FWHM with a
   0.008 kernel went 7.72 % → 7.59 %). The widths are now measured on the **net photopeak**: the
   maximum is searched in x ∈ [0.85, 1.15], the continuum level is the median of the smoothed
   spectrum 0.18-0.10 below the peak, the crossings are those of `B + f·(max − B)`, and the
   kernel σ is 0.008 (it widens a 5.7 % FWHM by about 5 % of itself; a continuum that ends under
   the low half of the peak narrows the widths by up to about 10 %). A spectrum whose continuum
   exceeds 60 % of the maximum has no width. These are comparative metrics (the same anode before
   and after, and across the fleet), not absolute resolutions.

The peak fitter's Levenberg-Marquardt loop also needed a looser stopping rule: with many empty
bins, Gauss-Newton converges only linearly and could exhaust its iterations while still improving
the deviance by 1e-8 per step (far below the statistical errors). The relative tolerance is now
1e-8 (MINPACK's default is 1.49e-8), and a fit still creeping by less than 1e-6 at the iteration
limit is accepted. This took `slice_fit_failed` from 425 anodes to 50.

### 4.2 Retuning (open item O1)

Full-system runs with each retuned default reverted in turn (final code, 8 workers). The FWHM
columns are fleet medians over all fitted anodes, after = the exported spectrum (corrected for `ok`
anodes, raw otherwise); "worse" counts `ok` anodes whose in-sample 511 keV FWHM grew by more than 1 %.

| Variant | ok | no_gain | no_depth_dep. | too_few | fit_failed | FWHM 511 | FWHM 662 | mean FWHM ratio 511 / 662 | worse | narrow flags |
|---|---|---|---|---|---|---|---|---|---|---|
| **final defaults** | 2,668 | 672 | 1,149 | 196 | 9 | 6.18 → 5.99 | 5.16 → 4.95 | 0.971 / 0.962 | 176 | 198 |
| window 1.5σ/2.5σ (plan) | 1,797 | 972 | 1,699 | 195 | 31 | 6.18 → 6.02 | 5.16 → 4.99 | 0.977 / 0.969 | 113 | 198 |
| `min_pairs` 800 (plan) | 2,634 | 636 | 1,024 | 398 | 2 | 6.18 → 5.99 | 5.16 → 4.95 | 0.971 / 0.963 | 169 | 198 |
| `max_source_loss` 0.005 (plan) | 2,649 | 691 | 1,149 | 196 | 9 | 6.18 → 5.99 | 5.16 → 4.95 | 0.971 / 0.962 | 174 | 198 |
| `coverage_r_lo` 0.15 (plan) | 2,668 | 672 | 1,149 | 196 | 9 | 6.18 → 5.99 | 5.16 → 4.95 | 0.971 / 0.962 | 176 | 1,067 |
| `min_gain` 0.005 (not adopted) | 2,885 | 455 | 1,149 | 196 | 9 | 6.18 → 5.98 | 5.16 → 4.95 | 0.970 / 0.961 | 196 | 198 |

- **Slice window `[μ − 2σ, μ + 3σ]`** (kept from phase 3): the plan's window accepts a third fewer
  anodes and improves the fleet less.
- **`min_pairs` 500** (was 800): the 202 extra anodes have many 1A1C events but few with a
  calibrated cathode (median 13,500 events, 504 selected); at 4 × 125 events per slice the degree
  test and the gate still protect them, and 34 more anodes are corrected.
- **`max_source_loss` 0.01** (was 0.005): the per-source width ratio has a noise of about
  ±0.01 at typical statistics, so 0.005 rejected corrections with a 1-4 % gain on noise.
- **`coverage_r_lo` 0.25** (was 0.15): r typically starts near 0.1 (the fleet median of the 1 %
  quantile is 0.092, p90 0.18), so 0.15 flagged a fifth of the fleet; 0.25 flags the unusual 4 %.
- **`min_gain` stays 0.01**: 0.005 adds 217 marginal anodes, and more of them get worse in sample.
- `p_degree` (0.01), `per_slice`, `concave_only` (off; 12 convex curves, open item O2) and the
  consistency thresholds were left as they are.

### 4.3 Census (final defaults)

`dcalib process` on `data_20260911_124617.cache.h5` (cache calibrations; 5,269 anodes with 1A1C
events, 97.5M events):

| Status | Anodes | Notes |
|---|---|---|
| ok | 2,668 | written to the `.dcc` (1,578 quadratic, 1,090 linear) |
| no_gain | 672 | cv_gain median 0.4 % (p90 1.0 %) |
| no_depth_dependence | 1,149 | slice spread median 1.2 % (ok anodes: 2.6 %) |
| too_few_events | 196 | |
| fit_failed | 9 | |
| no_calibrated_cathode | 84 | open item O7 / decision D3 |
| no_anode_calibration | 491 | |

Flags: `partial_cathode_coverage` 1,045 (131 of them still `ok`), `source_inconsistent` 437,
`narrow_ca_coverage` 198, `slice_fit_failed` 50, `extrapolation_risk` 19, `convex_curve` 12.

- **Depth effect.** The slice spread of the fitted anodes has a median of 2.1 % (p10 1.0 %, p90
  4.1 %); the plan's exploratory 1.7 % / 8 % used another peak finder. The r quantiles have
  medians 0.09 (1 %), 0.48 (50 %) and 1.09 (99 %). The Ge/Cs curves differ by 0.6 % at the
  median (p90 1.2 %) over the 10-90 % r range.
- **Gain.** cv_gain of the `ok` anodes: p10 1.6 %, median 4.3 %, p90 13.7 %.
- **Resolution** of the `ok` anodes (median, %; ratio after/before p10 / median / p90):
  FWHM 511 keV 6.22 → 5.90 (0.875 / 0.963 / 1.004), FWHM 662 keV 5.21 → 4.87 (0.839 / 0.955 /
  0.996), FWTM 511 keV 11.48 → 11.05 (0.897 / 0.972 / 1.008), FWTM 662 keV 10.01 → 9.54 (0.872 /
  0.967 / 1.006). In sample, 6.6 % of the `ok` anodes have a 511 keV FWHM more than 1 % wider
  after the correction (4.1 % at 662 keV). This is the width measurement's noise: section 5
  shows that a single anode's paired FWHM ratio scatters by about 2 %.
- **Scale (open item O3).** The median photopeak position of the exported spectra is 0.9984
  (511 keV) and 0.9990 (662 keV) for corrected anodes, and 0.9976 / 0.9980 for omitted ones: the
  offset between the two groups is about 0.1 %, well inside the ≲ 0.5 % the plan accepts.

### 4.4 Run time

`dcalib process`, full system, cache in the page cache (24 cores, WSL2):

| Workers | Wall time | Peak RSS (largest process) |
|---|---|---|
| 1 | 173 s | 2.2 GB |
| 4 | 48 s | 2.1 GB |
| 8 | 31 s | 2.1 GB |

Of the 31 s, loading and fingerprinting the calibrations takes 1.1 s, writing the outputs and the
sidecar 0.2 s.

## 5. Held-out validation (phase 5, plan 10.5)

`scripts/validate_holdout.py` applies the 2026-09-11 run (the `.dcc` and the cache's calibrations
of section 4.3) to the raw acquisitions of the previous day
(`sources/ge/data_20260910_120628.dat`, 208M events, and `sources/cs/data_20260910_121135.dat`).
It parses them into adc2kev `DiagnosticCache`s in its work directory (65 s per file), builds the
1A1C events with `dcalib.events` and measures every anode of the in-sample CSV on these events,
before and after the in-sample correction:

```bash
venv/bin/python scripts/validate_holdout.py --run OUT_OF_DCALIB_PROCESS --workers 8
# writes .scratch/holdout/holdout_summary.csv and holdout_report.md (14 s with the caches built)
```

2,605 of the 2,668 corrected anodes have held-out events. Paired widths (the same held-out
events before and after):

| Metric | Median before | Median after | Ratio after/before p10 / median / p90 | In-sample median ratio |
|---|---|---|---|---|
| FWHM 511 keV | 6.20 % | 5.92 % | 0.880 / 0.969 / 1.016 | 0.963 |
| FWTM 511 keV | 11.54 % | 11.10 % | 0.897 / 0.976 / 1.012 | 0.972 |
| FWHM 662 keV | 5.17 % | 4.89 % | 0.849 / 0.961 / 1.011 | 0.955 |
| FWTM 662 keV | 10.01 % | 9.56 % | 0.874 / 0.969 / 1.015 | 0.967 |

- **Gain.** The alignment gain (the `cv_gain` estimator, with the in-sample curve on the held-out
  events) has p10 0.5 %, median 3.6 % and p90 12.8 %, and is negative for 4.3 % of the corrected
  anodes. The in-sample `cv_gain` of the same anodes has a median of 4.3 %: it is slightly
  optimistic (median difference −0.6 points) and correlates with the held-out gain at 0.74. The
  held-out width ratios are within a point of the in-sample ones.
- **Anodes that look worse.** 11-12 % of the corrected anodes have a held-out width more than 1 %
  larger after the correction. Bootstrapping 73 corrected anodes of five boards gives a paired
  FWHM ratio noise of 2.2 % per anode (p90 6.7 %), so at the median ratio about 15 % of the
  anodes would exceed 1.01 from the measurement noise alone: the "worse" anodes are consistent
  with noise, not with harmful corrections.
- **Drift.** With the 2026-09-11 calibration, the uncorrected held-out photopeaks sit at 0.9961
  (511 keV) and 0.9983 (662 keV) E/E0: the gains drifted by about 0.4 % and 0.2 % between the
  days. The paired comparison is insensitive to it (open item O9).

**Conclusion (plan 10.5 done when).** The corrected anodes have a positive held-out median gain
(3.6 % in the alignment measure; FWHM −3.1 % at 511 keV and −3.9 % at 662 keV) that is consistent
with `cv_gain` (4.3 %) and with the in-sample widths, so the gate defaults were not retuned
further.
