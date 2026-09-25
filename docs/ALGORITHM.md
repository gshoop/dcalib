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

The gain noise at N = 2,000 is ±0.012, against ±0.025 for the best width-based variant.

### 3.3 Resolution metrics

`fwhm_*` and `fwtm_*` are the full widths at half and a tenth of the maximum of the spectrum smoothed
with a Gaussian kernel (σ = 0.004 in x on a 0.0005 grid over [0.6, 1.3]), in % of the peak position.
The plan defined `fwtm_*` this way ("a lightly smoothed histogram") but `fwhm_*` as the Gaussian-core
FWHM, which section 3.2 shows is blind to depth broadening. The kernel widens a 5.7 % FWHM by about
1.4 % of itself.

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
