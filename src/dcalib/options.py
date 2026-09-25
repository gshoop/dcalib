"""Depth-fit options, sources and the status/flag vocabulary of the whole package.

``DepthOptions`` holds every threshold of the event building and the default
depth fit (plan sections 5.1 and 5.3). The numeric defaults are the plan's
initial values except four, retuned (open item O1, ``docs/ALGORITHM.md``):
the slice fit window ``[mu - 2 sigma, mu + 3 sigma]`` (was 1.5/2.5 sigma;
phase-3 synthetic study, confirmed by the phase-4 census), ``min_pairs`` 500
(was 800), ``max_source_loss`` 0.01 (was 0.005, below the per-source noise)
and ``coverage_r_lo`` 0.25 (was 0.15, which flagged a fifth of the fleet: r
typically starts near 0.1).

The status, flag and review strings below are the single source of truth for
the ``status``, ``flags`` and ``review`` CSV columns (plan 5.3 and 6.4).

:func:`effective_options` and :func:`same_fit` tell whether two option sets fit
identically. ``dcalib process`` and the GUI's Fit All use them to drop the
stored overrides that a new batch reproduces, and the GUI to choose between
storing a re-fit as an override and reverting to the batch.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, fields, replace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sources (the adc2kev cache's file_id)
# ---------------------------------------------------------------------------

SOURCE_GE = 0
"""file_id of the Ge-68 acquisition (511 keV annihilation photons)."""

SOURCE_CS = 1
"""file_id of the Cs-137 acquisition (662 keV)."""

SOURCE_IDS: tuple[int, ...] = (SOURCE_GE, SOURCE_CS)

SOURCE_ENERGY_KEV: dict[int, float] = {SOURCE_GE: 511.0, SOURCE_CS: 662.0}
"""Photopeak energy E0 of each source; ``x = A_keV / E0``."""

SOURCE_NAMES: dict[int, str] = {SOURCE_GE: "ge", SOURCE_CS: "cs"}

SOURCES_BOTH = "both"
SOURCE_CHOICES: tuple[str, ...] = (SOURCES_BOTH, "ge", "cs")
"""Values of ``DepthOptions.sources``."""

DCC_ENERGY_KEV = 511.0
"""Energy the ``.dcc`` consumers normalise to: ``A_corr = A * 511 / f(r)`` (plan 5.4)."""


def selected_sources(sources: str) -> tuple[int, ...]:
    """Return the source ids selected by a ``DepthOptions.sources`` value.

    Args:
        sources: ``"both"``, ``"ge"`` or ``"cs"``.

    Returns:
        ``(0, 1)``, ``(0,)`` or ``(1,)``.

    Raises:
        ValueError: If ``sources`` is not one of ``SOURCE_CHOICES``.
    """
    if sources == SOURCES_BOTH:
        return SOURCE_IDS
    for source_id, name in SOURCE_NAMES.items():
        if sources == name:
            return (source_id,)
    raise ValueError(f"sources must be one of {SOURCE_CHOICES}, got {sources!r}")


# ---------------------------------------------------------------------------
# Anode status (exactly one per anode)
# ---------------------------------------------------------------------------

STATUS_OK = "ok"
"""The correction was accepted and is written to the ``.dcc``."""

STATUS_NO_GAIN = "no_gain"
"""The cross-validated resolution gain is below ``min_gain``, or a source worsens by
more than ``max_source_loss``; the anode is omitted from the ``.dcc``."""

STATUS_NO_DEPTH_DEPENDENCE = "no_depth_dependence"
"""The degree test chose a constant curve; there is nothing to correct."""

STATUS_TOO_FEW_EVENTS = "too_few_events"
"""Fewer than ``min_pairs`` events pass the selection."""

STATUS_FIT_FAILED = "fit_failed"
"""Fewer than 3 slice photopeak fits succeeded (or the curve fit failed)."""

STATUS_NO_CALIBRATED_CATHODE = "no_calibrated_cathode"
"""Every 1A1C event of the anode has a cathode without a keV calibration (plan D3)."""

STATUS_NO_ANODE_CALIBRATION = "no_anode_calibration"
"""The anode itself has no valid keV calibration."""

STATUSES: tuple[str, ...] = (
    STATUS_OK,
    STATUS_NO_GAIN,
    STATUS_NO_DEPTH_DEPENDENCE,
    STATUS_TOO_FEW_EVENTS,
    STATUS_FIT_FAILED,
    STATUS_NO_CALIBRATED_CATHODE,
    STATUS_NO_ANODE_CALIBRATION,
)
"""All valid statuses, in the order the census reports them."""

# ---------------------------------------------------------------------------
# Warning flags (zero or more per anode)
# ---------------------------------------------------------------------------

FLAG_SLICE_FIT_FAILED = "slice_fit_failed"
"""At least one slice photopeak fit failed and the slice was dropped."""

FLAG_CONVEX_CURVE = "convex_curve"
"""The chosen curve is quadratic with a significantly positive p2 (legacy forbade it)."""

FLAG_SOURCE_INCONSISTENT = "source_inconsistent"
"""The Ge-only and Cs-only curves disagree over the anode's 10-90 % r range."""

FLAG_EXTRAPOLATION_RISK = "extrapolation_risk"
"""g(r) leaves the plausible band somewhere on the evaluation range of r."""

FLAG_NARROW_CA_COVERAGE = "narrow_ca_coverage"
"""The 1-99 % r range of the data is narrower than the coverage reference range."""

FLAG_PARTIAL_CATHODE_COVERAGE = "partial_cathode_coverage"
"""More than ``partial_cathode_frac`` of the anode's events had an uncalibrated cathode."""

FLAGS: tuple[str, ...] = (
    FLAG_SLICE_FIT_FAILED,
    FLAG_CONVEX_CURVE,
    FLAG_SOURCE_INCONSISTENT,
    FLAG_EXTRAPOLATION_RISK,
    FLAG_NARROW_CA_COVERAGE,
    FLAG_PARTIAL_CATHODE_COVERAGE,
)
"""All valid flags, in the canonical order used when an anode carries several."""

FLAG_SEPARATOR = ";"
"""Separator between flags in the CSV ``flags`` column."""

# ---------------------------------------------------------------------------
# Review (plan D10): a separate column, "rejected" or empty
# ---------------------------------------------------------------------------

REVIEW_REJECTED = "rejected"
REVIEW_NONE = ""
REVIEW_STATES: tuple[str, ...] = (REVIEW_NONE, REVIEW_REJECTED)


def order_flags(flags: Iterable[str]) -> tuple[str, ...]:
    """Return flags de-duplicated and in the canonical order of ``FLAGS``.

    Args:
        flags: Flag strings, each one of ``FLAGS``.

    Returns:
        The distinct flags, ordered as in ``FLAGS``.

    Raises:
        ValueError: If a flag is not one of ``FLAGS``.
    """
    present = set(flags)
    unknown = present - set(FLAGS)
    if unknown:
        raise ValueError(f"Unknown flag(s): {sorted(unknown)}")
    return tuple(flag for flag in FLAGS if flag in present)


def join_flags(flags: Iterable[str]) -> str:
    """Return the CSV ``flags`` value: canonical order, ``;``-separated."""
    return FLAG_SEPARATOR.join(order_flags(flags))


def split_flags(text: str) -> tuple[str, ...]:
    """Inverse of :func:`join_flags` (empty text gives no flags)."""
    return order_flags(part for part in text.split(FLAG_SEPARATOR) if part)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthOptions:
    """Options of the event building and the default depth fit (plan 5.1, 5.3).

    Instances are immutable and validated on construction; two instances with
    the same values compare equal. Energies are in normalised units
    ``x = A_keV / E0`` and depth in ``r = C_keV / A_keV``, both uncorrected.

    Attributes:
        sources: Sources to fit: ``"both"`` (pooled in x), ``"ge"`` or ``"cs"``.
        cts_window: Greedy clustering window in CTS ticks (inclusive; 48 = 1 us).
        x_lo: Lower edge of the photopeak selection in x.
        x_hi: Upper edge of the photopeak selection in x.
        r_lo: Lower edge of the r selection.
        r_hi: Upper edge of the r selection (above 1, unlike the legacy gate).
        min_pairs: Fewest selected events for an anode to be fitted.
        per_slice: Target events per r slice: ``n_slices = clamp(N // per_slice,
            min_slices, max_slices)``.
        min_slices: Fewest r slices (at least 3).
        max_slices: Most r slices.
        hist_lo: Lower edge of the slice photopeak histogram in x.
        hist_hi: Upper edge of the slice photopeak histogram in x.
        hist_bin: Bin width of the slice photopeak histogram in x.
        window_lo_sigma: Fit window starts at ``mu - window_lo_sigma * sigma``.
        window_hi_sigma: Fit window ends at ``mu + window_hi_sigma * sigma``.
        window_iterations: Times the fit window is re-centred and re-fitted.
        max_degree: Highest polynomial degree of g(r) (0, 1 or 2).
        p_degree: Significance level of the nested chi-square degree test.
        concave_only: Constrain p2 <= 0, as the legacy fit did.
        gate_iterations: Times the selection is redone on the corrected energy.
        min_gain: Smallest cross-validated relative FWHM gain that accepts a
            correction.
        max_source_loss: Largest relative FWHM loss any single source may show
            for the correction to be accepted.
        consistency_min_diff: Floor of the Ge/Cs curve difference that raises
            ``source_inconsistent``.
        consistency_nsigma: The difference must also exceed this many standard
            errors of the difference.
        consistency_min_slices: Each source needs ``consistency_min_slices *
            per_slice`` selected events for its own curve to be compared.
        extrap_r_max: g(r) is checked for extrapolation on ``[0, extrap_r_max]``.
        extrap_g_lo: ``extrapolation_risk`` when g drops below this there.
        extrap_g_hi: ``extrapolation_risk`` when g rises above this there.
        coverage_r_lo: ``narrow_ca_coverage`` when the 1 % r quantile is above this.
        coverage_r_hi: ``narrow_ca_coverage`` when the 99 % r quantile is below this.
        partial_cathode_frac: Fraction of events with an uncalibrated cathode
            above which ``partial_cathode_coverage`` is set.

    Example:
        >>> opts = DepthOptions(max_degree=1)
        >>> opts.is_default()
        False
        >>> DepthOptions.from_json(opts.to_json()) == opts
        True
    """

    sources: str = SOURCES_BOTH
    cts_window: int = 48
    x_lo: float = 0.75
    x_hi: float = 1.12
    r_lo: float = 0.0
    r_hi: float = 1.3
    min_pairs: int = 500
    per_slice: int = 400
    min_slices: int = 4
    max_slices: int = 12
    hist_lo: float = 0.80
    hist_hi: float = 1.15
    hist_bin: float = 0.0025
    window_lo_sigma: float = 2.0
    window_hi_sigma: float = 3.0
    window_iterations: int = 3
    max_degree: int = 2
    p_degree: float = 0.01
    concave_only: bool = False
    gate_iterations: int = 1
    min_gain: float = 0.01
    max_source_loss: float = 0.01
    consistency_min_diff: float = 0.01
    consistency_nsigma: float = 3.0
    consistency_min_slices: int = 3
    extrap_r_max: float = 1.3
    extrap_g_lo: float = 0.85
    extrap_g_hi: float = 1.10
    coverage_r_lo: float = 0.25
    coverage_r_hi: float = 0.95
    partial_cathode_frac: float = 0.5

    def __post_init__(self) -> None:
        """Validate and normalise the field types.

        Integers are accepted for float fields (and stored as float); numpy
        scalars are converted to the Python types. Booleans are rejected for
        numeric fields.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range.
        """
        for f in fields(self):
            value = getattr(self, f.name)
            kind: type = type(f.default) if f.default is not MISSING else str
            if kind is bool:
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(f"DepthOptions.{f.name} must be a bool, got {value!r}")
                object.__setattr__(self, f.name, bool(value))
            elif kind is int:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
                    raise TypeError(f"DepthOptions.{f.name} must be an int, got {value!r}")
                object.__setattr__(self, f.name, int(value))
            elif kind is float:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                    raise TypeError(f"DepthOptions.{f.name} must be a number, got {value!r}")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"DepthOptions.{f.name} must be finite, got {value}")
                object.__setattr__(self, f.name, value)
            elif not isinstance(value, str):
                raise TypeError(f"DepthOptions.{f.name} must be a str, got {value!r}")
        self._check_ranges()

    def _check_ranges(self) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"DepthOptions: {message}")

        require(
            self.sources in SOURCE_CHOICES,
            f"sources must be one of {SOURCE_CHOICES}, got {self.sources!r}",
        )
        require(self.cts_window >= 0, f"cts_window must be >= 0, got {self.cts_window}")
        require(0 < self.x_lo < self.x_hi, f"need 0 < x_lo < x_hi, got {self.x_lo}, {self.x_hi}")
        require(self.r_lo < self.r_hi, f"need r_lo < r_hi, got {self.r_lo}, {self.r_hi}")
        require(self.min_pairs >= 1, f"min_pairs must be >= 1, got {self.min_pairs}")
        require(self.per_slice >= 1, f"per_slice must be >= 1, got {self.per_slice}")
        require(
            3 <= self.min_slices <= self.max_slices,
            f"need 3 <= min_slices <= max_slices, got {self.min_slices}, {self.max_slices}",
        )
        require(
            0 < self.hist_lo < self.hist_hi,
            f"need 0 < hist_lo < hist_hi, got {self.hist_lo}, {self.hist_hi}",
        )
        require(
            self.hist_bin > 0 and (self.hist_hi - self.hist_lo) / self.hist_bin >= 10,
            f"hist_bin must be > 0 and give at least 10 bins, got {self.hist_bin}",
        )
        require(
            self.window_lo_sigma > 0 and self.window_hi_sigma > 0,
            "window_lo_sigma and window_hi_sigma must be > 0",
        )
        require(self.window_iterations >= 1, "window_iterations must be >= 1")
        require(0 <= self.max_degree <= 2, f"max_degree must be 0, 1 or 2, got {self.max_degree}")
        require(0 < self.p_degree < 1, f"p_degree must be in (0, 1), got {self.p_degree}")
        require(self.gate_iterations >= 0, "gate_iterations must be >= 0")
        require(self.max_source_loss >= 0, "max_source_loss must be >= 0")
        require(self.consistency_min_diff >= 0, "consistency_min_diff must be >= 0")
        require(self.consistency_nsigma > 0, "consistency_nsigma must be > 0")
        require(self.consistency_min_slices >= 1, "consistency_min_slices must be >= 1")
        require(self.extrap_r_max > 0, "extrap_r_max must be > 0")
        require(
            0 < self.extrap_g_lo < 1 < self.extrap_g_hi,
            f"need 0 < extrap_g_lo < 1 < extrap_g_hi, got {self.extrap_g_lo}, {self.extrap_g_hi}",
        )
        require(
            self.coverage_r_lo < self.coverage_r_hi,
            "need coverage_r_lo < coverage_r_hi",
        )
        require(
            0.0 <= self.partial_cathode_frac <= 1.0,
            f"partial_cathode_frac must be in [0, 1], got {self.partial_cathode_frac}",
        )

    @property
    def source_ids(self) -> tuple[int, ...]:
        """The selected source ids (see :func:`selected_sources`)."""
        return selected_sources(self.sources)

    def is_default(self) -> bool:
        """Return True if every option has its default value."""
        return self == DepthOptions()

    def to_dict(self) -> dict[str, Any]:
        """Return the options as a plain dict in field-declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self) -> str:
        """Serialise the options to a compact JSON object.

        Keys appear in field-declaration order, so the same options always give
        the same string (it is stored in the sidecar as ``options_json``).
        """
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def changed_fields(self) -> dict[str, Any]:
        """Return the options that differ from the defaults (for logs and the census)."""
        defaults = DepthOptions()
        return {
            name: value
            for name, value in self.to_dict().items()
            if value != getattr(defaults, name)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, strict: bool = True) -> DepthOptions:
        """Build options from a dict such as the one ``to_dict`` returns.

        Missing keys take their default values, so options saved before a new
        option was added still load. Unknown keys usually mean a typo or options
        written by a newer dcalib: with ``strict`` (the default, for user input)
        they are an error; with ``strict=False`` (for stored results) they are
        ignored with a warning.

        Raises:
            ValueError: If ``data`` has unknown keys (``strict`` only) or a value
                is out of range.
            TypeError: If ``data`` is not a dict or a value has the wrong type.
        """
        if not isinstance(data, dict):
            raise TypeError(f"DepthOptions data must be a dict, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            if strict:
                raise ValueError(f"Unknown DepthOptions key(s): {unknown}")
            logger.warning("Ignoring unknown DepthOptions key(s): %s", unknown)
            data = {key: value for key, value in data.items() if key in known}
        return cls(**data)

    @classmethod
    def from_json(cls, text: str, *, strict: bool = True) -> DepthOptions:
        """Parse options from JSON produced by ``to_json`` (see :meth:`from_dict`).

        Raises:
            ValueError: If the text is not a JSON object, has unknown keys
                (``strict`` only) or an out-of-range value.
            TypeError: If a value has the wrong type.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid DepthOptions JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("DepthOptions JSON must be an object")
        return cls.from_dict(data, strict=strict)


QUADRATIC_ONLY_FIELDS: frozenset[str] = frozenset({"concave_only"})
"""Fields that only matter when a quadratic curve is allowed (``max_degree == 2``)."""

CONSISTENCY_FIELDS: frozenset[str] = frozenset(
    {"consistency_min_diff", "consistency_nsigma", "consistency_min_slices"}
)
"""Fields of the Ge/Cs consistency check, which only runs when both sources are fitted."""


def effective_options(options: DepthOptions) -> DepthOptions:
    """Return the options as the fit uses them.

    Fields that cannot influence the result are reset to their defaults:
    ``concave_only`` when ``max_degree < 2``, and the consistency thresholds
    when only one source is fitted. Two option sets therefore fit identically
    exactly when their effective options are equal (:func:`same_fit`).

    Example:
        >>> effective_options(DepthOptions(max_degree=1, concave_only=True)).concave_only
        False
    """
    defaults = DepthOptions()
    reset: dict[str, Any] = {}
    if options.max_degree < 2:
        reset.update({name: getattr(defaults, name) for name in QUADRATIC_ONLY_FIELDS})
    if options.sources != SOURCES_BOTH:
        reset.update({name: getattr(defaults, name) for name in CONSISTENCY_FIELDS})
    return replace(options, **reset) if reset else options


def same_fit(options: DepthOptions, other: DepthOptions) -> bool:
    """Return whether two option sets fit every anode identically.

    Example:
        >>> same_fit(DepthOptions(max_degree=1, concave_only=True), DepthOptions(max_degree=1))
        True
        >>> same_fit(DepthOptions(min_gain=0.02), DepthOptions())
        False
    """
    return effective_options(options) == effective_options(other)
