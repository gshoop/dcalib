"""Tests of DepthOptions, the sources and the status/flag vocabulary."""

from __future__ import annotations

import numpy as np
import pytest

from dcalib import options as o
from dcalib.options import DepthOptions


class TestDepthOptions:
    def test_defaults_are_the_plan_values(self) -> None:
        d = DepthOptions()
        assert (d.x_lo, d.x_hi, d.r_lo, d.r_hi) == (0.75, 1.12, 0.0, 1.3)
        assert (d.min_pairs, d.per_slice, d.min_slices, d.max_slices) == (500, 400, 4, 12)
        assert (d.max_degree, d.p_degree, d.min_gain, d.max_source_loss) == (2, 0.01, 0.01, 0.01)
        assert (d.coverage_r_lo, d.coverage_r_hi) == (0.25, 0.95)
        assert d.cts_window == 48 and d.sources == "both" and not d.concave_only
        assert (d.window_lo_sigma, d.window_hi_sigma) == (2.0, 3.0)  # phase 3 retune
        assert d.is_default() and d.source_ids == (0, 1)

    def test_json_round_trip(self) -> None:
        opts = DepthOptions(sources="cs", max_degree=1, min_gain=0.02, concave_only=True)
        assert DepthOptions.from_json(opts.to_json()) == opts
        assert DepthOptions.from_json(DepthOptions().to_json()).is_default()
        assert opts.changed_fields() == {
            "sources": "cs",
            "max_degree": 1,
            "min_gain": 0.02,
            "concave_only": True,
        }

    def test_type_normalisation(self) -> None:
        opts = DepthOptions(x_lo=np.float32(0.7), min_pairs=np.int64(10), concave_only=np.True_)
        assert type(opts.x_lo) is float and type(opts.min_pairs) is int
        assert opts.concave_only is True
        assert DepthOptions(r_lo=0).r_lo == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_pairs": 1.5},
            {"min_pairs": True},
            {"x_lo": "0.7"},
            {"concave_only": 1},
            {"sources": 0},
        ],
    )
    def test_type_errors(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(TypeError):
            DepthOptions(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sources": "na22"},
            {"x_lo": 1.2},
            {"r_lo": 1.3},
            {"min_pairs": 0},
            {"min_slices": 2},
            {"min_slices": 13},
            {"hist_bin": 0.1},
            {"max_degree": 3},
            {"p_degree": 0.0},
            {"extrap_g_lo": 1.0},
            {"partial_cathode_frac": 1.5},
            {"x_hi": float("nan")},
            {"min_gain": float("inf")},
            {"cts_window": -1},
        ],
    )
    def test_range_errors(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            DepthOptions(**kwargs)  # type: ignore[arg-type]

    def test_from_dict_unknown_keys(self, caplog: pytest.LogCaptureFixture) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            DepthOptions.from_dict({"min_pairs": 10, "bogus": 1})
        opts = DepthOptions.from_dict({"min_pairs": 10, "bogus": 1}, strict=False)
        assert opts == DepthOptions(min_pairs=10)
        assert "bogus" in caplog.text
        with pytest.raises(ValueError, match="object"):
            DepthOptions.from_json("[1]")
        with pytest.raises(ValueError, match="Invalid"):
            DepthOptions.from_json("{")
        with pytest.raises(TypeError):
            DepthOptions.from_dict([])  # type: ignore[arg-type]

    def test_same_fit(self) -> None:
        assert o.same_fit(DepthOptions(max_degree=1, concave_only=True), DepthOptions(max_degree=1))
        assert not o.same_fit(DepthOptions(concave_only=True), DepthOptions())
        assert o.same_fit(
            DepthOptions(sources="ge", consistency_nsigma=5.0), DepthOptions(sources="ge")
        )
        assert not o.same_fit(DepthOptions(consistency_nsigma=5.0), DepthOptions())
        assert not o.same_fit(DepthOptions(min_gain=0.02), DepthOptions())
        assert o.effective_options(DepthOptions()) == DepthOptions()


class TestVocabulary:
    def test_sources(self) -> None:
        assert o.selected_sources("both") == (0, 1)
        assert o.selected_sources("ge") == (0,) and o.selected_sources("cs") == (1,)
        with pytest.raises(ValueError):
            o.selected_sources("x")
        assert o.SOURCE_ENERGY_KEV == {0: 511.0, 1: 662.0}

    def test_flags(self) -> None:
        flags = [o.FLAG_PARTIAL_CATHODE_COVERAGE, o.FLAG_SLICE_FIT_FAILED, o.FLAG_SLICE_FIT_FAILED]
        assert o.order_flags(flags) == (o.FLAG_SLICE_FIT_FAILED, o.FLAG_PARTIAL_CATHODE_COVERAGE)
        text = o.join_flags(flags)
        assert text == "slice_fit_failed;partial_cathode_coverage"
        assert o.split_flags(text) == o.order_flags(flags)
        assert o.split_flags("") == ()
        with pytest.raises(ValueError):
            o.order_flags(["nope"])

    def test_status_names(self) -> None:
        assert o.STATUSES[0] == "ok" and len(set(o.STATUSES)) == 7
        assert "no_calibrated_cathode" in o.STATUSES
        assert o.REVIEW_STATES == ("", "rejected")
