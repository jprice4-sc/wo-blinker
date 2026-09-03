"""Tests for two-state segmentation, debouncing and classification."""

from __future__ import annotations

import numpy as np

from blinker.params import AnalysisParams
from blinker.states import analyze_trace, dwell_times_from_state, segment


def test_dwell_times_drops_censored_edges() -> None:
    # Runs: ON(2), OFF(3), ON(1). Only the interior OFF(3) survives censoring.
    state = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    on, off = dwell_times_from_state(state, frame_interval_s=1.0)
    assert on.size == 0
    assert off.tolist() == [3.0]


def test_dwell_times_multiple_runs() -> None:
    # ON2 OFF2 ON3 OFF1 ON2 -> interior: OFF2, ON3, OFF1
    state = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    on, off = dwell_times_from_state(state, frame_interval_s=0.5)
    assert sorted(on.tolist()) == [1.5]
    assert sorted(off.tolist()) == [0.5, 1.0]


def test_segment_recovers_levels_and_separation() -> None:
    rng = np.random.default_rng(0)
    state = np.repeat([0, 1, 0, 1, 0, 1], 40).astype(bool)
    trace = np.where(state, 100.0, 10.0) + rng.normal(0, 2.0, state.size)
    fit = segment(trace)
    assert fit.off_level < fit.threshold < fit.on_level
    assert fit.amplitude > 80.0
    assert fit.contrast > 0.5
    assert fit.sep > 10.0


def test_flat_trace_is_static_not_blinker() -> None:
    rng = np.random.default_rng(1)
    trace = 50.0 + rng.normal(0, 3.0, 400)
    result = analyze_trace(trace, index=0, row=0, col=0, params=AnalysisParams())
    assert result.category == "static"
    assert not result.is_blinker


def test_clean_telegraph_is_stochastic() -> None:
    rng = np.random.default_rng(2)
    state = np.tile(np.repeat([0, 1], 25), 8).astype(bool)  # 16 clean switches
    trace = np.where(state, 500.0, 100.0) + rng.normal(0, 5.0, state.size)
    result = analyze_trace(trace, 0, 0, 0, AnalysisParams(min_transitions=4))
    assert result.category == "stochastic"
    assert result.is_blinker
    assert result.n_transitions >= 4


def test_debounce_merges_single_frame_flips() -> None:
    clean = np.repeat([0, 1, 0, 1], 30).astype(bool)
    trace = np.where(clean, 100.0, 0.0)
    trace[15] = 100.0  # a lone one-frame spike in an OFF run
    baseline = analyze_trace(trace, 0, 0, 0, AnalysisParams(debounce_frames=1))
    merged = analyze_trace(trace, 0, 0, 0, AnalysisParams(debounce_frames=5))
    assert merged.n_transitions < baseline.n_transitions
