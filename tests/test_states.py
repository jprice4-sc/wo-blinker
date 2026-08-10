"""Tests for two-state segmentation and dwell-time extraction."""

from __future__ import annotations

import numpy as np

from blinker.params import AnalysisParams
from blinker.states import analyze_trace, dwell_times_from_state, segment_two_state


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


def test_segment_two_state_recovers_levels() -> None:
    rng = np.random.default_rng(0)
    state = np.repeat([0, 1, 0, 1, 0, 1], 40).astype(bool)
    trace = np.where(state, 100.0, 10.0) + rng.normal(0, 2.0, state.size)
    fit = segment_two_state(trace, "gmm")
    assert fit.low_level < fit.threshold < fit.high_level
    assert fit.amplitude > 80.0
    assert fit.snr > 10.0


def test_flat_trace_is_not_a_blinker() -> None:
    rng = np.random.default_rng(1)
    trace = 50.0 + rng.normal(0, 3.0, 400)
    result = analyze_trace(trace, index=0, params=AnalysisParams(min_snr=3.0))
    assert not result.is_blinker


def test_min_dwell_merges_single_frame_flips() -> None:
    clean = np.repeat([0, 1, 0, 1], 30).astype(bool)
    trace = np.where(clean, 100.0, 0.0)
    trace[15] = 100.0  # a lone one-frame spike in an OFF run
    baseline = analyze_trace(trace, 0, AnalysisParams(min_dwell_frames=1, min_snr=1.0))
    merged = analyze_trace(trace, 0, AnalysisParams(min_dwell_frames=3, min_snr=1.0))
    assert merged.n_transitions < baseline.n_transitions
