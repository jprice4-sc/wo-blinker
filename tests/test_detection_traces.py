"""Tests for spot detection and trace extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blinker.detection import detect_spots, project
from blinker.params import AnalysisParams
from blinker.traces import extract_traces

if TYPE_CHECKING:
    from blinker.loading import Stack
    from blinker.synthetic import SimulatedStack


def test_projection_shapes(sim_stack: Stack) -> None:
    for kind in ("std", "max", "mean", "cv"):
        proj = project(sim_stack, kind)
        assert proj.shape == (sim_stack.height, sim_stack.width)


def test_detects_all_grid_emitters(sim_stack: Stack, sim: SimulatedStack) -> None:
    spots = detect_spots(sim_stack, AnalysisParams())
    assert spots.n == sim.positions.shape[0]
    # Every detected spot sits on a true emitter (grid is well separated).
    for row, col in spots.coords:
        dist = np.sqrt(((sim.positions - [row, col]) ** 2).sum(axis=1)).min()
        assert dist <= 1.5


def test_trace_amplitude_matches_ground_truth(sim_stack: Stack) -> None:
    params = AnalysisParams()
    spots = detect_spots(sim_stack, params)
    traces = extract_traces(sim_stack, spots, params)
    assert traces.n_spots == spots.n
    assert traces.n_frames == sim_stack.n_frames
    # Every emitter has OFF frames, so the per-trace minimum is the background-
    # subtracted OFF baseline and should sit near zero; the maximum is the ON
    # signal and should be well above the noise floor.
    lo = traces.traces.min(axis=1)
    hi = traces.traces.max(axis=1)
    assert np.all(np.abs(lo) < 0.2 * hi)
    assert np.all(hi > 5000.0)
