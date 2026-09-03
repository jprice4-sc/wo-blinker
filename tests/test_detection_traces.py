"""Tests for spot detection and trace extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tifffile

from blinker.detection import detect_spots, project
from blinker.loading import open_stack, stack_from_array
from blinker.traces import extract_traces

if TYPE_CHECKING:
    from pathlib import Path

    from blinker.loading import Stack
    from blinker.params import AnalysisParams
    from blinker.synthetic import SimulatedStack


def test_projection_shapes(sim_stack: Stack) -> None:
    for kind in ("mean", "std", "max", "cv"):
        proj = project(sim_stack, kind)
        assert proj.shape == (sim_stack.height, sim_stack.width)


def test_detects_all_grid_emitters(sim_stack: Stack, sim: SimulatedStack, sim_params: AnalysisParams) -> None:
    spots = detect_spots(sim_stack, sim_params)
    assert spots.n == sim.positions.shape[0]
    # Every detected spot sits on a true emitter (grid is well separated).
    for row, col in spots.coords:
        dist = np.sqrt(((sim.positions - [row, col]) ** 2).sum(axis=1)).min()
        assert dist <= 1.5


def test_box_trace_recovers_on_off(sim_stack: Stack, sim_params: AnalysisParams) -> None:
    spots = detect_spots(sim_stack, sim_params)
    traces = extract_traces(sim_stack, spots, sim_params)
    assert traces.n_spots == spots.n
    assert traces.n_frames == sim_stack.n_frames
    assert traces.roi_area == (2 * sim_params.roi_radius + 1) ** 2
    # Each emitter blinks, so per-trace max (ON) sits well above min (OFF).
    lo = traces.traces.min(axis=1)
    hi = traces.traces.max(axis=1)
    assert np.all(hi > 1.5 * lo)


def test_streaming_and_memory_traces_match(
    sim: SimulatedStack,
    sim_params: AnalysisParams,
    tmp_path: Path,
) -> None:
    """Traces from a streamed TiffStack match those from the in-memory Stack."""
    path = tmp_path / "movie.tif"
    tifffile.imwrite(str(path), sim.data.astype(np.float32))

    mem = stack_from_array(sim.data, frame_interval_s=sim.frame_interval_s)
    streamed = open_stack(path, sim.frame_interval_s)

    spots_mem = detect_spots(mem, sim_params)
    spots_str = detect_spots(streamed, sim_params)
    np.testing.assert_array_equal(spots_mem.coords, spots_str.coords)

    t_mem = extract_traces(mem, spots_mem, sim_params).traces
    t_str = extract_traces(streamed, spots_str, sim_params).traces
    np.testing.assert_allclose(t_mem, t_str)
