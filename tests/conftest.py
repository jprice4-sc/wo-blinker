"""Shared fixtures: synthetic stacks with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from blinker.loading import Stack, stack_from_array
from blinker.params import AnalysisParams
from blinker.synthetic import SimParams, SimulatedStack, simulate_blinking_stack


@pytest.fixture
def sim() -> SimulatedStack:
    """A clean, well-separated 3x3 grid of blinkers with a known state history."""
    params = SimParams(
        height=96,
        width=96,
        n_frames=500,
        n_emitters=9,
        layout="grid",
        mean_on_frames=20.0,
        mean_off_frames=30.0,
        frame_interval_s=0.05,
    )
    return simulate_blinking_stack(params, seed=3)


@pytest.fixture
def sim_stack(sim: SimulatedStack) -> Stack:
    """The synthetic grid wrapped as a :class:`~blinker.loading.Stack`."""
    return stack_from_array(sim.data, frame_interval_s=sim.frame_interval_s, name="sim")


@pytest.fixture
def sim_params(sim: SimulatedStack) -> AnalysisParams:
    """Analysis parameters tuned for the sparse synthetic grid.

    The synthetic emitters are sparse and bright, so detection uses the ``std``
    projection at a high percentile; ``debounce_frames=1`` leaves the ON/OFF
    state untouched so recovered transition counts match ground truth exactly.
    """
    return AnalysisParams(
        frame_interval_s=sim.frame_interval_s,
        projection="std",
        detection_percentile=99.0,
        min_separation=8,
        roi_radius=3,
        debounce_frames=1,
        min_transitions=2,
    )


@pytest.fixture
def flat_stack() -> Stack:
    """A featureless stack: constant background plus read noise, no emitters."""
    rng = np.random.default_rng(0)
    data = 100.0 + rng.normal(0.0, 5.0, size=(200, 48, 48))
    return stack_from_array(np.clip(data, 0, None).astype(np.float32), frame_interval_s=0.05, name="flat")
