"""Tests for the object-oriented Blinker facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from blinker import AnalysisParams, Blinker

if TYPE_CHECKING:
    from pathlib import Path

    from blinker.synthetic import SimulatedStack


def test_workflow_matches_functional_api(sim: SimulatedStack, sim_params: AnalysisParams) -> None:
    b = Blinker(sim_params)
    b.load_array(sim.data, name="sim").analyze()

    assert b.n_spots == sim.positions.shape[0]
    assert b.n_blinkers == sim.positions.shape[0]
    on, off = b.dwell_times()
    assert on.size > 0
    assert off.size > 0
    assert on.size == b.on_times.size
    assert "blinkers" in b.summary()


def test_overrides_reach_params() -> None:
    b = Blinker(AnalysisParams(sep_threshold=1.0), frame_interval_s=0.02, min_transitions=6)
    assert b.params.frame_interval_s == 0.02
    assert b.params.min_transitions == 6
    assert b.params.sep_threshold == 1.0


def test_analysis_is_lazy_and_cached(sim: SimulatedStack) -> None:
    b = Blinker(frame_interval_s=sim.frame_interval_s)
    b.load_array(sim.data)
    assert b.result is None
    assert b.stats.n_spots > 0  # first access triggers analysis
    assert b.result is not None  # now cached on the instance


def test_load_clears_previous_result(sim: SimulatedStack) -> None:
    b = Blinker(frame_interval_s=sim.frame_interval_s).load_array(sim.data)
    b.analyze()
    assert b.result is not None
    b.load_array(sim.data, name="again")
    assert b.result is None


def test_analyze_without_stack_raises() -> None:
    with pytest.raises(RuntimeError, match="no stack"):
        Blinker().analyze()


def test_plots_and_save(sim: SimulatedStack, tmp_path: Path) -> None:
    b = Blinker(frame_interval_s=sim.frame_interval_s).load_array(sim.data)
    for fig in (b.plot_overview(), b.plot_statistics(), b.plot_summary(), b.plot_trace(0)):
        assert fig is not None
        plt.close(fig)

    written = b.save(tmp_path, figures=True)
    assert (tmp_path / "summary.png").is_file()
    assert (tmp_path / "emitters.csv").is_file()
    assert any(p.name == "summary.png" for p in written)


def test_plot_trace_out_of_range(sim: SimulatedStack) -> None:
    b = Blinker(frame_interval_s=sim.frame_interval_s).load_array(sim.data)
    with pytest.raises(IndexError):
        b.plot_trace(10_000)
