"""End-to-end pipeline and batch tests against synthetic ground truth."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import tifffile

from blinker.params import AnalysisParams
from blinker.pipeline import (
    analyze_directory,
    analyze_stack,
    combine_results,
    save_combined_summary,
    save_results,
)

if TYPE_CHECKING:
    from pathlib import Path

    from blinker.loading import Stack
    from blinker.synthetic import SimulatedStack


def _true_transitions(states: np.ndarray) -> list[int]:
    return [int((np.diff(states[e].astype(int)) != 0).sum()) for e in range(states.shape[0])]


def test_end_to_end_recovers_ground_truth(sim_stack: Stack, sim: SimulatedStack) -> None:
    result = analyze_stack(sim_stack, AnalysisParams(frame_interval_s=sim.frame_interval_s, min_snr=3.0))
    assert result.stats.n_spots == sim.positions.shape[0]
    assert result.stats.n_blinkers == sim.positions.shape[0]

    true_trans = _true_transitions(sim.states)
    for r in result.trace_results:
        row, col = result.traces.coords[r.index]
        emitter = int(np.argmin(((sim.positions - [row, col]) ** 2).sum(axis=1)))
        assert r.n_transitions == true_trans[emitter]

    # Recovered dwell statistics should respect the simulated ON < OFF ordering.
    assert result.stats.on.summary.median < result.stats.off.summary.median
    assert result.stats.on.summary.median > 0


def test_dwell_medians_in_ballpark(sim_stack: Stack, sim: SimulatedStack) -> None:
    result = analyze_stack(sim_stack, AnalysisParams(frame_interval_s=sim.frame_interval_s))
    dt = sim.frame_interval_s
    # Geometric dwell medians for the simulated rates, in seconds.
    on_median_true = np.log(0.5) / np.log(1 - 1 / sim.params.mean_on_frames) * dt
    off_median_true = np.log(0.5) / np.log(1 - 1 / sim.params.mean_off_frames) * dt
    assert 0.4 * on_median_true < result.stats.on.summary.median < 2.5 * on_median_true
    assert 0.4 * off_median_true < result.stats.off.summary.median < 2.5 * off_median_true


def test_save_results_writes_expected_files(sim_stack: Stack, tmp_path: Path) -> None:
    result = analyze_stack(sim_stack, AnalysisParams())
    written = save_results(result, tmp_path, figures=True)
    names = {p.name for p in written}
    assert {"emitters.csv", "on_dwell_times.csv", "off_dwell_times.csv", "summary.json"} <= names
    assert (tmp_path / "overview.png").is_file()
    assert (tmp_path / "statistics.png").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert isinstance(summary, dict)
    stats = summary["stats"]
    params = summary["params"]
    assert isinstance(stats, dict)
    assert isinstance(params, dict)
    assert stats["n_spots"] == result.stats.n_spots
    assert params["state_method"] == "gmm"


def test_analyze_directory_batch(sim: SimulatedStack, tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("stack_a.tif", "stack_b.tif"):
        tifffile.imwrite(str(input_dir / name), sim.data.astype(np.float32))

    seen: list[str] = []
    results = analyze_directory(
        input_dir,
        output_dir,
        AnalysisParams(frame_interval_s=sim.frame_interval_s),
        figures=False,
        on_result=lambda path, _result: seen.append(path.name),
    )

    assert len(results) == 2
    assert sorted(seen) == ["stack_a.tif", "stack_b.tif"]
    for stem in ("stack_a", "stack_b"):
        assert (output_dir / stem / "emitters.csv").is_file()
        assert not (output_dir / stem / "overview.png").exists()

    summary_lines = (output_dir / "batch_summary.csv").read_text().strip().splitlines()
    assert summary_lines[0].startswith("stack,")
    assert len(summary_lines) == 3  # header + one row per stack


def test_analyze_directory_empty(tmp_path: Path) -> None:
    results = analyze_directory(tmp_path, tmp_path / "out", AnalysisParams())
    assert results == []
    assert (tmp_path / "out" / "batch_summary.csv").is_file()


def test_combine_results_pools_across_stacks(sim_stack: Stack, tmp_path: Path) -> None:
    params = AnalysisParams(frame_interval_s=0.05)
    results = [analyze_stack(sim_stack, params), analyze_stack(sim_stack, params)]
    combined = combine_results(results, params)

    assert combined.n_stacks == 2
    assert combined.n_spots == sum(r.stats.n_spots for r in results)
    assert combined.n_blinkers == sum(r.stats.n_blinkers for r in results)
    # Pooling two identical stacks doubles the dwell-time sample count.
    assert combined.on.summary.n == 2 * results[0].stats.on.summary.n

    path = save_combined_summary(combined, tmp_path)
    assert path.name == "combined_summary.json"
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    assert payload["n_stacks"] == 2
