"""Demo: generate a synthetic blinking stack and run the full analysis.

Run from the repository root::

    uv run python examples/analyze_synthetic.py

It writes a synthetic TIFF plus the analysis outputs (emitter table, pooled
dwell times, JSON summary and figures) into ``examples/synthetic_out/``.
"""

from __future__ import annotations

from pathlib import Path

import tifffile

from blinker import AnalysisParams, analyze_file, save_results
from blinker.synthetic import SimParams, simulate_blinking_stack

OUTDIR = Path(__file__).parent / "synthetic_out"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # A 5x5 grid of blinkers plus a slow photo-activation transient.
    params = SimParams(
        height=128,
        width=128,
        n_frames=600,
        n_emitters=25,
        layout="grid",
        mean_on_frames=25.0,
        mean_off_frames=40.0,
        frame_interval_s=0.05,
        activation_amplitude=300.0,
        activation_tau_frames=80.0,
    )
    sim = simulate_blinking_stack(params, seed=42)

    tif_path = OUTDIR / "synthetic_stack.tif"
    tifffile.imwrite(str(tif_path), sim.data)
    print(f"wrote synthetic stack: {tif_path}  {sim.data.shape}")

    analysis = AnalysisParams(frame_interval_s=params.frame_interval_s, min_snr=3.0)
    result = analyze_file(tif_path, analysis)

    stats = result.stats
    print(f"detected {stats.n_spots} spots, {stats.n_blinkers} blinkers")
    print(f"ON  dwell median : {stats.on.summary.median:.3g} s  (true ~{params.mean_on_frames * 0.05:.2g} s)")
    print(f"OFF dwell median : {stats.off.summary.median:.3g} s  (true ~{params.mean_off_frames * 0.05:.2g} s)")
    if stats.transient is not None:
        print(f"transient tau    : {stats.transient.tau:.3g} s  (true ~{params.activation_tau_frames * 0.05:.2g} s)")

    written = save_results(result, OUTDIR)
    print(f"wrote {len(written)} output files to {OUTDIR}/")


if __name__ == "__main__":
    main()
