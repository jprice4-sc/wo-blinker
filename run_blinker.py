#!/usr/bin/env python3
"""Blinking analysis of one TIFF stack or a folder of them.

Edit the constants in the CONFIG block below, then run:

    uv run python run_blinker.py

``INPUT`` may be a single ``.tif`` / ``.tiff`` file or a folder of them. Each
stack is streamed from disk (multi-GB files are fine) and its results are
written to ``OUTPUT_DIR/<stack name>/``:

  * ``emitters.csv``   -- one row per emitter (position, levels, cv, contrast,
    separation, duty cycle, transitions, mean ON/OFF dwell, category, blinker
    flag) -- the *individual-emitter* statistics.
  * ``summary.json``   -- the *population* statistics (blinker counts, pooled
    ON/OFF dwell with power-law/exponential fits, contrast/duty-cycle/cv
    distributions) and the parameters used.
  * ``on_dwell_times.csv`` / ``off_dwell_times.csv`` -- pooled dwell times.
  * ``overview.png`` / ``statistics.png`` -- diagnostic figures.

For a folder, ``batch_summary.csv`` (one row per stack) and
``combined_summary.json`` (all stacks pooled) are also written at the top of
``OUTPUT_DIR``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from blinker import (
    AnalysisParams,
    analyze_file,
    combine_results,
    find_stacks,
    save_combined_summary,
    save_results,
)
from blinker.pipeline import _write_batch_summary  # one-row-per-stack table

if TYPE_CHECKING:
    from blinker.params import Projection

# ======================================================================
# CONFIG -- edit these
# ======================================================================

# --- paths ---
INPUT = Path("/Users/jprice/Downloads/20%")  # a .tif/.tiff file OR a folder of them
OUTPUT_DIR = Path("/Users/jprice/Downloads/20%/blinker_out")  # results written here
SAVE_FIGURES = True  # write overview.png / statistics.png per stack

# --- time base ---
FRAME_INTERVAL_S = 1.0  # seconds between frames (use 1/fps); 1.0 => dwell times in frames

# --- spot detection ---
PROJECTION: Projection = "mean"  # "mean" | "std" | "max" | "cv"
DETECTION_PERCENTILE = 60.0  # keep local maxima brighter than this percentile of the projection
MIN_SEPARATION = 6  # min pixels between two detected emitters
ROI_RADIUS = 2  # box half-width (pixels); 2 => a 5x5 summing box
MAX_SPOTS: int | None = None  # cap on emitters (brightest kept); None = no cap

# --- two-state ON/OFF segmentation & classification ---
DEBOUNCE_FRAMES = 5  # median-filter window removing single-frame flicker (1 = off)
SEP_THRESHOLD = 2.0  # min level separation (sigma) to call a trace two-state
CONTRAST_THRESHOLD = 0.1  # min Michelson contrast to call a trace two-state
MIN_TRANSITIONS = 4  # min switches for a two-state trace to count as a stochastic blinker

# --- dwell-time statistics ---
POWER_LAW_XMIN_S: float | None = None  # lower cutoff for the power-law fit; None = smallest dwell

# ======================================================================
# end CONFIG
# ======================================================================

PARAMS = AnalysisParams(
    frame_interval_s=FRAME_INTERVAL_S,
    projection=PROJECTION,
    detection_percentile=DETECTION_PERCENTILE,
    min_separation=MIN_SEPARATION,
    roi_radius=ROI_RADIUS,
    max_spots=MAX_SPOTS,
    debounce_frames=DEBOUNCE_FRAMES,
    sep_threshold=SEP_THRESHOLD,
    contrast_threshold=CONTRAST_THRESHOLD,
    min_transitions=MIN_TRANSITIONS,
    power_law_xmin_s=POWER_LAW_XMIN_S,
)


def main() -> None:
    """Analyze the configured file or folder and write all outputs."""
    stacks = [INPUT] if INPUT.is_file() else find_stacks(INPUT)
    if not stacks:
        print(f"no TIFF stacks found at {INPUT}")
        return

    print(f"analyzing {len(stacks)} stack(s)  ->  {OUTPUT_DIR}/")
    results = []
    for path in stacks:
        result = analyze_file(path, PARAMS)
        save_results(result, OUTPUT_DIR / path.stem, figures=SAVE_FIGURES)
        results.append(result)
        s = result.stats
        print(
            f"  {path.name:<32} {s.n_blinkers:4d}/{s.n_spots:<5d} blinkers"
            f"  ON~{s.on.summary.median:.3g}  OFF~{s.off.summary.median:.3g}  (median dwell)",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(results) > 1:
        _write_batch_summary(results, OUTPUT_DIR / "batch_summary.csv")
        combined = combine_results(results, PARAMS)
        save_combined_summary(combined, OUTPUT_DIR)
        print(f"\ncombined across {combined.n_stacks} stacks:")
        print(f"  blinkers      : {combined.n_blinkers}/{combined.n_spots} ({combined.blink_fraction:.1%})")
        print(f"  ON  dwell     : median {combined.on.summary.median:.3g}  mean {combined.on.summary.mean:.3g}")
        print(f"  OFF dwell     : median {combined.off.summary.median:.3g}  mean {combined.off.summary.mean:.3g}")
        if combined.on.power_law is not None:
            print(f"  ON power-law  : alpha {combined.on.power_law.alpha:.2f}")
    else:
        s = results[0].stats
        print(
            f"\n{results[0].stack_name}: {s.n_spots} spots -> "
            f"{s.n_static} static, {s.n_stepper} steppers, {s.n_blinkers} stochastic blinkers",
        )
        if s.on.power_law is not None:
            print(f"  ON power-law exponent : alpha {s.on.power_law.alpha:.2f}")
        if s.on.exponential is not None:
            print(f"  ON dwell tau          : {s.on.exponential.tau:.3g}")

    print(f"\noutputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
