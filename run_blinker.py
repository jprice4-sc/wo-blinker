#!/usr/bin/env python3
"""Batch blinking analysis of a folder of TIFF stacks, with a combined roll-up.

Edit the constants in the CONFIG block below, then run:

    uv run python run_blinker.py

Every ``*.tif`` / ``*.tiff`` in ``INPUT_DIR`` is analyzed with the same
parameters. For each stack, its results (emitter table, pooled dwell times,
JSON summary, figures) are written to ``OUTPUT_DIR/<stack name>/``. Two
folder-level files combine the key metrics across all stacks:

  * ``OUTPUT_DIR/batch_summary.csv``     -- one row per stack
  * ``OUTPUT_DIR/combined_summary.json`` -- every stack pooled together
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from blinker import (
    AnalysisParams,
    analyze_directory,
    combine_results,
    save_combined_summary,
)

if TYPE_CHECKING:
    from blinker import AnalysisResult
    from blinker.params import Projection, StateMethod

# ======================================================================
# CONFIG -- edit these
# ======================================================================

# --- paths ---
INPUT_DIR = Path(
    "/Users/jprice/Documents/Montara/PL_studies/10_8_26/Power_dependant_timelapse"
)  # folder of TIFF stacks (point at the folder, not one file)
OUTPUT_DIR = Path("/Users/jprice/Documents/Montara/PL_studies/10_8_26/Analysed traces")  # results are written here
SAVE_FIGURES = True  # write overview.png / statistics.png per stack

# --- time base ---
FRAME_INTERVAL_S = 0.000002  # seconds between frames (use 1/fps)

# --- spot detection ---
PROJECTION: Projection = "std"  # "std" | "max" | "mean" | "cv"
DETECTION_PERCENTILE = 99.0  # keep peaks brighter than this percentile of the projection
MIN_SEPARATION = 5  # min pixels between two detected emitters
ROI_RADIUS = 3  # aperture radius (pixels) summed per emitter
BACKGROUND_INNER = 5  # inner radius of the background annulus (pixels)
BACKGROUND_OUTER = 8  # outer radius of the background annulus (pixels)
MAX_SPOTS: int | None = None  # cap on emitters (brightest kept); None = no cap

# --- two-state ON/OFF segmentation ---
STATE_METHOD: StateMethod = "gmm"  # "gmm" | "kmeans" | "otsu"
MIN_SNR = 3.0  # amplitude/noise a trace must exceed to count as a blinker
MIN_TRANSITIONS = 2  # min ON/OFF switches to count as a blinker
MIN_DWELL_FRAMES = 1  # merge runs shorter than this (1 = no merging)

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
    background_inner=BACKGROUND_INNER,
    background_outer=BACKGROUND_OUTER,
    max_spots=MAX_SPOTS,
    state_method=STATE_METHOD,
    min_snr=MIN_SNR,
    min_transitions=MIN_TRANSITIONS,
    min_dwell_frames=MIN_DWELL_FRAMES,
    power_law_xmin_s=POWER_LAW_XMIN_S,
)


def _report(path: Path, result: AnalysisResult) -> None:
    """Print a one-line summary as each stack finishes."""
    stats = result.stats
    print(
        f"  {path.name:<32} {stats.n_blinkers:4d}/{stats.n_spots:<4d} blinkers"
        f"  ON~{stats.on.summary.median:.3g}s  OFF~{stats.off.summary.median:.3g}s",
    )


def main() -> None:
    """Analyze every stack in INPUT_DIR and combine their key metrics."""
    print(f"analyzing {INPUT_DIR}/  ->  {OUTPUT_DIR}/")
    results = analyze_directory(
        INPUT_DIR,
        OUTPUT_DIR,
        PARAMS,
        figures=SAVE_FIGURES,
        on_result=_report,
    )
    if not results:
        print(f"no TIFF stacks found in {INPUT_DIR}/")
        return

    combined = combine_results(results, PARAMS)
    save_combined_summary(combined, OUTPUT_DIR)

    print(f"\ncombined across {combined.n_stacks} stacks:")
    print(f"  blinkers      : {combined.n_blinkers}/{combined.n_spots} ({combined.blink_fraction:.1%})")
    print(f"  ON  dwell (s) : median {combined.on.summary.median:.3g}  mean {combined.on.summary.mean:.3g}")
    print(f"  OFF dwell (s) : median {combined.off.summary.median:.3g}  mean {combined.off.summary.mean:.3g}")
    print(f"  amplitude     : median {combined.amplitude.median:.4g} counts")
    print(f"  duty cycle    : median {combined.duty_cycle.median:.3g}")
    if combined.on.power_law is not None:
        print(f"  ON power-law  : alpha {combined.on.power_law.alpha:.2f}")
    print(f"\nper-stack table  -> {OUTPUT_DIR / 'batch_summary.csv'}")
    print(f"combined roll-up -> {OUTPUT_DIR / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
