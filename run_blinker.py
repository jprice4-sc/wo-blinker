#!/usr/bin/env python3
"""Blinking analysis of a single TIFF stack.

Edit the constants in the CONFIG block below, then run:

    uv run python run_blinker.py

The stack at ``INPUT_FILE`` is analysed, and its results (emitter table,
pooled dwell times, JSON summary, figures) are written to ``OUTPUT_DIR/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from blinker import AnalysisParams, analyze_file, save_results

if TYPE_CHECKING:
    from blinker import AnalysisResult
    from blinker.params import Projection, StateMethod

# ======================================================================
# CONFIG -- edit these
# ======================================================================

# --- paths ---
INPUT_FILE = Path("/Users/jprice/Downloads/5%_1_MMStack_Default.ome.tif")  # the TIFF stack to analyse
OUTPUT_DIR = Path("blinker_out_5%")  # results are written here
SAVE_FIGURES = True  # write overview.png / statistics.png

# --- time base ---
FRAME_INTERVAL_S = 0.000002  # seconds between frames (use 1/fps)

# --- spot detection ---
PROJECTION: Projection = "std"  # "std" | "max" | "mean" | "cv"
DETECTION_PERCENTILE = 99.0  # keep peaks brighter than this percentile of the projection
MIN_SEPARATION = 5  # min pixels between two detected emitters
ROI_RADIUS = 3  # aperture radius (pixels) summed per emitter
BACKGROUND_INNER = 5  # inner radius of the background annulus (pixels)
BACKGROUND_OUTER = 8  # outer radius of the background annulus (pixels)
MAX_SPOTS: int | None = 2000  # cap on emitters (brightest kept); None = no cap

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
    """Print a one-line summary when the stack finishes."""
    stats = result.stats
    print(
        f"  {path.name:<30} {stats.n_blinkers:4d}/{stats.n_spots:<4d} blinkers"
        f"  ON~{stats.on.summary.median:.3g}s  OFF~{stats.off.summary.median:.3g}s",
    )


def main() -> None:
    """Analyse INPUT_FILE and write results to OUTPUT_DIR."""
    print(f"analysing {INPUT_FILE}  ->  {OUTPUT_DIR}/")
    result = analyze_file(INPUT_FILE, PARAMS)
    save_results(result, OUTPUT_DIR, figures=SAVE_FIGURES)
    _report(INPUT_FILE, result)
    print(f"done: results -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
