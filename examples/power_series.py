#!/usr/bin/env python3
"""Compare the SAME emitters across a power series (standalone example).

Point ``INPUT_DIR`` at a folder of TIFF stacks of the *same field of view*
acquired at increasing excitation power (power read from each filename, e.g.
``5%_...``, ``10%_...``). Emitters are detected ONCE on a combined projection
and the identical ROI positions are read in every stack, so emitter ``i`` is the
same physical emitter at every power. For each emitter the ON/OFF dwell times,
duty cycle and blink rate are tracked as power increases.

This uses only the public ``blinker`` API and does not modify the package.

Run:

    uv run python examples/power_series.py

Outputs (in ``OUTPUT_DIR``):
  * ``power_series.csv``  -- one row per (emitter, power)
  * ``power_trends.csv``  -- one row per power (median over blinkers)
  * ``power_trends.png``  -- ON/OFF dwell, duty cycle, blink rate vs power
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max

from blinker import (
    AnalysisParams,
    Spots,
    analyze_traces,
    extract_traces,
    find_stacks,
    load_stack,
    project,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from blinker import Stack, TraceResult
    from blinker.params import Projection, StateMethod

plt.switch_backend("Agg")

# ======================================================================
# CONFIG -- edit these
# ======================================================================

# --- paths ---
INPUT_DIR = Path(
    "/Users/jprice/Documents/Montara/PL_studies/10_8_26/Power_dependant_timelapse"
)  # folder of same-FOV stacks, one per power
OUTPUT_DIR = Path(
    "/Users/jprice/Documents/Montara/PL_studies/10_8_26/Power_dependant_results"
)  # results are written here

# --- time base ---
FRAME_INTERVAL_S = 0.000002  # seconds between frames (use 1/fps)

# --- spot detection (applied to a combined projection over all powers) ---
PROJECTION: Projection = "std"
DETECTION_PERCENTILE = 99.0
MIN_SEPARATION = 5
ROI_RADIUS = 3
BACKGROUND_INNER = 5
BACKGROUND_OUTER = 8
MAX_SPOTS: int | None = 2000

# --- two-state ON/OFF segmentation ---
STATE_METHOD: StateMethod = "gmm"
MIN_SNR = 3.0
MIN_TRANSITIONS = 2
MIN_DWELL_FRAMES = 1

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
)

_METRICS = ("on_median_s", "off_median_s", "duty_cycle", "blink_rate_hz")
_METRIC_LABELS = {
    "on_median_s": "ON dwell median (s)",
    "off_median_s": "OFF dwell median (s)",
    "duty_cycle": "duty cycle (fraction ON)",
    "blink_rate_hz": "blink rate (Hz)",
}
_POWER_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_POWER_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def power_from_filename(name: str) -> float:
    """Parse the excitation power from a filename (prefers a ``NN%`` token)."""
    match = _POWER_PCT.search(name) or _POWER_NUM.search(name)
    if match is None:
        msg = f"could not parse a power value from {name!r}"
        raise ValueError(msg)
    return float(match.group(1))


def detect_shared_spots(stacks: list[Stack], params: AnalysisParams) -> Spots:
    """Detect emitters once on the max of every stack's projection."""
    combined = np.maximum.reduce([project(stack, params.projection) for stack in stacks])
    threshold = float(np.percentile(combined, params.detection_percentile))
    num_peaks = params.max_spots if params.max_spots is not None else np.inf
    coords = peak_local_max(
        combined,
        min_distance=params.min_separation,
        threshold_abs=threshold,
        exclude_border=params.background_outer,
        num_peaks=num_peaks,
    )
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    return Spots(coords=coords, projection=combined, projection_kind=params.projection)


def _safe_median(values: NDArray[np.float64]) -> float:
    """Median of a possibly-empty array (``nan`` if empty)."""
    return float(np.median(values)) if values.size else float("nan")


def _safe_nanmedian(values: NDArray[np.float64]) -> float:
    """Median ignoring NaNs (``nan`` if everything is NaN)."""
    valid = values[~np.isnan(values)]
    return float(np.median(valid)) if valid.size else float("nan")


def _metric_value(result: TraceResult, key: str, duration_s: float) -> float:
    """Extract one scalar metric from a per-emitter result."""
    if key == "on_median_s":
        return _safe_median(result.on_times)
    if key == "off_median_s":
        return _safe_median(result.off_times)
    if key == "duty_cycle":
        return result.duty_cycle
    # blink_rate_hz: state switches per second.
    return result.n_transitions / duration_s if duration_s > 0 else float("nan")


def _metric_matrices(
    per_power: list[tuple[float, str, float, list[TraceResult]]],
    n_emitters: int,
) -> dict[str, NDArray[np.float64]]:
    """Build a ``(n_powers, n_emitters)`` matrix for each tracked metric."""
    n_powers = len(per_power)
    mats = {key: np.full((n_powers, n_emitters), np.nan) for key in _METRICS}
    for p, (_power, _name, duration, results) in enumerate(per_power):
        for e, result in enumerate(results):
            for key in _METRICS:
                mats[key][p, e] = _metric_value(result, key, duration)
    return mats


def _write_long_table(
    per_power: list[tuple[float, str, float, list[TraceResult]]],
    coords: NDArray[np.int64],
    path: Path,
) -> None:
    """Write one row per (emitter, power)."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "emitter",
                "row",
                "col",
                "power",
                "is_blinker",
                "on_median_s",
                "off_median_s",
                "duty_cycle",
                "blink_rate_hz",
                "n_transitions",
                "amplitude",
                "snr",
            ],
        )
        for power, _name, duration, results in per_power:
            for e, result in enumerate(results):
                row, col = coords[e]
                writer.writerow(
                    [
                        e,
                        int(row),
                        int(col),
                        power,
                        int(result.is_blinker),
                        _safe_median(result.on_times),
                        _safe_median(result.off_times),
                        result.duty_cycle,
                        result.n_transitions / duration if duration > 0 else float("nan"),
                        result.n_transitions,
                        result.fit.amplitude,
                        result.fit.snr,
                    ]
                )


def _write_trends_table(
    per_power: list[tuple[float, str, float, list[TraceResult]]],
    mats: dict[str, NDArray[np.float64]],
    path: Path,
) -> None:
    """Write one row per power: median over blinkers of each metric."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "power",
                "n_emitters",
                "n_blinkers",
                "blink_fraction",
                "on_median_s",
                "off_median_s",
                "duty_cycle",
                "blink_rate_hz",
            ],
        )
        for p, (power, _name, _duration, results) in enumerate(per_power):
            blinker_mask = np.array([r.is_blinker for r in results], dtype=bool)
            n_emitters = len(results)
            n_blinkers = int(blinker_mask.sum())
            fraction = n_blinkers / n_emitters if n_emitters else 0.0
            medians = [_safe_nanmedian(mats[key][p, blinker_mask]) for key in _METRICS]
            writer.writerow([power, n_emitters, n_blinkers, fraction, *medians])


def _plot_trends(powers: NDArray[np.float64], mats: dict[str, NDArray[np.float64]], path: Path) -> None:
    """Plot each metric vs power: faint per-emitter lines + bold median."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key in zip(axes.ravel(), _METRICS, strict=True):
        matrix = mats[key]
        for e in range(matrix.shape[1]):
            ax.plot(powers, matrix[:, e], color="0.8", lw=0.6)
        median = np.array([_safe_nanmedian(matrix[p]) for p in range(matrix.shape[0])])
        ax.plot(powers, median, "o-", color="crimson", lw=1.6, label="median over blinkers")
        ax.set_xlabel("power")
        ax.set_ylabel(_METRIC_LABELS[key])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Emitter behavior vs excitation power")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    """Run the power-series comparison over INPUT_DIR."""
    paths = find_stacks(INPUT_DIR)
    if not paths:
        print(f"no TIFF stacks found in {INPUT_DIR}/")
        return

    stacks = [load_stack(path, FRAME_INTERVAL_S) for path in paths]
    shapes = {(stack.height, stack.width) for stack in stacks}
    if len(shapes) != 1:
        msg = f"stacks have different frame sizes {shapes}; a shared-FOV power series needs one size"
        raise ValueError(msg)

    spots = detect_shared_spots(stacks, PARAMS)
    print(f"detected {spots.n} shared emitters across {len(stacks)} stacks")

    per_power: list[tuple[float, str, float, list[TraceResult]]] = []
    for path, stack in zip(paths, stacks, strict=True):
        power = power_from_filename(path.name)
        traceset = extract_traces(stack, spots, PARAMS)
        results = analyze_traces(traceset, PARAMS)
        per_power.append((power, path.stem, stack.duration_s, results))
        n_blinkers = sum(r.is_blinker for r in results)
        print(f"  {path.name:<32} power={power:<6g} {n_blinkers}/{spots.n} blinkers")
    per_power.sort(key=lambda item: item[0])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mats = _metric_matrices(per_power, spots.n)
    powers = np.array([power for power, _n, _d, _r in per_power], dtype=np.float64)
    _write_long_table(per_power, spots.coords, OUTPUT_DIR / "power_series.csv")
    _write_trends_table(per_power, mats, OUTPUT_DIR / "power_trends.csv")
    _plot_trends(powers, mats, OUTPUT_DIR / "power_trends.png")

    print("\npower   ON(s)   OFF(s)  duty   blink(Hz)")
    for p, (power, _name, _duration, _results) in enumerate(per_power):
        on = _safe_nanmedian(mats["on_median_s"][p])
        off = _safe_nanmedian(mats["off_median_s"][p])
        duty = _safe_nanmedian(mats["duty_cycle"][p])
        rate = _safe_nanmedian(mats["blink_rate_hz"][p])
        print(f"{power:<6g}  {on:<7.3g} {off:<7.3g} {duty:<6.3g} {rate:<.3g}")
    print(f"\nwrote power_series.csv, power_trends.csv, power_trends.png to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
