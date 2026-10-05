#!/usr/bin/env python3
"""Blinking analysis across an illumination-power series, with trend plots.

Point ``FOLDER`` at a directory of same-field-of-view stacks acquired at
different illumination powers, with the power encoded as a leading number in the
filename (e.g. ``5%_...``, ``10%_...``, ``50%_...``). The script:

1. detects the emitter sites **once** on a single reference stack (highest
   power by default) so the *same* ROIs are measured at every power;
2. streams every stack, extracts those fixed ROIs, and runs the full blinking
   analysis (per-emitter + population);
3. writes ``power_series.csv`` (one row per power) and ``power_trends.png``
   (the key population statistics versus power), and optionally the full
   per-stack outputs under ``OUTPUT_DIR/<power>pct/``.

Run from the repository root::

    uv run python examples/power_series.py

It uses only the public :mod:`blinker` API and does not modify the package.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blinker import (
    AnalysisParams,
    AnalysisResult,
    aggregate,
    analyze_traces,
    detect_spots,
    extract_traces,
    open_stack,
    save_results,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from blinker import AggregateStats, EmitterResult

plt.switch_backend("Agg")

# ======================================================================
# CONFIG -- edit these
# ======================================================================

FOLDER = Path("/Users/jprice/Documents/Montara/PL_studies/10_8_26/Power_dependant_timelapse")
OUTPUT_DIR = FOLDER / "blinker_power_series"
SAVE_PER_STACK = True  # also write emitters.csv/summary.json/figures per power

FRAME_INTERVAL_S = 1.0  # seconds per frame (1.0 => dwell times in frames; use 1/fps for seconds)

PARAMS = AnalysisParams(
    frame_interval_s=FRAME_INTERVAL_S,
    # "mean" samples the whole persistent array; use "std" (with a high
    # detection_percentile, e.g. 99) to instead target only the fluctuating sites.
    projection="mean",
    detection_percentile=60.0,
    min_separation=6,
    roi_radius=2,  # 5x5 box
    debounce_frames=5,
    sep_threshold=2.0,
    contrast_threshold=0.1,
    min_transitions=4,
)

# ======================================================================
# end CONFIG
# ======================================================================

_POWER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*%")

# The per-power statistics tracked across the series (csv columns + plot data).
_COLUMNS = (
    "power_pct",
    "n_frames",
    "n_spots",
    "n_blinkers",
    "blink_fraction",
    "on_median",
    "on_mean",
    "off_median",
    "off_mean",
    "duty_median",
    "contrast_median",
    "cv_median",
    "amplitude_median",
    "mean_transitions_per_blinker",
    "on_powerlaw_alpha",
    "off_powerlaw_alpha",
)


def _power_of(path: Path) -> float | None:
    """Parse the leading ``<number>%`` power from a filename, or ``None``."""
    match = _POWER_RE.match(path.name)
    return float(match.group(1)) if match else None


def _find_power_stacks(folder: Path) -> list[tuple[float, Path]]:
    """Return ``(power, path)`` for every power-tagged TIFF, sorted by power."""
    found: dict[float, Path] = {}
    for path in folder.iterdir():
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        power = _power_of(path)
        if power is not None:
            found[power] = path
    return sorted(found.items())


def _row_for(
    power: float,
    n_frames: int,
    emitters: list[EmitterResult],
    stats: AggregateStats,
) -> dict[str, object]:
    """Assemble one CSV/plot row from a stack's per-emitter + aggregate results."""
    blink_tr = [r.n_transitions for r in emitters if r.is_blinker]
    return {
        "power_pct": power,
        "n_frames": n_frames,
        "n_spots": stats.n_spots,
        "n_blinkers": stats.n_blinkers,
        "blink_fraction": stats.blink_fraction,
        "on_median": stats.on.summary.median,
        "on_mean": stats.on.summary.mean,
        "off_median": stats.off.summary.median,
        "off_mean": stats.off.summary.mean,
        "duty_median": stats.duty_cycle.median,
        "contrast_median": stats.contrast.median,
        "cv_median": stats.cv.median,
        "amplitude_median": stats.amplitude.median,
        "mean_transitions_per_blinker": float(np.mean(blink_tr)) if blink_tr else float("nan"),
        "on_powerlaw_alpha": stats.on.power_law.alpha if stats.on.power_law else float("nan"),
        "off_powerlaw_alpha": stats.off.power_law.alpha if stats.off.power_law else float("nan"),
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the per-power statistics table."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trends(rows: list[dict[str, object]], unit: str, path: Path) -> None:
    """Plot the key population statistics versus illumination power."""
    p = np.array([r["power_pct"] for r in rows], dtype=float)

    def col(name: str) -> NDArray[np.float64]:
        return np.array([r[name] for r in rows], dtype=float)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (1) blinker count + blinker fraction
    a = ax[0, 0]
    a.plot(p, col("n_blinkers"), "o-", color="tab:red", label="n blinkers")
    a.set_ylabel("n stochastic blinkers", color="tab:red")
    a.tick_params(axis="y", labelcolor="tab:red")
    b = a.twinx()
    b.plot(p, 100 * col("blink_fraction"), "s--", color="0.4", label="fraction")
    b.set_ylabel("blinker fraction (%)", color="0.4")
    a.set_title("Blinker count & fraction")

    # (2) ON / OFF median dwell
    a = ax[0, 1]
    a.plot(p, col("on_median"), "o-", color="tab:red", label="ON")
    a.plot(p, col("off_median"), "s-", color="tab:blue", label="OFF")
    a.set_ylabel(f"median dwell ({unit})")
    a.set_title("Median ON/OFF dwell")
    a.legend()

    # (3) duty cycle
    a = ax[0, 2]
    a.plot(p, col("duty_median"), "o-", color="tab:green")
    a.set_ylabel("median duty cycle (fraction ON)")
    a.set_title("Duty cycle")

    # (4) contrast
    a = ax[1, 0]
    a.plot(p, col("contrast_median"), "o-", color="tab:purple")
    a.set_ylabel("median ON/OFF contrast")
    a.set_title("Blink contrast")

    # (5) switching activity
    a = ax[1, 1]
    a.plot(p, col("mean_transitions_per_blinker"), "o-", color="tab:orange")
    a.set_ylabel("mean transitions / blinker")
    a.set_title("Switching activity")

    # (6) dwell-time power-law exponents
    a = ax[1, 2]
    a.plot(p, col("on_powerlaw_alpha"), "o-", color="tab:red", label="ON")
    a.plot(p, col("off_powerlaw_alpha"), "s-", color="tab:blue", label="OFF")
    a.set_ylabel("dwell power-law alpha")
    a.set_title("Dwell-time power-law exponent")
    a.legend()

    for a in ax.ravel():
        a.set_xlabel("illumination power (%)")
        a.grid(alpha=0.3)
    fig.suptitle(f"Blinking statistics vs illumination power  ({FOLDER.name})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    """Run the power series and write the CSV + trend plot."""
    stacks = _find_power_stacks(FOLDER)
    if not stacks:
        print(f"no power-tagged TIFF stacks (e.g. '10%_...') found in {FOLDER}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unit = "s" if FRAME_INTERVAL_S != 1.0 else "frames"

    # Detect emitter sites once on the highest-power stack, then reuse those exact
    # ROIs at every power so we track the same physical sites.
    ref_power, ref_path = max(stacks, key=lambda pp: pp[0])
    ref_source = open_stack(ref_path, FRAME_INTERVAL_S)
    print(f"detecting emitters on reference stack {ref_power:g}% ({ref_path.name}) ...")
    spots = detect_spots(ref_source, PARAMS)
    print(f"  {spots.n} emitter sites; reusing them at every power")

    rows: list[dict[str, object]] = []
    for power, path in stacks:
        source = open_stack(path, FRAME_INTERVAL_S)
        if (source.height, source.width) != (ref_source.height, ref_source.width):
            print(f"  SKIP {path.name}: frame size {source.height}x{source.width} != reference; ROIs would not align")
            continue
        traces = extract_traces(source, spots, PARAMS)
        emitters = analyze_traces(traces, PARAMS)
        stats = aggregate(emitters, PARAMS)
        rows.append(_row_for(power, source.n_frames, emitters, stats))
        print(
            f"  {power:5g}%  {stats.n_blinkers:4d}/{stats.n_spots} blinkers  "
            f"ON~{stats.on.summary.median:.3g}  OFF~{stats.off.summary.median:.3g}  ({unit})"
        )
        if SAVE_PER_STACK:
            result = AnalysisResult(
                stack_name=f"{power:g}pct",
                params=PARAMS,
                spots=spots,
                traces=traces,
                emitters=emitters,
                stats=stats,
            )
            save_results(result, OUTPUT_DIR / f"{power:g}pct", figures=True)

    csv_path = OUTPUT_DIR / "power_series.csv"
    plot_path = OUTPUT_DIR / "power_trends.png"
    _write_csv(rows, csv_path)
    _plot_trends(rows, unit, plot_path)
    print(f"\nwrote {csv_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
