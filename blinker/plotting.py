"""Figure generation for a completed analysis.

Two figures summarize a run:

* *overview* -- the detection image with emitters marked by class, the strongest
  blinker traces with their ON/OFF thresholds, and histograms of the per-spot
  fluctuation and switching activity.
* *statistics* -- the pooled ON/OFF dwell-time distribution (log-binned) and its
  survival curve on log-log axes with the exponential and power-law fits, which
  is where heavy-tailed (power-law) blinking kinetics show up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .pipeline import AnalysisResult
    from .states import EmitterResult
    from .stats import DwellStats

plt.switch_backend("Agg")

_MAX_EXAMPLE_TRACES = 6
_HIST_BINS = 30
_LOG_BINS = 35


def _blinkers(result: AnalysisResult) -> list[EmitterResult]:
    """Stochastic blinkers, strongest (best-separated) first."""
    blinkers = [r for r in result.emitters if r.is_blinker]
    blinkers.sort(key=lambda r: r.fit.sep, reverse=True)
    return blinkers


def _plot_detection(ax: Axes, result: AnalysisResult) -> None:
    """Draw the detection image with emitters marked by class."""
    proj = result.spots.projection
    ax.imshow(proj, cmap="inferno", vmax=float(np.percentile(proj, 99.5)))
    groups = {
        "static": ("white", 4, 0.4),
        "stepper": ("deepskyblue", 6, 1.0),
        "stochastic": ("red", 7, 1.3),
    }
    for category, (color, size, lw) in groups.items():
        pts = np.array([(r.col, r.row) for r in result.emitters if r.category == category])
        if pts.size:
            ax.plot(pts[:, 0], pts[:, 1], "o", mfc="none", mec=color, ms=size, mew=lw, label=f"{category} ({len(pts)})")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.set_title(f"{result.spots.projection_kind} projection: {result.stats.n_blinkers} blinkers")
    ax.axis("off")


def _plot_example_traces(ax: Axes, result: AnalysisResult) -> None:
    """Draw the strongest blinker traces with their fitted ON/OFF thresholds."""
    t = np.arange(result.traces.n_frames) * result.params.frame_interval_s
    for r in _blinkers(result)[:_MAX_EXAMPLE_TRACES]:
        (line,) = ax.plot(t, result.traces.traces[r.index], lw=0.6)
        ax.axhline(r.fit.threshold, color=line.get_color(), ls=":", lw=0.6, alpha=0.6)
    ax.set_xlabel(f"time ({'s' if result.params.frame_interval_s != 1.0 else 'frames'})")
    ax.set_ylabel("intensity (counts)")
    ax.set_title("Strongest blinker traces")


def _plot_cv(ax: Axes, result: AnalysisResult) -> None:
    """Histogram of per-spot relative fluctuation (all detected emitters)."""
    cv = np.array([r.cv for r in result.emitters], dtype=np.float64)
    if cv.size:
        ax.hist(cv, bins=_HIST_BINS, color="0.5")
    ax.set_xlabel("relative fluctuation std/mean")
    ax.set_title("Per-spot fluctuation (all emitters)")


def _plot_transitions(ax: Axes, result: AnalysisResult) -> None:
    """Histogram of switching activity across blinkers."""
    ntr = np.array([r.n_transitions for r in result.emitters if r.is_blinker], dtype=np.float64)
    if ntr.size:
        ax.hist(ntr, bins=_HIST_BINS, color="tab:red", alpha=0.8)
    ax.set_xlabel("transitions per blinker")
    ax.set_title("Switching activity (blinkers)")


def make_overview_figure(result: AnalysisResult) -> Figure:
    """Build the overview figure.

    Args:
        result: A completed analysis.

    Returns:
        The Matplotlib figure (caller is responsible for closing it).
    """
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    _plot_detection(ax[0, 0], result)
    _plot_example_traces(ax[0, 1], result)
    _plot_cv(ax[1, 0], result)
    _plot_transitions(ax[1, 1], result)
    fig.tight_layout()
    return fig


def _pooled_dwells(result: AnalysisResult) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool ON and OFF dwell times across all blinkers."""
    blinkers = [r for r in result.emitters if r.is_blinker]
    on = np.concatenate([r.on_times for r in blinkers]) if blinkers else np.empty(0)
    off = np.concatenate([r.off_times for r in blinkers]) if blinkers else np.empty(0)
    return on, off


def _plot_dwell_hist(ax: Axes, on: NDArray[np.float64], off: NDArray[np.float64], unit: str) -> None:
    """Log-binned ON/OFF dwell-time histograms."""
    hi = max(float(on.max(initial=1.0)), float(off.max(initial=1.0)))
    lo = min(float(on[on > 0].min(initial=1.0)), float(off[off > 0].min(initial=1.0)))
    bins = np.logspace(np.log10(max(lo, 1e-9)), np.log10(hi + 1), _LOG_BINS).tolist()
    for data, label, color in ((on, "ON", (0.85, 0.2, 0.2)), (off, "OFF", (0.2, 0.4, 0.85))):
        if data.size:
            ax.hist(data, bins=bins, alpha=0.55, color=color, label=f"{label} (mean {data.mean():.3g} {unit})")
    ax.set_xscale("log")
    ax.set_xlabel(f"dwell time ({unit})")
    ax.set_ylabel("count")
    ax.set_title("ON/OFF dwell-time distributions")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def _plot_ccdf(ax: Axes, times: NDArray[np.float64], stats: DwellStats, label: str, color: str) -> None:
    """Empirical survival curve with exponential and power-law overlays."""
    times = np.sort(times[times > 0])
    if times.size == 0:
        return
    ccdf = 1.0 - np.arange(times.size) / times.size
    ax.loglog(times, ccdf, ".", ms=4, color=color, label=f"{label} (n={times.size})")
    if stats.exponential is not None:
        ax.loglog(times, np.exp(-times / stats.exponential.tau), "-", color=color, lw=1.0, alpha=0.6)
    if stats.power_law is not None:
        pl = stats.power_law
        mask = times >= pl.xmin
        if mask.any():
            ref = ccdf[mask][0]
            ax.loglog(times[mask], ref * (times[mask] / pl.xmin) ** (-(pl.alpha - 1.0)), "--", color=color, lw=1.0)


def make_stats_figure(result: AnalysisResult) -> Figure:
    """Build the dwell-time statistics figure.

    Args:
        result: A completed analysis.

    Returns:
        The Matplotlib figure (caller is responsible for closing it).
    """
    unit = "s" if result.params.frame_interval_s != 1.0 else "frames"
    on, off = _pooled_dwells(result)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    _plot_dwell_hist(ax[0], on, off, unit)
    _plot_ccdf(ax[1], on, result.stats.on, "ON", "tab:red")
    _plot_ccdf(ax[1], off, result.stats.off, "OFF", "tab:blue")
    ax[1].set_xlabel(f"dwell time ({unit})")
    ax[1].set_ylabel("P(T > t)")
    ax[1].set_title("Survival (dots) vs exponential (solid) / power-law (dashed)")
    if ax[1].get_legend_handles_labels()[0]:
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_figures(result: AnalysisResult, overview_path: Path, stats_path: Path, dpi: int = 120) -> None:
    """Render both figures to PNG files and release them.

    Args:
        result: A completed analysis.
        overview_path: Destination for the overview figure.
        stats_path: Destination for the statistics figure.
        dpi: Output resolution in dots per inch.
    """
    overview = make_overview_figure(result)
    overview.savefig(overview_path, dpi=dpi)
    plt.close(overview)
    stats_fig = make_stats_figure(result)
    stats_fig.savefig(stats_path, dpi=dpi)
    plt.close(stats_fig)
