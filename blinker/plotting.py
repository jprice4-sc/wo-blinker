"""Figure generation for a completed analysis.

Two figures summarize a run: an *overview* (detection image with emitters
circled, the field-wide transient, and the strongest blinker traces with their
fitted thresholds) and a *statistics* panel (ON/OFF dwell-time complementary
CDFs with the fitted models, plus amplitude and duty-cycle histograms).
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
    from .stats import DwellStats

plt.switch_backend("Agg")

_MAX_EXAMPLE_TRACES = 6
_HIST_BINS = 30


def _plot_detection(ax: Axes, result: AnalysisResult) -> None:
    """Draw the detection image with blinkers and non-blinkers marked."""
    proj = result.spots.projection
    ax.imshow(proj, cmap="inferno", vmax=float(np.percentile(proj, 99.5)))
    coords = result.traces.coords
    is_blinker = np.array([r.is_blinker for r in result.trace_results], dtype=bool)
    if (~is_blinker).any():
        ax.plot(coords[~is_blinker, 1], coords[~is_blinker, 0], "o", mfc="none", mec="white", ms=5, mew=0.5)
    if is_blinker.any():
        ax.plot(coords[is_blinker, 1], coords[is_blinker, 0], "o", mfc="none", mec="cyan", ms=6, mew=0.9)
    ax.set_title(f"{result.spots.projection_kind} projection: {int(is_blinker.sum())} blinkers")


def _plot_transient(ax: Axes, result: AnalysisResult) -> None:
    """Draw the field-wide mean intensity and its stretched-exponential fit."""
    t = np.arange(result.global_trace.size) * result.params.frame_interval_s
    ax.plot(t, result.global_trace, lw=0.8, color="0.3")
    fit = result.stats.transient
    if fit is not None:
        model = fit.i_inf - fit.d_i * np.exp(-((t / fit.tau) ** fit.beta))
        ax.plot(t, model, "--", color="crimson", label=f"tau={fit.tau:.1f}s beta={fit.beta:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_title("Field-wide mean vs time")


def _plot_example_traces(ax: Axes, result: AnalysisResult) -> None:
    """Draw the strongest blinker traces with their fitted thresholds."""
    blinkers = [r for r in result.trace_results if r.is_blinker]
    blinkers.sort(key=lambda r: r.fit.snr, reverse=True)
    t = np.arange(result.traces.n_frames) * result.params.frame_interval_s
    for r in blinkers[:_MAX_EXAMPLE_TRACES]:
        (line,) = ax.plot(t, result.traces.traces[r.index], lw=0.7)
        ax.axhline(r.fit.threshold, color=line.get_color(), ls=":", lw=0.6, alpha=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("intensity (counts)")
    ax.set_title("Strongest blinker traces")


def _plot_ccdf(ax: Axes, times: NDArray[np.float64], stats: DwellStats, label: str, color: str) -> None:
    """Overlay an empirical complementary CDF and its fitted models."""
    times = np.sort(np.asarray(times, dtype=np.float64))
    if times.size == 0:
        return
    ccdf = 1.0 - np.arange(times.size) / times.size
    ax.loglog(times, ccdf, "o", ms=3, color=color, label=f"{label} (n={times.size})")
    if stats.power_law is not None:
        pl = stats.power_law
        mask = times >= pl.xmin
        ref = ccdf[mask][0] if mask.any() else 1.0
        ax.loglog(times[mask], ref * (times[mask] / pl.xmin) ** (-(pl.alpha - 1.0)), "-", color=color, lw=1.0)


def make_overview_figure(result: AnalysisResult) -> Figure:
    """Build the overview figure.

    Args:
        result: A completed analysis.

    Returns:
        The Matplotlib figure (caller is responsible for closing it).
    """
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    _plot_detection(ax[0, 0], result)
    _plot_transient(ax[0, 1], result)
    _plot_example_traces(ax[1, 0], result)
    _plot_histogram(ax[1, 1], result)
    fig.tight_layout()
    return fig


def _plot_histogram(ax: Axes, result: AnalysisResult) -> None:
    """Draw the blinker duty-cycle histogram."""
    duty = np.array([r.duty_cycle for r in result.trace_results if r.is_blinker], dtype=np.float64)
    if duty.size:
        ax.hist(duty, bins=np.linspace(0, 1, _HIST_BINS).tolist())
    ax.set_xlabel("duty cycle (fraction ON)")
    ax.set_title("Blinker duty cycles")


def make_stats_figure(result: AnalysisResult) -> Figure:
    """Build the dwell-time and amplitude statistics figure.

    Args:
        result: A completed analysis.

    Returns:
        The Matplotlib figure (caller is responsible for closing it).
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    blinkers = [r for r in result.trace_results if r.is_blinker]
    on = np.concatenate([r.on_times for r in blinkers]) if blinkers else np.empty(0)
    off = np.concatenate([r.off_times for r in blinkers]) if blinkers else np.empty(0)
    _plot_ccdf(ax[0], on, result.stats.on, "ON", "tab:blue")
    _plot_ccdf(ax[0], off, result.stats.off, "OFF", "tab:red")
    ax[0].set_xlabel("dwell time (s)")
    ax[0].set_ylabel("P(T >= t)")
    ax[0].set_title("Dwell-time CCDF")
    ax[0].legend(fontsize=8)

    amps = np.array([r.fit.amplitude for r in blinkers], dtype=np.float64)
    if amps.size:
        ax[1].hist(amps, bins=_HIST_BINS)
    ax[1].set_xlabel("ON/OFF amplitude (counts)")
    ax[1].set_title("Blinker amplitudes")

    ntr = np.array([r.n_transitions for r in blinkers], dtype=np.float64)
    if ntr.size:
        ax[2].hist(ntr, bins=_HIST_BINS)
    ax[2].set_xlabel("transitions per blinker")
    ax[2].set_title("Switching activity")
    fig.tight_layout()
    return fig


def save_figures(result: AnalysisResult, overview_path: Path, stats_path: Path, dpi: int = 110) -> None:
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
