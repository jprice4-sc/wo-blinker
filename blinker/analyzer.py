"""Object-oriented facade over the functional blinking-analysis pipeline.

:class:`Blinker` bundles a stack, its analysis parameters and the computed
results into one object with a fluent, stateful workflow::

    from blinker import Blinker, AnalysisParams

    b = Blinker(AnalysisParams(frame_interval_s=0.05, min_snr=3.0))
    b.load("stack.tif").analyze()
    print(b.summary())
    fig = b.plot_summary()
    b.save("results")

Analysis is computed lazily: accessing any result (``on_times``, ``stats``, a
plot, ...) runs the pipeline once and caches it. The functional API
(:func:`~blinker.pipeline.analyze_stack` and friends) is unchanged; this class
only wraps it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import numpy as np

from .loading import load_stack, stack_from_array
from .params import AnalysisParams
from .pipeline import analyze_stack, save_results
from .stats import format_summary

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .detection import Spots
    from .loading import Stack
    from .pipeline import AnalysisResult
    from .states import TraceResult
    from .stats import AggregateStats
    from .traces import TraceSet

_SUMMARY_DPI = 110


class Blinker:
    """Stateful, object-oriented interface to the blinking analysis.

    Attributes:
        params: The analysis parameters applied to the loaded stack.
        stack: The currently loaded stack, or ``None`` before :meth:`load`.
        result: The cached analysis, or ``None`` before it has been computed.
    """

    def __init__(self, params: AnalysisParams | None = None, **overrides: object) -> None:
        """Create an analyzer.

        Args:
            params: Base analysis parameters (defaults to :class:`AnalysisParams`).
            **overrides: Individual :class:`AnalysisParams` fields to override,
                e.g. ``Blinker(frame_interval_s=0.05, min_snr=4)``.
        """
        base = params if params is not None else AnalysisParams()
        self.params: AnalysisParams = replace(base, **overrides) if overrides else base  # type: ignore[arg-type]
        self.stack: Stack | None = None
        self.result: AnalysisResult | None = None

    @override
    def __repr__(self) -> str:
        """Show the load/analysis state."""
        where = self.stack.name if self.stack is not None else "no stack"
        state = "analyzed" if self.result is not None else "not analyzed"
        return f"Blinker({where!r}, {state})"

    # -- loading -------------------------------------------------------
    def load(self, path: str | Path) -> Self:
        """Load a TIFF stack from disk (clears any previous result).

        Args:
            path: Path to a multi-page TIFF stack.

        Returns:
            ``self``, for chaining.
        """
        self.stack = load_stack(path, self.params.frame_interval_s)
        self.result = None
        return self

    def load_array(self, array: NDArray[np.generic], name: str = "array") -> Self:
        """Load an in-memory ``(T, H, W)`` array as the stack.

        Args:
            array: A 3-D (or squeezable-to-3-D) image array.
            name: Identifier for the stack.

        Returns:
            ``self``, for chaining.
        """
        self.stack = stack_from_array(array, self.params.frame_interval_s, name)
        self.result = None
        return self

    # -- computation ---------------------------------------------------
    def analyze(self) -> Self:
        """Detect emitters, extract traces, segment ON/OFF and aggregate.

        Returns:
            ``self``, for chaining.

        Raises:
            RuntimeError: If no stack has been loaded.
        """
        if self.stack is None:
            raise RuntimeError("no stack loaded; call load() first")
        self.result = analyze_stack(self.stack, self.params)
        return self

    def _result(self) -> AnalysisResult:
        """Return the analysis, computing it on first access."""
        if self.result is None:
            self.analyze()
        result = self.result
        if result is None:  # pragma: no cover - analyze() sets it or raises
            raise RuntimeError("analysis did not produce a result")
        return result

    # -- results -------------------------------------------------------
    @property
    def results(self) -> list[TraceResult]:
        """Per-emitter blinking results (analysis runs on first access)."""
        return self._result().trace_results

    @property
    def stats(self) -> AggregateStats:
        """Population-level aggregate statistics."""
        return self._result().stats

    @property
    def spots(self) -> Spots:
        """Detected emitter locations and the detection image."""
        return self._result().spots

    @property
    def traces(self) -> TraceSet:
        """Extracted per-emitter intensity traces."""
        return self._result().traces

    @property
    def n_spots(self) -> int:
        """Number of detected emitters."""
        return self.stats.n_spots

    @property
    def n_blinkers(self) -> int:
        """Number of emitters classified as genuine blinkers."""
        return self.stats.n_blinkers

    @property
    def on_times(self) -> NDArray[np.float64]:
        """All ON dwell times (seconds), pooled over blinkers."""
        blinkers = [r for r in self.results if r.is_blinker]
        return np.concatenate([r.on_times for r in blinkers]) if blinkers else np.empty(0)

    @property
    def off_times(self) -> NDArray[np.float64]:
        """All OFF dwell times (seconds), pooled over blinkers."""
        blinkers = [r for r in self.results if r.is_blinker]
        return np.concatenate([r.off_times for r in blinkers]) if blinkers else np.empty(0)

    def dwell_times(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the pooled ``(on_times, off_times)`` in seconds."""
        return self.on_times, self.off_times

    def summary(self) -> str:
        """Return a human-readable summary of the key metrics."""
        return format_summary(self.stats)

    # -- plotting ------------------------------------------------------
    def plot_overview(self) -> Figure:
        """Detection map, transient, strongest traces and duty-cycle histogram."""
        from . import plotting  # noqa: PLC0415  (defer matplotlib import)

        return plotting.make_overview_figure(self._result())

    def plot_statistics(self) -> Figure:
        """Dwell-time CCDFs with fits, amplitude and switching-activity histograms."""
        from . import plotting  # noqa: PLC0415

        return plotting.make_stats_figure(self._result())

    def plot_summary(self) -> Figure:
        """Compact summary: detection map, dwell CCDF and key numbers."""
        from . import plotting  # noqa: PLC0415

        return plotting.make_summary_figure(self._result())

    def plot_trace(self, index: int) -> Figure:
        """Plot one emitter's trace with its fitted ON/OFF levels.

        Args:
            index: Row index of the emitter in the trace set.

        Returns:
            The Matplotlib figure.
        """
        from . import plotting  # noqa: PLC0415

        return plotting.make_trace_figure(self._result(), index)

    def save(self, outdir: str | Path, *, figures: bool = True) -> list[Path]:
        """Write all outputs (tables, JSON, figures) to a directory.

        Args:
            outdir: Directory to write into; created if it does not exist.
            figures: Whether to render the PNG figures (adds ``summary.png``).

        Returns:
            The paths written.
        """
        result = self._result()
        written = save_results(result, outdir, figures=figures)
        if figures:
            from . import plotting  # noqa: PLC0415

            summary_path = Path(outdir) / "summary.png"
            plotting.save_summary_figure(result, summary_path, dpi=_SUMMARY_DPI)
            written = [*written, summary_path]
        return written
