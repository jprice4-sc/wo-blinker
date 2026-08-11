"""End-to-end single-stack analysis and result serialization.

:func:`analyze_stack` runs the full pipeline (detect -> extract -> segment ->
aggregate) and returns a self-contained :class:`AnalysisResult`.
:func:`save_results` writes the human- and machine-readable outputs: a
per-emitter table, the pooled dwell times, a JSON summary and the figures.

Batch orchestration over many files is intentionally out of scope here; loop
over :func:`analyze_file` to process several stacks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .detection import detect_spots
from .loading import load_stack
from .states import analyze_traces
from .stats import aggregate, combine
from .traces import extract_traces

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from .detection import Spots
    from .loading import Stack
    from .params import AnalysisParams
    from .states import TraceResult
    from .stats import AggregateStats, CombinedStats
    from .traces import TraceSet

_DEFAULT_TIFF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")

_BATCH_COLUMNS = (
    "stack",
    "n_spots",
    "n_blinkers",
    "blink_fraction",
    "on_median_s",
    "off_median_s",
    "amplitude_median",
    "transient_tau_s",
)

_EMITTER_COLUMNS = (
    "index",
    "row",
    "col",
    "low_level",
    "high_level",
    "amplitude",
    "snr",
    "n_transitions",
    "duty_cycle",
    "mean_on_s",
    "mean_off_s",
    "is_blinker",
)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Everything produced by analyzing one stack.

    Attributes:
        stack_name: Identifier of the analyzed stack.
        params: Parameters the analysis was run with.
        spots: Detected emitter locations and the detection image.
        traces: Extracted background-subtracted traces.
        trace_results: Per-emitter blinking statistics.
        stats: Population-level aggregate statistics.
        global_trace: Mean frame intensity versus frame index.
    """

    stack_name: str
    params: AnalysisParams
    spots: Spots
    traces: TraceSet
    trace_results: list[TraceResult]
    stats: AggregateStats
    global_trace: NDArray[np.float64]


def analyze_stack(stack: Stack, params: AnalysisParams) -> AnalysisResult:
    """Run the full blinking analysis on an in-memory stack.

    Args:
        stack: The loaded time-series stack.
        params: Analysis parameters.

    Returns:
        The completed :class:`AnalysisResult`.
    """
    spots = detect_spots(stack, params)
    traceset = extract_traces(stack, spots, params)
    trace_results = analyze_traces(traceset, params)
    global_trace = stack.data.mean(axis=(1, 2), dtype=np.float64)
    stats = aggregate(trace_results, traceset, global_trace, params)
    return AnalysisResult(
        stack_name=stack.name,
        params=params,
        spots=spots,
        traces=traceset,
        trace_results=trace_results,
        stats=stats,
        global_trace=global_trace,
    )


def analyze_file(path: str | Path, params: AnalysisParams) -> AnalysisResult:
    """Load a TIFF stack from disk and analyze it.

    Args:
        path: Path to a multi-page TIFF stack.
        params: Analysis parameters.

    Returns:
        The completed :class:`AnalysisResult`.
    """
    stack = load_stack(path, params.frame_interval_s)
    return analyze_stack(stack, params)


def _emitter_row(result: AnalysisResult, r: TraceResult) -> list[object]:
    """Build one per-emitter CSV row."""
    row, col = result.traces.coords[r.index]
    mean_on = float(r.on_times.mean()) if r.on_times.size else float("nan")
    mean_off = float(r.off_times.mean()) if r.off_times.size else float("nan")
    return [
        r.index,
        int(row),
        int(col),
        r.fit.low_level,
        r.fit.high_level,
        r.fit.amplitude,
        r.fit.snr,
        r.n_transitions,
        r.duty_cycle,
        mean_on,
        mean_off,
        int(r.is_blinker),
    ]


def _write_emitter_table(result: AnalysisResult, path: Path) -> None:
    """Write the per-emitter statistics table as CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_EMITTER_COLUMNS)
        writer.writerows(_emitter_row(result, r) for r in result.trace_results)


def _write_dwell_times(result: AnalysisResult, on_path: Path, off_path: Path) -> None:
    """Write pooled ON/OFF dwell times (one value per line) as CSV."""
    blinkers = [r for r in result.trace_results if r.is_blinker]
    for kind_path, arrays in ((on_path, [r.on_times for r in blinkers]), (off_path, [r.off_times for r in blinkers])):
        pooled = np.concatenate(arrays) if arrays else np.empty(0)
        with kind_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dwell_time_s"])
            writer.writerows([[float(v)] for v in pooled])


def _write_summary(result: AnalysisResult, path: Path) -> None:
    """Write the aggregate statistics and parameters as JSON."""
    payload = {
        "stack_name": result.stack_name,
        "params": asdict(result.params),
        "stats": asdict(result.stats),
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def save_results(result: AnalysisResult, outdir: str | Path, *, figures: bool = True) -> list[Path]:
    """Write all analysis outputs to a directory.

    Args:
        result: A completed analysis.
        outdir: Directory to write into; created if it does not exist.
        figures: Whether to render and save the PNG figures.

    Returns:
        The paths written, in a stable order.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    emitter_path = out / "emitters.csv"
    on_path = out / "on_dwell_times.csv"
    off_path = out / "off_dwell_times.csv"
    summary_path = out / "summary.json"

    _write_emitter_table(result, emitter_path)
    _write_dwell_times(result, on_path, off_path)
    _write_summary(result, summary_path)
    written = [emitter_path, on_path, off_path, summary_path]

    if figures:
        from . import plotting  # noqa: PLC0415  (defer matplotlib import until needed)

        overview_path = out / "overview.png"
        stats_path = out / "statistics.png"
        plotting.save_figures(result, overview_path, stats_path)
        written += [overview_path, stats_path]

    return written


def find_stacks(input_dir: str | Path, patterns: Sequence[str] = _DEFAULT_TIFF_PATTERNS) -> list[Path]:
    """List TIFF stacks in a directory.

    Args:
        input_dir: Directory to search (non-recursively).
        patterns: Glob patterns matched against file names.

    Returns:
        Sorted, de-duplicated list of matching paths.
    """
    directory = Path(input_dir)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(directory.glob(pattern))
    return sorted(found)


def _batch_row(result: AnalysisResult) -> list[object]:
    """Build one per-stack row for the batch summary."""
    stats = result.stats
    return [
        result.stack_name,
        stats.n_spots,
        stats.n_blinkers,
        stats.blink_fraction,
        stats.on.summary.median,
        stats.off.summary.median,
        stats.amplitude.median,
        stats.transient.tau if stats.transient is not None else float("nan"),
    ]


def _write_batch_summary(results: list[AnalysisResult], path: Path) -> None:
    """Write a one-row-per-stack summary table across a batch."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_BATCH_COLUMNS)
        writer.writerows(_batch_row(result) for result in results)


def analyze_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    params: AnalysisParams,
    *,
    figures: bool = True,
    on_result: Callable[[Path, AnalysisResult], None] | None = None,
) -> list[AnalysisResult]:
    """Analyze every TIFF stack in a directory and write per-stack outputs.

    Stacks are selected by :func:`find_stacks` (``*.tif`` / ``*.tiff``). Each
    stack's outputs land in ``output_dir/<stack name>/``; a combined
    ``batch_summary.csv`` (one row per stack) is written at the top of
    ``output_dir``.

    Args:
        input_dir: Directory of multi-page TIFF stacks.
        output_dir: Directory to write results into; created if missing.
        params: Analysis parameters, shared by every stack.
        figures: Whether to render PNG figures for each stack.
        on_result: Optional callback invoked as ``on_result(path, result)`` after
            each stack, e.g. to print progress.

    Returns:
        One :class:`AnalysisResult` per analyzed stack, in filename order.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[AnalysisResult] = []
    for path in find_stacks(input_dir):
        result = analyze_file(path, params)
        save_results(result, out / path.stem, figures=figures)
        results.append(result)
        if on_result is not None:
            on_result(path, result)
    _write_batch_summary(results, out / "batch_summary.csv")
    return results


def combine_results(results: list[AnalysisResult], params: AnalysisParams) -> CombinedStats:
    """Pool the key metrics of several analyzed stacks into one summary.

    Args:
        results: Completed analyzes, e.g. the return value of
            :func:`analyze_directory`.
        params: Analysis parameters (supplies the power-law cutoff).

    Returns:
        The combined statistics across every stack.
    """
    return combine([result.trace_results for result in results], params)


def save_combined_summary(combined: CombinedStats, output_dir: str | Path) -> Path:
    """Write the combined-across-stacks statistics as JSON.

    Args:
        combined: The pooled statistics from :func:`combine_results`.
        output_dir: Directory to write ``combined_summary.json`` into.

    Returns:
        The path written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "combined_summary.json"
    with path.open("w") as handle:
        json.dump(asdict(combined), handle, indent=2)
    return path
