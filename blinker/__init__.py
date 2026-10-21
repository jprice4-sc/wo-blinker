"""Systematic blinking-statistics analysis of TIFF time-series stacks.

The public API mirrors the pipeline stages:

* :func:`load_stack` / :func:`open_stack` -- read a TIFF stack (in memory, or
  streamed for large files).
* :func:`detect_spots` -- auto-detect emitters on a temporal projection.
* :func:`extract_traces` -- pull per-emitter box-summed intensity traces.
* :func:`analyze_traces` -- Otsu two-state segmentation and classification.
* :func:`aggregate` -- pool population-level statistics.

For a one-call analysis use :func:`analyze_file` (streams from disk) or
:func:`analyze_stack` (in-memory) followed by :func:`save_results`.
"""

from __future__ import annotations

from .analyzer import Blinker
from .detection import Spots, detect_spots, project
from .loading import FrameSource, Stack, TiffStack, load_stack, open_stack, stack_from_array
from .params import AnalysisParams
from .pipeline import (
    AnalysisResult,
    analyze_directory,
    analyze_file,
    analyze_stack,
    combine_results,
    find_stacks,
    save_combined_summary,
    save_results,
)
from .states import (
    EmitterResult,
    StateFit,
    analyze_trace,
    analyze_traces,
    dwell_times_from_state,
    segment,
)
from .stats import AggregateStats, CombinedStats, DwellStats, ExponentialFit, PowerLawFit, Summary, aggregate
from .traces import TraceSet, extract_traces

__version__ = "0.2.0"

__all__ = [
    "AggregateStats",
    "AnalysisParams",
    "AnalysisResult",
    "Blinker",
    "CombinedStats",
    "DwellStats",
    "EmitterResult",
    "ExponentialFit",
    "FrameSource",
    "PowerLawFit",
    "Spots",
    "Stack",
    "StateFit",
    "Summary",
    "TiffStack",
    "TraceSet",
    "aggregate",
    "analyze_directory",
    "analyze_file",
    "analyze_stack",
    "analyze_trace",
    "analyze_traces",
    "combine_results",
    "detect_spots",
    "dwell_times_from_state",
    "extract_traces",
    "find_stacks",
    "load_stack",
    "open_stack",
    "project",
    "save_combined_summary",
    "save_results",
    "segment",
    "stack_from_array",
]
