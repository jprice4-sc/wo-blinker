"""Systematic blinking-statistics analysis of TIFF time-series stacks.

The public API mirrors the pipeline stages:

* :func:`load_stack` -- read a TIFF stack.
* :func:`detect_spots` -- auto-detect emitters.
* :func:`extract_traces` -- pull per-emitter intensity traces.
* :func:`analyze_traces` -- two-state ON/OFF segmentation.
* :func:`aggregate` -- pool population-level statistics.

For a one-call analysis use :func:`analyze_file` (or :func:`analyze_stack` for
in-memory data) followed by :func:`save_results`.
"""

from __future__ import annotations

from .detection import Spots, detect_spots, project
from .loading import Stack, load_stack, stack_from_array
from .params import AnalysisParams
from .pipeline import (
    AnalysisResult,
    analyze_directory,
    analyze_file,
    analyze_stack,
    find_stacks,
    save_results,
)
from .states import StateFit, TraceResult, analyze_trace, analyze_traces, segment_two_state
from .stats import AggregateStats, DwellStats, aggregate
from .traces import TraceSet, extract_traces

__version__ = "0.1.0"

__all__ = [
    "AggregateStats",
    "AnalysisParams",
    "AnalysisResult",
    "DwellStats",
    "Spots",
    "Stack",
    "StateFit",
    "TraceResult",
    "TraceSet",
    "aggregate",
    "analyze_directory",
    "analyze_file",
    "analyze_stack",
    "analyze_trace",
    "analyze_traces",
    "detect_spots",
    "extract_traces",
    "find_stacks",
    "load_stack",
    "project",
    "save_results",
    "segment_two_state",
    "stack_from_array",
]
