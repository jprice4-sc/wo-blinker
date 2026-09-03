"""Two-state segmentation of intensity traces, and per-emitter classification.

Each trace is split into a dark (OFF) and bright (ON) level with Otsu's
threshold. From the raw split we measure how cleanly the two states separate:
their Michelson ``contrast`` and their separation ``sep`` in units of the pooled
within-state noise. The ON/OFF state sequence is then debounced with a median
filter (removing single-frame threshold flicker) before we count transitions
and read off ON/OFF dwell times as the run lengths.

Every emitter is classified from those numbers:

* **static** -- the two levels are not genuinely separated (fails the ``sep`` or
  ``contrast`` cut); the trace is essentially flat noise.
* **stepper** -- clearly two-state but with only a few switches (a one-off step
  or activation, not stochastic blinking).
* **stochastic** -- clearly two-state and switching often (``>= min_transitions``);
  this is a genuine blinker.

The first and last runs of every trace are censored (the true switch happened
outside the record) and are excluded from the reported dwell times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .params import AnalysisParams
    from .traces import TraceSet

Category = Literal["static", "stepper", "stochastic"]

_MIN_STATE_SAMPLES = 3
_MIN_RUNS_FOR_CENSORING = 2
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class StateFit:
    """The two levels recovered from a single trace and how well they separate.

    Attributes:
        off_level: Mean intensity of the OFF (dark) state.
        on_level: Mean intensity of the ON (bright) state.
        threshold: Otsu intensity separating the two states.
        contrast: Michelson contrast ``(on-off)/(on+off)``.
        sep: State separation ``(on-off)/(std_on+std_off)``, in noise sigmas.
    """

    off_level: float
    on_level: float
    threshold: float
    contrast: float
    sep: float

    @property
    def amplitude(self) -> float:
        """Intensity difference between the ON and OFF levels."""
        return self.on_level - self.off_level


@dataclass(frozen=True, slots=True)
class EmitterResult:
    """Blinking statistics and classification for one emitter trace.

    Attributes:
        index: Row index into the originating :class:`~blinker.traces.TraceSet`.
        row: Emitter row position in the frame.
        col: Emitter column position in the frame.
        mean: Time-mean of the trace.
        std: Temporal standard deviation of the trace.
        cv: Relative fluctuation ``std / mean``.
        fit: The two-state fit (levels, contrast, separation).
        on_fraction: Fraction of debounced frames in the ON state (duty cycle).
        n_transitions: Number of debounced ON/OFF switches.
        on_times: ON dwell durations in seconds (censored edge runs removed).
        off_times: OFF dwell durations in seconds (censored edge runs removed).
        category: ``"static"``, ``"stepper"`` or ``"stochastic"``.
    """

    index: int
    row: int
    col: int
    mean: float
    std: float
    cv: float
    fit: StateFit
    on_fraction: float
    n_transitions: int
    on_times: NDArray[np.float64]
    off_times: NDArray[np.float64]
    category: Category

    @property
    def is_blinker(self) -> bool:
        """Whether this emitter is a genuine stochastic blinker."""
        return self.category == "stochastic"

    @property
    def mean_on_s(self) -> float:
        """Mean ON dwell time in seconds (``nan`` if none survived censoring)."""
        return float(self.on_times.mean()) if self.on_times.size else float("nan")

    @property
    def mean_off_s(self) -> float:
        """Mean OFF dwell time in seconds (``nan`` if none survived censoring)."""
        return float(self.off_times.mean()) if self.off_times.size else float("nan")


def segment(trace: NDArray[np.float64]) -> StateFit:
    """Split a trace into two levels with Otsu's threshold and score the split.

    Args:
        trace: The intensity trace.

    Returns:
        The recovered :class:`StateFit`. A flat or single-level trace yields
        ``contrast = sep = 0``.
    """
    x = np.asarray(trace, dtype=np.float64)
    if x.min() >= x.max():
        level = float(x.mean())
        return StateFit(off_level=level, on_level=level, threshold=level, contrast=0.0, sep=0.0)
    threshold = float(threshold_otsu(x))
    on = x > threshold
    off = ~on
    if on.sum() < _MIN_STATE_SAMPLES or off.sum() < _MIN_STATE_SAMPLES:
        level = float(x.mean())
        return StateFit(off_level=level, on_level=level, threshold=threshold, contrast=0.0, sep=0.0)
    on_level = float(x[on].mean())
    off_level = float(x[off].mean())
    noise = float(x[on].std() + x[off].std())
    amp = on_level - off_level
    contrast = amp / (on_level + off_level + _EPS)
    sep = amp / (noise + _EPS)
    return StateFit(off_level=off_level, on_level=on_level, threshold=threshold, contrast=contrast, sep=sep)


def debounce(state: NDArray[np.bool_], frames: int) -> NDArray[np.bool_]:
    """Median-filter a boolean state sequence to remove short threshold flicker.

    Args:
        state: Boolean ON/OFF sequence.
        frames: Median-filter window in frames (``<= 1`` returns ``state``).

    Returns:
        The debounced boolean sequence.
    """
    if frames <= 1:
        return state
    return median_filter(state.astype(np.int8), size=frames).astype(bool)


def _run_lengths(state: NDArray[np.bool_]) -> tuple[NDArray[np.int64], bool]:
    """Return consecutive run lengths of a boolean sequence and its first value.

    Args:
        state: Boolean state sequence.

    Returns:
        The run lengths and the value of the first run.
    """
    changes = np.flatnonzero(np.diff(state.astype(np.int8)) != 0)
    edges = np.concatenate(([0], changes + 1, [state.size]))
    return np.diff(edges).astype(np.int64), bool(state[0])


def dwell_times_from_state(
    state: NDArray[np.bool_],
    frame_interval_s: float,
    *,
    drop_censored: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract ON and OFF dwell times from a boolean state sequence.

    Args:
        state: Boolean state sequence (``True`` == ON).
        frame_interval_s: Seconds per frame.
        drop_censored: If ``True``, drop the first and last runs, whose true
            durations are unknown.

    Returns:
        Arrays of ON and OFF dwell times, in seconds.
    """
    runs, first = _run_lengths(state)
    values = np.array([first if k % 2 == 0 else not first for k in range(runs.size)], dtype=bool)
    if drop_censored and runs.size > _MIN_RUNS_FOR_CENSORING:
        runs, values = runs[1:-1], values[1:-1]
    seconds = runs.astype(np.float64) * frame_interval_s
    return seconds[values], seconds[~values]


def _classify(fit: StateFit, n_transitions: int, params: AnalysisParams) -> Category:
    """Classify an emitter from its two-state fit and switching activity."""
    bimodal = fit.sep >= params.sep_threshold and fit.contrast >= params.contrast_threshold
    if not bimodal:
        return "static"
    return "stochastic" if n_transitions >= params.min_transitions else "stepper"


def analyze_trace(
    trace: NDArray[np.float64],
    index: int,
    row: int,
    col: int,
    params: AnalysisParams,
) -> EmitterResult:
    """Compute blinking statistics and a classification for one emitter trace.

    Args:
        trace: The intensity trace.
        index: Row index of this trace in its trace set.
        row: Emitter row position in the frame.
        col: Emitter column position in the frame.
        params: Analysis parameters.

    Returns:
        The :class:`EmitterResult` for this trace.
    """
    x = np.asarray(trace, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std())
    cv = std / (abs(mean) + _EPS)

    fit = segment(x)
    state = debounce(x > fit.threshold, params.debounce_frames)
    runs, _ = _run_lengths(state)
    n_transitions = int(runs.size - 1)
    on_fraction = float(state.mean())
    on_times, off_times = dwell_times_from_state(state, params.frame_interval_s)
    category = _classify(fit, n_transitions, params)
    return EmitterResult(
        index=index,
        row=row,
        col=col,
        mean=mean,
        std=std,
        cv=cv,
        fit=fit,
        on_fraction=on_fraction,
        n_transitions=n_transitions,
        on_times=on_times,
        off_times=off_times,
        category=category,
    )


def analyze_traces(traceset: TraceSet, params: AnalysisParams) -> list[EmitterResult]:
    """Analyze every trace in a trace set.

    Args:
        traceset: The extracted traces.
        params: Analysis parameters.

    Returns:
        One :class:`EmitterResult` per trace, in trace-set row order.
    """
    return [
        analyze_trace(
            traceset.traces[i],
            i,
            int(traceset.coords[i, 0]),
            int(traceset.coords[i, 1]),
            params,
        )
        for i in range(traceset.n_spots)
    ]
