"""Two-state segmentation of intensity traces and dwell-time extraction.

Each emitter trace is modeled as switching between a dark (OFF) and a bright
(ON) level. We recover those two levels per trace by clustering the intensity
samples into two groups (a 1-D Gaussian mixture by default, with k-means and
Otsu alternatives), place a threshold at the crossover, and read off the ON/OFF
dwell times as the run lengths of the thresholded state sequence.

The first and last runs of every trace are censored (the true switch happened
before frame 0 or after the last frame), so they are excluded from the reported
dwell times to avoid biasing the distributions toward short times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from skimage.filters import threshold_otsu

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .params import AnalysisParams, StateMethod
    from .traces import TraceSet

_KMEANS_ITERS = 50
_GMM_ITERS = 100
_GMM_TOL = 1e-6
_INIT_LOW_PCT = 25.0
_INIT_HIGH_PCT = 75.0
_MIN_TRANSITIONS_FOR_QUAD = 2
_TWO_STATE = 2


@dataclass(frozen=True, slots=True)
class StateFit:
    """The two levels and threshold recovered from a single trace.

    Attributes:
        low_level: Mean intensity of the OFF (dark) state.
        high_level: Mean intensity of the ON (bright) state.
        threshold: Intensity separating the two states.
        noise: Pooled within-state standard deviation.
    """

    low_level: float
    high_level: float
    threshold: float
    noise: float

    @property
    def amplitude(self) -> float:
        """Intensity difference between the ON and OFF levels."""
        return self.high_level - self.low_level

    @property
    def snr(self) -> float:
        """Amplitude divided by the within-state noise."""
        if self.noise <= 0:
            return float("inf")
        return self.amplitude / self.noise


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Blinking statistics for one emitter trace.

    Attributes:
        index: Row index into the originating trace set (``TraceSet``).
        fit: The two-state fit used to threshold the trace.
        n_transitions: Number of ON/OFF (or OFF/ON) switches.
        duty_cycle: Fraction of frames spent in the ON state.
        on_times: ON dwell durations in seconds (censored edge runs removed).
        off_times: OFF dwell durations in seconds (censored edge runs removed).
        is_blinker: Whether the trace passed the SNR and transition-count cuts.
    """

    index: int
    fit: StateFit
    n_transitions: int
    duty_cycle: float
    on_times: NDArray[np.float64]
    off_times: NDArray[np.float64]
    is_blinker: bool


def _kmeans_1d(x: NDArray[np.float64]) -> tuple[float, float, NDArray[np.bool_]]:
    """Cluster 1-D samples into a low and a high group by Lloyd iteration.

    Args:
        x: Sample values.

    Returns:
        The low center, high center, and a boolean mask that is ``True`` for
        samples assigned to the high cluster.
    """
    lo = float(np.percentile(x, _INIT_LOW_PCT))
    hi = float(np.percentile(x, _INIT_HIGH_PCT))
    high_mask = x > (lo + hi) / 2
    for _ in range(_KMEANS_ITERS):
        if high_mask.any():
            hi = float(x[high_mask].mean())
        if (~high_mask).any():
            lo = float(x[~high_mask].mean())
        new_mask = x > (lo + hi) / 2
        if np.array_equal(new_mask, high_mask):
            break
        high_mask = new_mask
    return lo, hi, high_mask


def _gaussian(x: NDArray[np.float64], mean: float, var: float) -> NDArray[np.float64]:
    """Evaluate an (unnormalized-safe) Gaussian density."""
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def _gmm_1d(x: NDArray[np.float64]) -> tuple[float, float, float]:
    """Fit a two-component 1-D Gaussian mixture and return its crossover.

    Args:
        x: Sample values.

    Returns:
        The low mean, high mean and the threshold where the two weighted
        components have equal density.
    """
    lo, hi, mask = _kmeans_1d(x)
    var_floor = max(float(np.var(x)) * 1e-4, 1e-12)
    means = np.array([lo, hi], dtype=np.float64)
    variances = np.array(
        [
            max(float(np.var(x[~mask])) if (~mask).any() else var_floor, var_floor),
            max(float(np.var(x[mask])) if mask.any() else var_floor, var_floor),
        ],
        dtype=np.float64,
    )
    weights = np.array([float((~mask).mean()), float(mask.mean())], dtype=np.float64)
    weights = np.clip(weights, 1e-6, None)
    weights /= weights.sum()

    prev = -np.inf
    for _ in range(_GMM_ITERS):
        resp = np.stack([weights[k] * _gaussian(x, means[k], variances[k]) for k in range(_TWO_STATE)])
        total = resp.sum(axis=0) + 1e-300
        resp /= total
        nk = resp.sum(axis=1) + 1e-12
        means = (resp * x).sum(axis=1) / nk
        variances = np.maximum((resp * (x - means[:, None]) ** 2).sum(axis=1) / nk, var_floor)
        weights = nk / nk.sum()
        loglik = float(np.log(total).sum())
        if abs(loglik - prev) < _GMM_TOL * (abs(prev) + 1.0):
            break
        prev = loglik

    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    threshold = _mixture_threshold(means, variances, weights)
    return float(means[0]), float(means[1]), threshold


def _mixture_threshold(
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Return the intensity where the two weighted Gaussians cross.

    Args:
        means: Component means, sorted ascending.
        variances: Component variances.
        weights: Component mixing weights.

    Returns:
        The crossover intensity between the two means, or their midpoint if no
        crossover lies between them.
    """
    m0, m1 = float(means[0]), float(means[1])
    v0, v1 = float(variances[0]), float(variances[1])
    midpoint = 0.5 * (m0 + m1)
    # Solve a x^2 + b x + c = 0 for equal weighted densities.
    a = 0.5 * (1.0 / v0 - 1.0 / v1)
    b = m1 / v1 - m0 / v0
    c = 0.5 * (m0 * m0 / v0 - m1 * m1 / v1) + np.log(
        (weights[0] * np.sqrt(v1)) / (weights[1] * np.sqrt(v0) + 1e-300) + 1e-300,
    )
    if abs(a) < 1e-12:  # noqa: PLR2004
        return midpoint if abs(b) < 1e-12 else float(-c / b)  # noqa: PLR2004
    disc = b * b - 4 * a * c
    if disc < 0:
        return midpoint
    sqrt_disc = float(np.sqrt(disc))
    roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
    between = [r for r in roots if m0 <= r <= m1]
    return float(between[0]) if between else midpoint


def segment_two_state(trace: NDArray[np.float64], method: StateMethod) -> StateFit:
    """Split a single trace into two intensity levels.

    Args:
        trace: The intensity trace.
        method: Segmentation method (``"gmm"``, ``"kmeans"`` or ``"otsu"``).

    Returns:
        The recovered :class:`StateFit`.
    """
    x = np.asarray(trace, dtype=np.float64)
    if method == "gmm":
        low, high, threshold = _gmm_1d(x)
    elif method == "kmeans":
        low, high, _mask = _kmeans_1d(x)
        threshold = 0.5 * (low + high)
        low, high = (low, high) if low <= high else (high, low)
    else:  # "otsu"
        threshold = float(threshold_otsu(x)) if x.min() < x.max() else float(x.mean())
        low = float(x[x <= threshold].mean()) if (x <= threshold).any() else float(x.min())
        high = float(x[x > threshold].mean()) if (x > threshold).any() else float(x.max())

    state = x > threshold
    noise = _pooled_noise(x, state=state)
    return StateFit(low_level=low, high_level=high, threshold=threshold, noise=noise)


def _pooled_noise(x: NDArray[np.float64], *, state: NDArray[np.bool_]) -> float:
    """Pooled within-state standard deviation of a thresholded trace."""
    parts = [x[state], x[~state]]
    variances = [float(np.var(p)) for p in parts if p.size > 1]
    if not variances:
        return float(np.std(x))
    return float(np.sqrt(np.mean(variances)))


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


def _enforce_min_dwell(state: NDArray[np.bool_], min_frames: int) -> NDArray[np.bool_]:
    """Flip runs shorter than ``min_frames`` into their neighbours."""
    if min_frames <= 1:
        return state
    out = state.copy()
    while True:
        runs, first = _run_lengths(out)
        short = np.flatnonzero(runs < min_frames)
        if short.size == 0:
            return out
        starts = np.concatenate(([0], np.cumsum(runs)[:-1]))
        k = int(short[np.argmin(runs[short])])
        value = first if k % 2 == 0 else not first
        out[starts[k] : starts[k] + runs[k]] = not value


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
    if drop_censored and runs.size > _MIN_TRANSITIONS_FOR_QUAD:
        runs, values = runs[1:-1], values[1:-1]
    seconds = runs.astype(np.float64) * frame_interval_s
    return seconds[values], seconds[~values]


def analyze_trace(trace: NDArray[np.float64], index: int, params: AnalysisParams) -> TraceResult:
    """Compute blinking statistics for a single emitter trace.

    Args:
        trace: The intensity trace.
        index: Row index of this trace in its trace set.
        params: Analysis parameters.

    Returns:
        The :class:`TraceResult` for this trace.
    """
    fit = segment_two_state(trace, params.state_method)
    state = _enforce_min_dwell(np.asarray(trace) > fit.threshold, params.min_dwell_frames)
    runs, _ = _run_lengths(state)
    n_transitions = int(runs.size - 1)
    duty_cycle = float(state.mean())
    on_times, off_times = dwell_times_from_state(state, params.frame_interval_s)
    is_blinker = fit.snr >= params.min_snr and n_transitions >= params.min_transitions
    return TraceResult(
        index=index,
        fit=fit,
        n_transitions=n_transitions,
        duty_cycle=duty_cycle,
        on_times=on_times,
        off_times=off_times,
        is_blinker=is_blinker,
    )


def analyze_traces(traceset: TraceSet, params: AnalysisParams) -> list[TraceResult]:
    """Analyze every trace in a trace set.

    Args:
        traceset: The extracted traces.
        params: Analysis parameters.

    Returns:
        One :class:`TraceResult` per trace, in trace-set row order.
    """
    return [analyze_trace(traceset.traces[i], i, params) for i in range(traceset.n_spots)]
