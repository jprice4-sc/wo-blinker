"""Aggregate blinking statistics pooled across all emitters in a stack.

Given the per-trace results this module builds the population-level summary an
experimentalist actually reports: pooled ON/OFF dwell-time distributions with
power-law and exponential fits, the emitter amplitude and duty-cycle
distributions, the fraction of detected spots that blink, and a stretched-
exponential fit of the field-wide photo-activation transient.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .params import AnalysisParams
    from .states import TraceResult
    from .traces import TraceSet

_MIN_FIT_SAMPLES = 5
_MIN_TRANSIENT_FRAMES = 10
_TRANSIENT_SNR = 3.0
_BETA_LOW = 0.05
_BETA_HIGH = 2.0


@dataclass(frozen=True, slots=True)
class Summary:
    """Descriptive statistics of a 1-D sample.

    Attributes:
        n: Number of samples.
        mean: Arithmetic mean (``nan`` if empty).
        median: Median (``nan`` if empty).
        std: Standard deviation (``nan`` if empty).
        minimum: Smallest value (``nan`` if empty).
        maximum: Largest value (``nan`` if empty).
    """

    n: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """Maximum-likelihood continuous power-law fit ``p(t) ~ t**-alpha``.

    Attributes:
        alpha: Scaling exponent.
        alpha_stderr: Standard error of the exponent estimate.
        xmin: Lower cutoff used in the fit.
        n: Number of samples at or above ``xmin``.
    """

    alpha: float
    alpha_stderr: float
    xmin: float
    n: int


@dataclass(frozen=True, slots=True)
class ExponentialFit:
    """Maximum-likelihood exponential fit ``p(t) ~ exp(-t/tau)``.

    Attributes:
        tau: Characteristic time (mean dwell time).
        rate: Reciprocal of ``tau``.
        n: Number of samples used.
    """

    tau: float
    rate: float
    n: int


@dataclass(frozen=True, slots=True)
class DwellStats:
    """Dwell-time distribution summary with fitted models.

    Attributes:
        summary: Descriptive statistics of the pooled dwell times (seconds).
        power_law: Power-law fit, or ``None`` if too few samples.
        exponential: Exponential fit, or ``None`` if too few samples.
    """

    summary: Summary
    power_law: PowerLawFit | None
    exponential: ExponentialFit | None


@dataclass(frozen=True, slots=True)
class StretchedExpFit:
    """Stretched-exponential transient ``I(t) = I_inf - dI * exp(-(t/tau)**beta)``.

    Attributes:
        i_inf: Steady-state intensity.
        d_i: Amplitude of the transient.
        tau: Characteristic time in seconds.
        beta: Stretching exponent.
    """

    i_inf: float
    d_i: float
    tau: float
    beta: float


@dataclass(frozen=True, slots=True)
class AggregateStats:
    """Population-level blinking statistics for one stack.

    Attributes:
        n_spots: Number of detected emitters.
        n_blinkers: Number classified as genuine blinkers.
        blink_fraction: ``n_blinkers / n_spots``.
        on: Pooled ON dwell-time statistics.
        off: Pooled OFF dwell-time statistics.
        amplitude: Distribution of blinker ON/OFF amplitudes.
        duty_cycle: Distribution of blinker duty cycles.
        transient: Field-wide photo-activation transient fit, or ``None``.
    """

    n_spots: int
    n_blinkers: int
    blink_fraction: float
    on: DwellStats
    off: DwellStats
    amplitude: Summary
    duty_cycle: Summary
    transient: StretchedExpFit | None


@dataclass(frozen=True, slots=True)
class CombinedStats:
    """Blinking statistics pooled across several stacks.

    Attributes:
        n_stacks: Number of stacks combined.
        n_spots: Total detected emitters across all stacks.
        n_blinkers: Total genuine blinkers across all stacks.
        blink_fraction: ``n_blinkers / n_spots`` over the whole set.
        on: ON dwell-time statistics pooled across every blinker.
        off: OFF dwell-time statistics pooled across every blinker.
        amplitude: Distribution of blinker ON/OFF amplitudes, all stacks.
        duty_cycle: Distribution of blinker duty cycles, all stacks.
    """

    n_stacks: int
    n_spots: int
    n_blinkers: int
    blink_fraction: float
    on: DwellStats
    off: DwellStats
    amplitude: Summary
    duty_cycle: Summary


def summarize(values: NDArray[np.float64]) -> Summary:
    """Compute descriptive statistics, tolerating an empty input.

    Args:
        values: 1-D sample array.

    Returns:
        The :class:`Summary`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        nan = float("nan")
        return Summary(n=0, mean=nan, median=nan, std=nan, minimum=nan, maximum=nan)
    return Summary(
        n=int(values.size),
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=float(values.std()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def fit_power_law(times: NDArray[np.float64], xmin: float | None = None) -> PowerLawFit | None:
    """Fit a continuous power law by maximum likelihood.

    Args:
        times: Dwell times (seconds), all positive.
        xmin: Lower cutoff. ``None`` uses the smallest positive sample.

    Returns:
        The :class:`PowerLawFit`, or ``None`` if there are too few samples.
    """
    times = np.asarray(times, dtype=np.float64)
    times = times[times > 0]
    cutoff = float(times.min()) if xmin is None else float(xmin)
    sample = times[times >= cutoff]
    if sample.size < _MIN_FIT_SAMPLES or cutoff <= 0:
        return None
    n = int(sample.size)
    alpha = 1.0 + n / float(np.sum(np.log(sample / cutoff)))
    stderr = (alpha - 1.0) / np.sqrt(n)
    return PowerLawFit(alpha=alpha, alpha_stderr=stderr, xmin=cutoff, n=n)


def fit_exponential(times: NDArray[np.float64]) -> ExponentialFit | None:
    """Fit an exponential distribution by maximum likelihood.

    Args:
        times: Dwell times (seconds), all positive.

    Returns:
        The :class:`ExponentialFit`, or ``None`` if there are too few samples.
    """
    times = np.asarray(times, dtype=np.float64)
    times = times[times > 0]
    if times.size < _MIN_FIT_SAMPLES:
        return None
    tau = float(times.mean())
    return ExponentialFit(tau=tau, rate=1.0 / tau, n=int(times.size))


def _dwell_stats(times: NDArray[np.float64], xmin: float | None) -> DwellStats:
    """Bundle descriptive statistics and fits for one dwell-time sample."""
    return DwellStats(
        summary=summarize(times),
        power_law=fit_power_law(times, xmin),
        exponential=fit_exponential(times),
    )


def fit_transient(global_trace: NDArray[np.float64], frame_interval_s: float) -> StretchedExpFit | None:
    """Fit a stretched-exponential to the field-wide mean-intensity transient.

    Args:
        global_trace: Mean frame intensity versus frame index.
        frame_interval_s: Seconds per frame.

    Returns:
        The :class:`StretchedExpFit`, or ``None`` if the fit does not converge.
    """
    y = np.asarray(global_trace, dtype=np.float64)
    if y.size < _MIN_TRANSIENT_FRAMES:
        return None
    t = np.arange(y.size, dtype=np.float64) * frame_interval_s

    # A stretched-exponential transient only exists if the mean genuinely drifts.
    # Compare the net change across the record to the frame-to-frame noise; if the
    # trace is essentially flat there is nothing to fit.
    edge = max(y.size // 10, 1)
    drift = abs(float(y[-edge:].mean() - y[:edge].mean()))
    noise = max(float(np.std(np.diff(y))) / np.sqrt(2.0), 1e-12)
    if drift < _TRANSIENT_SNR * noise:
        return None

    def model(t: NDArray[np.float64], i_inf: float, d_i: float, tau: float, beta: float) -> NDArray[np.float64]:
        return i_inf - d_i * np.exp(-((t / tau) ** beta))

    p0 = [float(y[-1]), float(y[-1] - y[0]), max(float(t[-1]) / 4.0, 1e-6), 0.7]
    bounds = ([-np.inf, -np.inf, 1e-9, _BETA_LOW], [np.inf, np.inf, np.inf, _BETA_HIGH])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("error", optimize.OptimizeWarning)
            popt, _ = optimize.curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError, optimize.OptimizeWarning):
        return None
    return StretchedExpFit(i_inf=float(popt[0]), d_i=float(popt[1]), tau=float(popt[2]), beta=float(popt[3]))


def aggregate(
    results: list[TraceResult],
    traceset: TraceSet,
    global_trace: NDArray[np.float64],
    params: AnalysisParams,
) -> AggregateStats:
    """Pool per-trace results into population-level blinking statistics.

    Args:
        results: Per-trace analysis results.
        traceset: The trace set the results came from (unused fields reserved
            for future spatial statistics).
        global_trace: Mean frame intensity versus frame index, for the transient
            fit.
        params: Analysis parameters (supplies the power-law cutoff).

    Returns:
        The :class:`AggregateStats`.
    """
    del traceset  # reserved for spatial-clustering statistics
    blinkers = [r for r in results if r.is_blinker]
    on = np.concatenate([r.on_times for r in blinkers]) if blinkers else np.empty(0)
    off = np.concatenate([r.off_times for r in blinkers]) if blinkers else np.empty(0)
    amplitudes = np.array([r.fit.amplitude for r in blinkers], dtype=np.float64)
    duty = np.array([r.duty_cycle for r in blinkers], dtype=np.float64)
    n_spots = len(results)
    return AggregateStats(
        n_spots=n_spots,
        n_blinkers=len(blinkers),
        blink_fraction=len(blinkers) / n_spots if n_spots else 0.0,
        on=_dwell_stats(on, params.power_law_xmin_s),
        off=_dwell_stats(off, params.power_law_xmin_s),
        amplitude=summarize(amplitudes),
        duty_cycle=summarize(duty),
        transient=fit_transient(global_trace, params.frame_interval_s),
    )


def combine(per_stack_results: list[list[TraceResult]], params: AnalysisParams) -> CombinedStats:
    """Pool per-trace results from many stacks into one combined summary.

    Every blinker from every stack contributes to the pooled dwell-time,
    amplitude and duty-cycle statistics.

    Args:
        per_stack_results: One list of per-trace results per stack.
        params: Analysis parameters (supplies the power-law cutoff).

    Returns:
        The :class:`CombinedStats`.
    """
    blinkers = [r for stack in per_stack_results for r in stack if r.is_blinker]
    n_spots = sum(len(stack) for stack in per_stack_results)
    on = np.concatenate([r.on_times for r in blinkers]) if blinkers else np.empty(0)
    off = np.concatenate([r.off_times for r in blinkers]) if blinkers else np.empty(0)
    amplitudes = np.array([r.fit.amplitude for r in blinkers], dtype=np.float64)
    duty = np.array([r.duty_cycle for r in blinkers], dtype=np.float64)
    return CombinedStats(
        n_stacks=len(per_stack_results),
        n_spots=n_spots,
        n_blinkers=len(blinkers),
        blink_fraction=len(blinkers) / n_spots if n_spots else 0.0,
        on=_dwell_stats(on, params.power_law_xmin_s),
        off=_dwell_stats(off, params.power_law_xmin_s),
        amplitude=summarize(amplitudes),
        duty_cycle=summarize(duty),
    )
