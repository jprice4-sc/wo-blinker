"""Aggregate blinking statistics pooled across all emitters in a stack.

Given the per-emitter results this module builds the population-level summary:
how many emitters are static / one-off steppers / stochastic blinkers, the
pooled ON/OFF dwell-time distributions (with power-law and exponential fits),
and the distributions of the emitter contrast, separation, duty cycle and
amplitude. Only stochastic blinkers contribute to the dwell-time and blinker
distributions; the coefficient-of-variation summary is over every detected spot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .params import AnalysisParams
    from .states import EmitterResult

_MIN_FIT_SAMPLES = 5


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
class AggregateStats:
    """Population-level blinking statistics for one stack.

    Attributes:
        n_spots: Number of detected emitters.
        n_static: Emitters classed as flat/static.
        n_stepper: Emitters with a clear but one-off (non-stochastic) switch.
        n_blinkers: Emitters classed as genuine stochastic blinkers.
        blink_fraction: ``n_blinkers / n_spots``.
        on: Pooled ON dwell-time statistics (blinkers only).
        off: Pooled OFF dwell-time statistics (blinkers only).
        amplitude: Distribution of blinker ON-OFF amplitudes.
        contrast: Distribution of blinker Michelson contrasts.
        separation: Distribution of blinker state separations (sigma).
        duty_cycle: Distribution of blinker ON fractions.
        cv: Distribution of relative fluctuation over every detected spot.
    """

    n_spots: int
    n_static: int
    n_stepper: int
    n_blinkers: int
    blink_fraction: float
    on: DwellStats
    off: DwellStats
    amplitude: Summary
    contrast: Summary
    separation: Summary
    duty_cycle: Summary
    cv: Summary


@dataclass(frozen=True, slots=True)
class CombinedStats:
    """Blinking statistics pooled across several stacks.

    Attributes:
        n_stacks: Number of stacks combined.
        n_spots: Total detected emitters across all stacks.
        n_blinkers: Total stochastic blinkers across all stacks.
        blink_fraction: ``n_blinkers / n_spots`` over the whole set.
        on: ON dwell-time statistics pooled across every blinker.
        off: OFF dwell-time statistics pooled across every blinker.
        amplitude: Distribution of blinker ON-OFF amplitudes, all stacks.
        contrast: Distribution of blinker contrasts, all stacks.
        duty_cycle: Distribution of blinker ON fractions, all stacks.
    """

    n_stacks: int
    n_spots: int
    n_blinkers: int
    blink_fraction: float
    on: DwellStats
    off: DwellStats
    amplitude: Summary
    contrast: Summary
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
    if times.size == 0:
        return None
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


def _pool_dwells(blinkers: list[EmitterResult]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Concatenate ON and OFF dwell times across a list of blinkers."""
    on = np.concatenate([r.on_times for r in blinkers]) if blinkers else np.empty(0)
    off = np.concatenate([r.off_times for r in blinkers]) if blinkers else np.empty(0)
    return on, off


def aggregate(results: list[EmitterResult], params: AnalysisParams) -> AggregateStats:
    """Pool per-emitter results into population-level blinking statistics.

    Args:
        results: Per-emitter analysis results.
        params: Analysis parameters (supplies the power-law cutoff).

    Returns:
        The :class:`AggregateStats`.
    """
    blinkers = [r for r in results if r.category == "stochastic"]
    on, off = _pool_dwells(blinkers)
    n_spots = len(results)
    return AggregateStats(
        n_spots=n_spots,
        n_static=sum(r.category == "static" for r in results),
        n_stepper=sum(r.category == "stepper" for r in results),
        n_blinkers=len(blinkers),
        blink_fraction=len(blinkers) / n_spots if n_spots else 0.0,
        on=_dwell_stats(on, params.power_law_xmin_s),
        off=_dwell_stats(off, params.power_law_xmin_s),
        amplitude=summarize(np.array([r.fit.amplitude for r in blinkers], dtype=np.float64)),
        contrast=summarize(np.array([r.fit.contrast for r in blinkers], dtype=np.float64)),
        separation=summarize(np.array([r.fit.sep for r in blinkers], dtype=np.float64)),
        duty_cycle=summarize(np.array([r.on_fraction for r in blinkers], dtype=np.float64)),
        cv=summarize(np.array([r.cv for r in results], dtype=np.float64)),
    )


def combine(per_stack_results: list[list[EmitterResult]], params: AnalysisParams) -> CombinedStats:
    """Pool per-emitter results from many stacks into one combined summary.

    Args:
        per_stack_results: One list of per-emitter results per stack.
        params: Analysis parameters (supplies the power-law cutoff).

    Returns:
        The :class:`CombinedStats`.
    """
    blinkers = [r for stack in per_stack_results for r in stack if r.category == "stochastic"]
    n_spots = sum(len(stack) for stack in per_stack_results)
    on, off = _pool_dwells(blinkers)
    return CombinedStats(
        n_stacks=len(per_stack_results),
        n_spots=n_spots,
        n_blinkers=len(blinkers),
        blink_fraction=len(blinkers) / n_spots if n_spots else 0.0,
        on=_dwell_stats(on, params.power_law_xmin_s),
        off=_dwell_stats(off, params.power_law_xmin_s),
        amplitude=summarize(np.array([r.fit.amplitude for r in blinkers], dtype=np.float64)),
        contrast=summarize(np.array([r.fit.contrast for r in blinkers], dtype=np.float64)),
        duty_cycle=summarize(np.array([r.on_fraction for r in blinkers], dtype=np.float64)),
    )
