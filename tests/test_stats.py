"""Tests for the aggregate statistics and distribution fits."""

from __future__ import annotations

import numpy as np

from blinker.stats import fit_exponential, fit_power_law, fit_transient, summarize


def test_summarize_empty() -> None:
    summary = summarize(np.empty(0))
    assert summary.n == 0
    assert np.isnan(summary.mean)


def test_summarize_basic() -> None:
    summary = summarize(np.array([1.0, 2.0, 3.0]))
    assert summary.n == 3
    assert summary.mean == 2.0
    assert summary.median == 2.0
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0


def test_fit_exponential_recovers_tau() -> None:
    rng = np.random.default_rng(0)
    tau_true = 4.0
    samples = rng.exponential(tau_true, size=20000)
    fit = fit_exponential(samples)
    assert fit is not None
    assert fit.tau == np.mean(samples)
    assert abs(fit.tau - tau_true) < 0.2


def test_fit_power_law_recovers_alpha() -> None:
    rng = np.random.default_rng(1)
    alpha_true = 2.5
    xmin = 1.0
    # Inverse-transform sampling for a continuous power law with exponent alpha.
    u = rng.random(50000)
    samples = xmin * (1.0 - u) ** (-1.0 / (alpha_true - 1.0))
    fit = fit_power_law(samples, xmin=xmin)
    assert fit is not None
    assert abs(fit.alpha - alpha_true) < 0.1


def test_fit_power_law_too_few_samples() -> None:
    assert fit_power_law(np.array([1.0, 2.0])) is None


def test_transient_none_when_flat() -> None:
    rng = np.random.default_rng(2)
    flat = 100.0 + rng.normal(0, 1.0, 300)
    assert fit_transient(flat, frame_interval_s=0.1) is None


def test_transient_recovers_time_constant() -> None:
    frame_interval = 0.1
    tau_true = 3.0
    t = np.arange(400) * frame_interval
    signal = 500.0 - 400.0 * np.exp(-t / tau_true)
    fit = fit_transient(signal, frame_interval_s=frame_interval)
    assert fit is not None
    assert abs(fit.tau - tau_true) < 0.5
    assert fit.beta > 0.5
