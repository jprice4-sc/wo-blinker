"""Synthetic blinking-stack generation for tests, examples and calibration.

Produces a TIFF-like ``(T, H, W)`` stack of Gaussian emitters that switch
between an ON and OFF level as independent two-state (telegraph) processes, on
top of a constant background with Poisson shot noise and Gaussian read noise.
Because the ground-truth positions and state sequences are returned alongside
the data, the same generator drives the end-to-end tests and the demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

Layout = Literal["random", "grid"]


@dataclass(frozen=True, slots=True)
class SimParams:
    """Parameters of a synthetic blinking stack.

    Attributes:
        height: Frame height in pixels.
        width: Frame width in pixels.
        n_frames: Number of frames.
        n_emitters: Number of blinking emitters.
        layout: ``"grid"`` places emitters on an evenly spaced lattice (no
            overlap, deterministic positions); ``"random"`` scatters them.
        psf_sigma: Gaussian point-spread-function width in pixels.
        on_amplitude: Peak emitter counts added while ON.
        background: Constant background level in counts.
        mean_on_frames: Mean ON dwell time in frames.
        mean_off_frames: Mean OFF dwell time in frames.
        frame_interval_s: Seconds per frame.
        read_noise: Gaussian read-noise standard deviation in counts.
        margin: Keep-out border, in pixels, so emitters stay clear of the edge.
        activation_amplitude: Extra background rise (counts) that decays in as a
            stretched exponential, simulating photo-activation. ``0`` disables it.
        activation_tau_frames: Time constant of the activation transient, frames.
    """

    height: int = 64
    width: int = 64
    n_frames: int = 400
    n_emitters: int = 8
    layout: Layout = "random"
    psf_sigma: float = 1.5
    on_amplitude: float = 900.0
    background: float = 100.0
    mean_on_frames: float = 18.0
    mean_off_frames: float = 28.0
    frame_interval_s: float = 0.05
    read_noise: float = 5.0
    margin: int = 10
    activation_amplitude: float = 0.0
    activation_tau_frames: float = 50.0


@dataclass(frozen=True, slots=True)
class SimulatedStack:
    """A synthetic stack together with its ground truth.

    Attributes:
        data: ``(n_frames, height, width)`` float32 image data.
        positions: ``(n_emitters, 2)`` integer ``(row, col)`` emitter centers.
        states: ``(n_emitters, n_frames)`` boolean ON/OFF ground truth.
        params: The parameters used to generate the stack.
    """

    data: NDArray[np.float32]
    positions: NDArray[np.int64]
    states: NDArray[np.bool_]
    params: SimParams = field(default_factory=SimParams)

    @property
    def frame_interval_s(self) -> float:
        """Seconds per frame."""
        return self.params.frame_interval_s


def _telegraph_states(rng: np.random.Generator, params: SimParams) -> NDArray[np.bool_]:
    """Generate independent two-state Markov sequences for each emitter."""
    p_on_off = 1.0 / params.mean_on_frames
    p_off_on = 1.0 / params.mean_off_frames
    states = np.zeros((params.n_emitters, params.n_frames), dtype=bool)
    states[:, 0] = rng.random(params.n_emitters) < 0.5  # noqa: PLR2004
    flips = rng.random((params.n_emitters, params.n_frames))
    for t in range(1, params.n_frames):
        prev = states[:, t - 1]
        switch = np.where(prev, flips[:, t] < p_on_off, flips[:, t] < p_off_on)
        states[:, t] = np.where(switch, ~prev, prev)
    return states


def _emitter_positions(rng: np.random.Generator, params: SimParams) -> NDArray[np.int64]:
    """Place emitter centers either on a lattice or at random within the margin."""
    lo_r, hi_r = params.margin, params.height - params.margin
    lo_c, hi_c = params.margin, params.width - params.margin
    if params.layout == "grid":
        ncols = int(np.ceil(np.sqrt(params.n_emitters)))
        nrows = int(np.ceil(params.n_emitters / ncols))
        rows = np.linspace(lo_r, hi_r, nrows).round().astype(np.int64)
        cols = np.linspace(lo_c, hi_c, ncols).round().astype(np.int64)
        grid = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(-1, 2)
        return grid[: params.n_emitters].astype(np.int64)
    rows = rng.integers(lo_r, hi_r, params.n_emitters)
    cols = rng.integers(lo_c, hi_c, params.n_emitters)
    return np.stack([rows, cols], axis=1).astype(np.int64)


def simulate_blinking_stack(params: SimParams | None = None, seed: int = 0) -> SimulatedStack:
    """Generate a synthetic stack of blinking Gaussian emitters.

    Args:
        params: Simulation parameters. Defaults to :class:`SimParams`.
        seed: Seed for the random number generator.

    Returns:
        The :class:`SimulatedStack`, including ground-truth positions and states.
    """
    params = params or SimParams()
    rng = np.random.default_rng(seed)
    positions = _emitter_positions(rng, params)
    states = _telegraph_states(rng, params)

    yy, xx = np.mgrid[0 : params.height, 0 : params.width]
    background = np.full(params.n_frames, params.background, dtype=np.float64)
    if params.activation_amplitude:
        frames = np.arange(params.n_frames, dtype=np.float64)
        background += params.activation_amplitude * (1.0 - np.exp(-frames / params.activation_tau_frames))
    clean = np.empty((params.n_frames, params.height, params.width), dtype=np.float64)
    clean[:] = background[:, None, None]
    two_sigma_sq = 2.0 * params.psf_sigma**2
    for e in range(params.n_emitters):
        row, col = positions[e]
        psf = params.on_amplitude * np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / two_sigma_sq)
        clean += states[e, :, None, None] * psf[None, :, :]

    noisy = rng.poisson(np.clip(clean, 0, None)).astype(np.float64)
    noisy += rng.normal(0.0, params.read_noise, size=noisy.shape)
    data = np.clip(noisy, 0, None).astype(np.float32)
    return SimulatedStack(data=data, positions=positions, states=states, params=params)
