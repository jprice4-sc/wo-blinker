"""Per-emitter intensity trace extraction with local background subtraction.

For every detected spot we sum the pixels inside a small circular aperture to
form a raw intensity trace, then subtract a per-frame background estimated from a
surrounding annulus. Estimating background per frame (rather than once) keeps
slow field-wide drift and photo-activation transients out of the emitter traces,
so a trace's fluctuations reflect the emitter itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .detection import Spots
    from .loading import Stack
    from .params import AnalysisParams


@dataclass(frozen=True, slots=True)
class TraceSet:
    """Background-subtracted intensity traces for a set of emitters.

    Attributes:
        traces: ``(n_spots, n_frames)`` background-subtracted integrated
            intensity, in raw detector counts summed over the aperture.
        raw: ``(n_spots, n_frames)`` aperture sum before background subtraction.
        background: ``(n_spots, n_frames)`` per-pixel background level.
        coords: ``(n_spots, 2)`` integer ``(row, col)`` emitter positions.
        frame_interval_s: Seconds between successive frames.
        aperture_area: Number of pixels in the summing aperture.
    """

    traces: NDArray[np.float64]
    raw: NDArray[np.float64]
    background: NDArray[np.float64]
    coords: NDArray[np.int64]
    frame_interval_s: float
    aperture_area: int

    @property
    def n_spots(self) -> int:
        """Number of emitters."""
        return int(self.traces.shape[0])

    @property
    def n_frames(self) -> int:
        """Number of frames per trace."""
        return int(self.traces.shape[1])


def _disk_offsets(radius: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Row/column offsets of pixels within ``radius`` of the origin."""
    span = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(span, span, indexing="ij")
    mask = dr * dr + dc * dc <= radius * radius
    return dr[mask].astype(np.int64), dc[mask].astype(np.int64)


def _annulus_offsets(inner: int, outer: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Row/column offsets of pixels in the annulus ``inner < r <= outer``."""
    span = np.arange(-outer, outer + 1)
    dr, dc = np.meshgrid(span, span, indexing="ij")
    rsq = dr * dr + dc * dc
    mask = (rsq > inner * inner) & (rsq <= outer * outer)
    return dr[mask].astype(np.int64), dc[mask].astype(np.int64)


def extract_traces(stack: Stack, spots: Spots, params: AnalysisParams) -> TraceSet:
    """Extract a background-subtracted trace for every detected spot.

    Args:
        stack: The input time-series stack.
        spots: Detected emitter locations.
        params: Analysis parameters controlling aperture and annulus geometry.

    Returns:
        The extracted :class:`TraceSet`. Row ``i`` corresponds to
        ``spots.coords[i]``.
    """
    data = stack.data
    ap_dr, ap_dc = _disk_offsets(params.roi_radius)
    an_dr, an_dc = _annulus_offsets(params.background_inner, params.background_outer)
    aperture_area = int(ap_dr.size)

    n_spots = spots.n
    n_frames = stack.n_frames
    raw = np.empty((n_spots, n_frames), dtype=np.float64)
    background = np.empty((n_spots, n_frames), dtype=np.float64)

    for i, (row, col) in enumerate(spots.coords):
        ap_pixels = data[:, row + ap_dr, col + ap_dc]  # (n_frames, aperture_area)
        an_pixels = data[:, row + an_dr, col + an_dc]  # (n_frames, annulus_area)
        raw[i] = ap_pixels.sum(axis=1)
        background[i] = np.median(an_pixels, axis=1)

    traces = raw - aperture_area * background
    return TraceSet(
        traces=traces,
        raw=raw,
        background=background,
        coords=spots.coords,
        frame_interval_s=stack.frame_interval_s,
        aperture_area=aperture_area,
    )
