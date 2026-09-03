"""Automatic emitter detection from a temporal projection of the stack.

The stack is collapsed along time into a single 2-D image and emitters are the
local maxima of that image. The default projection is the per-pixel time
**mean**, which shows every persistent emitter (the reference data is a dense
lattice of always-present sites that we then classify by how they fluctuate).
The projection is accumulated by streaming the stack in chunks, so it works on
files far larger than memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from skimage.feature import peak_local_max

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .loading import FrameSource
    from .params import AnalysisParams, Projection


@dataclass(frozen=True, slots=True)
class Spots:
    """Detected emitter locations and the image they were found in.

    Attributes:
        coords: Integer ``(n_spots, 2)`` array of ``(row, col)`` peak positions.
        projection: The 2-D projection image used for detection.
        projection_kind: Which temporal projection produced ``projection``.
    """

    coords: NDArray[np.int64]
    projection: NDArray[np.float64]
    projection_kind: Projection

    @property
    def n(self) -> int:
        """Number of detected spots."""
        return int(self.coords.shape[0])


def project(source: FrameSource, kind: Projection) -> NDArray[np.float64]:
    """Collapse a stack along time into a single 2-D detection image.

    Args:
        source: The input time-series source (in-memory or streamed).
        kind: Projection to compute (``"mean"``, ``"std"``, ``"max"`` or ``"cv"``
            for the coefficient of variation).

    Returns:
        A ``(height, width)`` float64 projection image.
    """
    h, w = source.height, source.width
    n = source.n_frames
    if kind == "max":
        acc = np.full((h, w), -np.inf, dtype=np.float64)
        for _start, block in source.iter_chunks():
            np.maximum(acc, block.max(axis=0), out=acc)
        return acc

    s1 = np.zeros((h, w), dtype=np.float64)
    s2 = np.zeros((h, w), dtype=np.float64) if kind in ("std", "cv") else None
    for _start, block in source.iter_chunks():
        s1 += block.sum(axis=0, dtype=np.float64)
        if s2 is not None:
            s2 += np.einsum("thw,thw->hw", block, block, dtype=np.float64)
    mean = s1 / n
    if kind == "mean":
        return mean
    variance = np.clip(s2 / n - mean * mean, 0, None)  # type: ignore[operator]
    std = np.sqrt(variance)
    if kind == "std":
        return std
    return std / (mean + 1.0)  # cv


def detect_spots(source: FrameSource, params: AnalysisParams) -> Spots:
    """Locate candidate emitters as local maxima of a temporal projection.

    Args:
        source: The input time-series source.
        params: Analysis parameters controlling projection, threshold, minimum
            spot separation and border exclusion.

    Returns:
        The detected :class:`Spots`, ordered brightest-first in the projection.
    """
    projection = project(source, params.projection)
    threshold = float(np.percentile(projection, params.detection_percentile))
    num_peaks = params.max_spots if params.max_spots is not None else np.inf
    coords = peak_local_max(
        projection,
        min_distance=params.min_separation,
        threshold_abs=threshold,
        exclude_border=params.roi_radius,
        num_peaks=num_peaks,
    )
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    return Spots(coords=coords, projection=projection, projection_kind=params.projection)
