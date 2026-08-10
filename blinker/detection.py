"""Automatic emitter detection from a temporal projection of the stack.

Blinking emitters are the pixels whose intensity varies over time, so the
default detection image is the per-pixel temporal standard deviation. Local
maxima of that image are the candidate emitters; peaks too close to the frame
edge (where the background annulus would fall off the image) are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from skimage.feature import peak_local_max

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .loading import Stack
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


def project(stack: Stack, kind: Projection) -> NDArray[np.float64]:
    """Collapse a stack along time into a single 2-D detection image.

    Args:
        stack: The input time-series stack.
        kind: Projection to compute (``"std"``, ``"max"``, ``"mean"`` or
            ``"cv"`` for the coefficient of variation).

    Returns:
        A ``(height, width)`` float64 projection image.
    """
    data = stack.data
    if kind == "std":
        return np.std(data, axis=0, dtype=np.float64)
    if kind == "max":
        return np.max(data, axis=0).astype(np.float64)
    if kind == "mean":
        return np.mean(data, axis=0, dtype=np.float64)
    # kind == "cv": coefficient of variation.
    mean = np.mean(data, axis=0, dtype=np.float64)
    std = np.std(data, axis=0, dtype=np.float64)
    return std / (mean + 1.0)


def detect_spots(stack: Stack, params: AnalysisParams) -> Spots:
    """Locate candidate emitters as local maxima of a temporal projection.

    Args:
        stack: The input time-series stack.
        params: Analysis parameters controlling projection, threshold, minimum
            spot separation and border exclusion.

    Returns:
        The detected :class:`Spots`, ordered brightest-first in the projection.
    """
    projection = project(stack, params.projection)
    threshold = float(np.percentile(projection, params.detection_percentile))
    num_peaks = params.max_spots if params.max_spots is not None else np.inf
    coords = peak_local_max(
        projection,
        min_distance=params.min_separation,
        threshold_abs=threshold,
        exclude_border=params.background_outer,
        num_peaks=num_peaks,
    )
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    return Spots(coords=coords, projection=projection, projection_kind=params.projection)
