"""Per-emitter intensity trace extraction.

For every detected spot we sum the pixels inside a small square box centered on
the emitter to form its intensity trace -- the same simple aperture used in the
reference analysis (a 5x5 box for ``roi_radius=2``). No background annulus is
subtracted: the per-trace Otsu threshold in :mod:`blinker.states` adapts to each
emitter's own baseline. Traces are built by streaming the stack in chunks, so a
multi-GB file is handled without loading it all at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .detection import Spots
    from .loading import FrameSource
    from .params import AnalysisParams


@dataclass(frozen=True, slots=True)
class TraceSet:
    """Integrated intensity traces for a set of emitters.

    Attributes:
        traces: ``(n_spots, n_frames)`` box-summed intensity, in detector counts.
        coords: ``(n_spots, 2)`` integer ``(row, col)`` emitter positions.
        frame_interval_s: Seconds between successive frames.
        roi_area: Number of pixels summed per emitter (the box area).
    """

    traces: NDArray[np.float64]
    coords: NDArray[np.int64]
    frame_interval_s: float
    roi_area: int

    @property
    def n_spots(self) -> int:
        """Number of emitters."""
        return int(self.traces.shape[0])

    @property
    def n_frames(self) -> int:
        """Number of frames per trace."""
        return int(self.traces.shape[1])


def extract_traces(source: FrameSource, spots: Spots, params: AnalysisParams) -> TraceSet:
    """Extract a box-summed intensity trace for every detected spot.

    Args:
        source: The input time-series source (in-memory or streamed).
        spots: Detected emitter locations.
        params: Analysis parameters (supplies the box half-width ``roi_radius``).

    Returns:
        The extracted :class:`TraceSet`. Row ``i`` corresponds to
        ``spots.coords[i]``.
    """
    r = params.roi_radius
    h, w = source.height, source.width
    ys = spots.coords[:, 0]
    xs = spots.coords[:, 1]
    n_spots = spots.n

    # (n_spots, box) gather indices for each emitter's square box, clipped to
    # the frame so a spot near the edge still yields a (smaller) valid sum.
    offs = np.arange(-r, r + 1)
    drow, dcol = np.meshgrid(offs, offs, indexing="ij")
    rows = np.clip(ys[:, None] + drow.ravel()[None, :], 0, h - 1)
    cols = np.clip(xs[:, None] + dcol.ravel()[None, :], 0, w - 1)
    box_area = int(drow.size)

    traces = np.empty((n_spots, source.n_frames), dtype=np.float64)
    for start, block in source.iter_chunks():
        gathered = block[:, rows, cols]  # (chunk, n_spots, box_area)
        traces[:, start : start + block.shape[0]] = gathered.sum(axis=2).T
    return TraceSet(
        traces=traces,
        coords=spots.coords,
        frame_interval_s=source.frame_interval_s,
        roi_area=box_area,
    )
