"""Loading of TIFF time-series stacks into a normalized in-memory representation.

Microscopy TIFFs come in many shapes: a plain multi-page ``(T, H, W)`` stack, or
a hyperstack carrying singleton channel/z axes (``TZCYX`` and friends). This
module collapses whatever ``tifffile`` returns into a single ``(T, H, W)``
float32 array wrapped in a :class:`Stack`, so the rest of the pipeline never has
to reason about axis order or bit depth again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile

if TYPE_CHECKING:
    from numpy.typing import NDArray

MIN_FRAMES = 2
_TIME_SERIES_NDIM = 3


@dataclass(frozen=True, slots=True)
class Stack:
    """A time-series image stack with a physical time base.

    Attributes:
        data: Image data as a ``(n_frames, height, width)`` float32 array.
        frame_interval_s: Seconds between successive frames.
        name: Human-readable identifier, typically the source file stem.
        source_path: Path the data was read from, or ``None`` for in-memory data.
    """

    data: NDArray[np.float32]
    frame_interval_s: float
    name: str
    source_path: Path | None = None

    @property
    def n_frames(self) -> int:
        """Number of frames in the stack."""
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.data.shape[2])

    @property
    def duration_s(self) -> float:
        """Total acquisition duration in seconds."""
        return self.n_frames * self.frame_interval_s

    @property
    def times_s(self) -> NDArray[np.float64]:
        """Frame time stamps in seconds, starting at zero."""
        return np.arange(self.n_frames, dtype=np.float64) * self.frame_interval_s


def _to_time_series(array: NDArray[np.generic], *, name: str) -> NDArray[np.float32]:
    """Collapse an arbitrary TIFF array to a ``(T, H, W)`` float32 stack.

    Singleton axes (channel, z) are squeezed out. The largest remaining axis is
    assumed to be time and is moved to the front.

    Args:
        array: Raw array as returned by ``tifffile.imread``.
        name: Identifier used in error messages.

    Returns:
        A contiguous ``(n_frames, height, width)`` float32 array.

    Raises:
        ValueError: If the array cannot be interpreted as a time series of 2-D
            frames.
    """
    squeezed = np.squeeze(array)
    if squeezed.ndim == _TIME_SERIES_NDIM:
        stack = squeezed
    elif squeezed.ndim > _TIME_SERIES_NDIM:
        # Hyperstack with a real extra axis (e.g. multi-channel). Treat the
        # smallest leading axis as channels and keep the first one.
        lead = int(np.argmin(squeezed.shape[:-2]))
        stack = np.take(squeezed, indices=0, axis=lead)
        if stack.ndim != _TIME_SERIES_NDIM:
            raise ValueError(
                f"{name!r}: could not reduce shape {array.shape} to a (T, H, W) stack; "
                "load it manually and pass a 3-D array to Stack.from_array",
            )
    else:
        raise ValueError(
            f"{name!r}: expected a time series but got a single {squeezed.shape} frame",
        )

    if stack.shape[0] < MIN_FRAMES:
        raise ValueError(f"{name!r}: need at least {MIN_FRAMES} frames, got {stack.shape[0]}")
    return np.ascontiguousarray(stack, dtype=np.float32)


def load_stack(path: str | Path, frame_interval_s: float = 1.0) -> Stack:
    """Read a TIFF time-series stack from disk.

    Args:
        path: Path to a multi-page TIFF file.
        frame_interval_s: Seconds between successive frames.

    Returns:
        The loaded :class:`Stack`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be interpreted as a ``(T, H, W)`` stack.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = tifffile.imread(str(path))
    data = _to_time_series(np.asarray(raw), name=path.name)
    return Stack(data=data, frame_interval_s=frame_interval_s, name=path.stem, source_path=path)


def stack_from_array(
    array: NDArray[np.generic],
    frame_interval_s: float = 1.0,
    name: str = "array",
) -> Stack:
    """Wrap an in-memory ``(T, H, W)`` array as a :class:`Stack`.

    Args:
        array: A 3-D (or squeezable-to-3-D) image array.
        frame_interval_s: Seconds between successive frames.
        name: Identifier for the resulting stack.

    Returns:
        The wrapped :class:`Stack`.
    """
    data = _to_time_series(np.asarray(array), name=name)
    return Stack(data=data, frame_interval_s=frame_interval_s, name=name, source_path=None)
