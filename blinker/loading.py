"""Loading of TIFF time-series stacks, in-memory or streamed frame-by-frame.

Microscopy TIFFs come in many shapes: a plain multi-page ``(T, H, W)`` stack, or
a hyperstack carrying singleton channel/z axes. Everything downstream consumes a
:class:`FrameSource` -- an object that knows its ``(n_frames, height, width)``
shape and can yield the stack in float32 chunks along time. Two implementations
share that interface:

* :class:`Stack` holds the whole stack in memory (handy for small data, tests
  and synthetic stacks).
* :class:`TiffStack` streams a file page-by-page and never materializes the
  whole stack, so a multi-GB uint8 movie is analyzed in a few hundred MB.

Use :func:`open_stack` to get a streaming source from a path and
:func:`load_stack` when you specifically want the data in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import tifffile

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

MIN_FRAMES = 2
_TIME_SERIES_NDIM = 3
_RGB_SAMPLES = (3, 4)
DEFAULT_CHUNK = 200


@runtime_checkable
class FrameSource(Protocol):
    """A time-series image source that yields float32 frames in time-chunks.

    Both :class:`Stack` (in memory) and :class:`TiffStack` (streamed) satisfy
    this protocol, so detection and trace extraction work the same on either.
    """

    @property
    def frame_interval_s(self) -> float:
        """Seconds between successive frames."""
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier for the stack."""
        ...

    @property
    def source_path(self) -> Path | None:
        """Path the data came from, or ``None`` for in-memory data."""
        ...

    @property
    def n_frames(self) -> int:
        """Number of frames in the stack."""
        ...

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        ...

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        ...

    def iter_chunks(self, chunk: int = DEFAULT_CHUNK) -> Iterator[tuple[int, NDArray[np.float32]]]:
        """Yield ``(start_index, block)`` float32 slices of the stack along time."""
        ...


@dataclass(frozen=True, slots=True)
class Stack:
    """A time-series image stack held in memory.

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

    def iter_chunks(self, chunk: int = DEFAULT_CHUNK) -> Iterator[tuple[int, NDArray[np.float32]]]:
        """Yield ``(start_index, block)`` slices of the stack along time."""
        for start in range(0, self.n_frames, chunk):
            yield start, self.data[start : start + chunk]


@dataclass(frozen=True, slots=True)
class TiffStack:
    """A TIFF time-series stack streamed from disk one chunk at a time.

    The file is opened lazily on each :meth:`iter_chunks` pass; the frames are
    read as needed and returned as float32, so the whole stack is never resident
    in memory at once. Uncompressed, contiguous files (including ImageJ
    hyperstacks larger than 4 GB, which hold a single TIFF page) are
    memory-mapped; anything else is read page by page.

    Attributes:
        source_path: Path to the multi-page TIFF.
        frame_interval_s: Seconds between successive frames.
        name: Human-readable identifier (the file stem).
        shape: ``(n_frames, height, width)``.
        channel: Colour sample kept from RGB(A) frames, or ``None`` for mono.
        contiguous: Whether the image data can be memory-mapped.
    """

    source_path: Path
    frame_interval_s: float
    name: str
    shape: tuple[int, int, int]
    channel: int | None = None
    contiguous: bool = False

    @property
    def n_frames(self) -> int:
        """Number of frames in the stack."""
        return self.shape[0]

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self.shape[1]

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self.shape[2]

    def iter_chunks(self, chunk: int = DEFAULT_CHUNK) -> Iterator[tuple[int, NDArray[np.float32]]]:
        """Yield ``(start_index, block)`` float32 slices, reading frames on demand."""
        n, h, w = self.shape
        if self.contiguous:
            mapped = tifffile.memmap(str(self.source_path), mode="r")
            for start in range(0, n, chunk):
                raw = mapped[start : start + chunk]
                if self.channel is not None:
                    raw = raw[..., self.channel]
                yield start, raw.astype(np.float32)
            return
        with tifffile.TiffFile(str(self.source_path)) as tif:
            pages = tif.pages
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                block = np.empty((stop - start, h, w), dtype=np.float32)
                for k in range(start, stop):
                    page = pages[k].asarray()
                    block[k - start] = page[..., self.channel] if self.channel is not None else page
                yield start, block


def open_stack(path: str | Path, frame_interval_s: float = 1.0) -> TiffStack:
    """Open a multi-page TIFF as a streaming :class:`TiffStack` (no full load).

    RGB(A) frames (``(T, H, W, 3|4)``, as written by some mono cameras) are
    reduced to their first colour channel while streaming.

    Args:
        path: Path to a ``(T, H, W)`` or ``(T, H, W, 3|4)`` TIFF stack.
        frame_interval_s: Seconds between successive frames.

    Returns:
        A :class:`TiffStack` streaming source.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a time series of 2-D (mono or RGB)
            frames with at least :data:`MIN_FRAMES` frames.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        shape = tuple(int(s) for s in series.shape)
        axes = series.axes
        contiguous = series.dataoffset is not None
    channel = None
    if len(shape) == _TIME_SERIES_NDIM + 1 and axes.endswith("S") and shape[-1] in _RGB_SAMPLES:
        channel = 0
        shape = shape[:-1]
    if len(shape) != _TIME_SERIES_NDIM:
        raise ValueError(
            f"{path.name!r}: expected a (T, H, W) or (T, H, W, 3) series, got shape {shape} ({axes}); "
            "load it with load_stack() and pass the array to stack_from_array() instead",
        )
    if shape[0] < MIN_FRAMES:
        raise ValueError(f"{path.name!r}: need at least {MIN_FRAMES} frames, got {shape[0]}")
    return TiffStack(
        source_path=path,
        frame_interval_s=frame_interval_s,
        name=path.stem,
        shape=shape,  # type: ignore[arg-type]
        channel=channel,
        contiguous=contiguous,
    )


def _to_time_series(array: NDArray[np.generic], *, name: str) -> NDArray[np.float32]:
    """Collapse an arbitrary TIFF array to a ``(T, H, W)`` float32 stack.

    Singleton axes (channel, z) are squeezed out. For a real extra axis (e.g.
    multi-channel) the smallest leading axis is treated as channels and the
    first one is kept.

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
        lead = int(np.argmin(squeezed.shape[:-2]))
        stack = np.take(squeezed, indices=0, axis=lead)
        if stack.ndim != _TIME_SERIES_NDIM:
            raise ValueError(
                f"{name!r}: could not reduce shape {array.shape} to a (T, H, W) stack; "
                "load it manually and pass a 3-D array to stack_from_array",
            )
    else:
        raise ValueError(
            f"{name!r}: expected a time series but got a single {squeezed.shape} frame",
        )

    if stack.shape[0] < MIN_FRAMES:
        raise ValueError(f"{name!r}: need at least {MIN_FRAMES} frames, got {stack.shape[0]}")
    return np.ascontiguousarray(stack, dtype=np.float32)


def load_stack(path: str | Path, frame_interval_s: float = 1.0) -> Stack:
    """Read a whole TIFF time-series stack into memory.

    Prefer :func:`open_stack` for large files; this loads the entire stack as
    float32 at once.

    Args:
        path: Path to a multi-page TIFF file.
        frame_interval_s: Seconds between successive frames.

    Returns:
        The loaded in-memory :class:`Stack`.

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
