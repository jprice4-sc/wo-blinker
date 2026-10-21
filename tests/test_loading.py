"""Tests for TIFF loading and stack normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from blinker.loading import load_stack, open_stack, stack_from_array

if TYPE_CHECKING:
    from pathlib import Path


def test_stack_from_3d_array() -> None:
    data = np.zeros((10, 8, 8), dtype=np.uint16)
    stack = stack_from_array(data, frame_interval_s=0.1)
    assert stack.n_frames == 10
    assert (stack.height, stack.width) == (8, 8)
    assert stack.data.dtype == np.float32
    assert stack.duration_s == pytest.approx(1.0)
    assert stack.times_s[1] == pytest.approx(0.1)


def test_singleton_axes_are_squeezed() -> None:
    # e.g. a (T, Z=1, Y, X) hyperstack collapses to (T, Y, X).
    data = np.zeros((6, 1, 8, 8), dtype=np.uint16)
    stack = stack_from_array(data)
    assert stack.data.shape == (6, 8, 8)


def test_single_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="single"):
        stack_from_array(np.zeros((8, 8)))


def test_single_frame_stack_is_rejected() -> None:
    # A (1, H, W) array is one frame once the singleton axis is squeezed.
    with pytest.raises(ValueError, match="single"):
        stack_from_array(np.zeros((1, 8, 8)))


def test_load_stack_roundtrip(tmp_path: Path) -> None:
    data = (np.arange(4 * 8 * 8).reshape(4, 8, 8) % 255).astype(np.uint16)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(str(path), data, photometric="minisblack")
    stack = load_stack(path, frame_interval_s=0.2)
    assert stack.n_frames == 4
    assert stack.name == "movie"
    assert stack.source_path == path
    np.testing.assert_allclose(stack.data, data.astype(np.float32))


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_stack("/no/such/file.tif")


def _stream(path: Path) -> np.ndarray:
    stack = open_stack(path)
    return np.concatenate([block for _, block in stack.iter_chunks(chunk=3)])


@pytest.mark.parametrize("writer", ["tifffile", "imagej"])
def test_open_stack_streams_mono(tmp_path: Path, writer: str) -> None:
    data = (np.arange(7 * 8 * 8).reshape(7, 8, 8) % 255).astype(np.uint8)
    path = tmp_path / "mono.tif"
    tifffile.imwrite(str(path), data, imagej=writer == "imagej")
    assert open_stack(path).shape == (7, 8, 8)
    np.testing.assert_array_equal(_stream(path), data.astype(np.float32))


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_open_stack_keeps_first_rgb_channel(tmp_path: Path, compression: str | None) -> None:
    # Mono cameras sometimes save as RGB; contiguous files are memory-mapped,
    # compressed ones fall back to page-by-page reading.
    data = (np.arange(7 * 8 * 8 * 3).reshape(7, 8, 8, 3) % 255).astype(np.uint8)
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(str(path), data, photometric="rgb", compression=compression)
    stack = open_stack(path)
    assert stack.shape == (7, 8, 8)
    assert stack.channel == 0
    assert stack.contiguous is (compression is None)
    np.testing.assert_array_equal(_stream(path), data[..., 0].astype(np.float32))
