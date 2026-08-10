"""Tests for TIFF loading and stack normalisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from blinker.loading import load_stack, stack_from_array

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
