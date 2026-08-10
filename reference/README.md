# reference/

Preserved verbatim, **not maintained or imported** by the package. Excluded from
`ruff` and `mypy` (see `ruff.toml` / `pyproject.toml`).

- `avi_microled_reference.py` — the original template script this repo grew from:
  a streaming **AVI** (OpenCV) blinking-analysis pipeline for a microLED panel,
  using a fixed block grid of sites and a mid-level threshold. The `blinker`
  package generalises it to **TIFF** time-series stacks with automatic spot
  detection and two-state (GMM) segmentation. Kept here for algorithmic
  reference (stretched-exponential transient, split-median step detection,
  Poisson trap density, nearest-neighbour clustering).

  It depends on `opencv-python`, which is **not** a dependency of the package.
