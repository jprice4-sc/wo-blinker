# wo-blinker

Systematic blinking-statistics analysis of TIFF time-series stacks.

Given a multi-page TIFF (pages = frames) of blinking emitters — single molecules,
quantum dots, defect centres, microLED sites — `blinker` automatically finds the
emitters, extracts a background-subtracted intensity trace for each, segments
every trace into ON/OFF states, and reports the blinking statistics:

- **ON / OFF dwell times** — per emitter and pooled, with power-law and
  exponential fits of the distributions.
- **Intensity amplitudes** — the ON−OFF step of each emitter.
- **Duty cycle** — fraction of time each emitter spends ON.
- **Switching activity** — number of transitions per emitter.
- **Photo-activation transient** — a stretched-exponential fit of the field-wide
  mean intensity versus time.

## Pipeline

```
TIFF stack ──► detect spots ──► extract traces ──► two-state ──► aggregate stats
 loading.py     detection.py       traces.py        states.py       stats.py
```

| stage | module | what it does |
| --- | --- | --- |
| load | [loading.py](blinker/loading.py) | read a TIFF into a normalised `(T, H, W)` float32 `Stack` |
| detect | [detection.py](blinker/detection.py) | local-maxima peak finding on a temporal projection (default: per-pixel std) |
| extract | [traces.py](blinker/traces.py) | sum a circular aperture per emitter, subtract a per-frame background annulus |
| segment | [states.py](blinker/states.py) | 1-D two-state clustering (GMM / k-means / Otsu) → threshold → ON/OFF dwell runs |
| aggregate | [stats.py](blinker/stats.py) | pool dwell times, fit distributions, summarise amplitudes / duty cycle / transient |
| orchestrate | [pipeline.py](blinker/pipeline.py) | run all stages and write CSV / JSON / PNG outputs |

Analysis parameters live in a single immutable [`AnalysisParams`](blinker/params.py).

## Install

```bash
uv sync
```

## Run script (edit constants, then run)

There is no command-line parser. [run_blinker.py](run_blinker.py) has a `CONFIG`
block at the top with the input/output directories and every analysis parameter
as a plainly visible constant — edit them and run:

```bash
uv run python run_blinker.py
```

It analyses every `*.tif` / `*.tiff` in `INPUT_DIR` with the same parameters. The
constants map one-to-one onto [`AnalysisParams`](blinker/params.py):

```python
INPUT_DIR  = Path("data")
OUTPUT_DIR = Path("blinker_out")
FRAME_INTERVAL_S = 0.05        # seconds per frame (use 1/fps)
PROJECTION = "std"             # "std" | "max" | "mean" | "cv"
DETECTION_PERCENTILE = 99.0    # keep peaks above this percentile of the projection
ROI_RADIUS = 3                 # aperture radius (px); BACKGROUND_INNER/OUTER = annulus
STATE_METHOD = "gmm"           # "gmm" | "kmeans" | "otsu"
MIN_SNR = 3.0                  # amplitude/noise a trace must exceed to be a blinker
# ... (see the file for the full list)
```

Outputs, written under `OUTPUT_DIR`:

- `<stack name>/emitters.csv` — one row per emitter (position, levels, amplitude,
  SNR, transitions, duty cycle, mean ON/OFF times, blinker flag).
- `<stack name>/on_dwell_times.csv`, `off_dwell_times.csv` — pooled dwell times (s).
- `<stack name>/summary.json` — aggregate statistics and the parameters used.
- `<stack name>/overview.png`, `statistics.png` — diagnostic figures.
- `batch_summary.csv` — one row per stack (blinker counts, medians, transient τ).

## Python API

```python
from blinker import AnalysisParams, analyze_file, save_results

params = AnalysisParams(frame_interval_s=0.05, state_method="gmm", min_snr=3.0)
result = analyze_file("stack.tif", params)

print(result.stats.n_blinkers, "blinkers of", result.stats.n_spots, "spots")
print("ON dwell median:", result.stats.on.summary.median, "s")
if result.stats.on.power_law:
    print("ON power-law exponent:", result.stats.on.power_law.alpha)

save_results(result, "results")
```

Analyse a whole folder in one call (this is what the run script uses):

```python
from blinker import AnalysisParams, analyze_directory

analyze_directory("data", "blinker_out", AnalysisParams(frame_interval_s=0.05))
```

Work with individual stages directly when you need finer control:

```python
from blinker import load_stack, detect_spots, extract_traces, analyze_traces

stack = load_stack("stack.tif", frame_interval_s=0.05)
spots = detect_spots(stack, params)
traces = extract_traces(stack, spots, params)
results = analyze_traces(traces, params)  # per-emitter TraceResult list
```

## Demo (no data required)

`blinker.synthetic` generates a ground-truth stack of telegraph-blinking Gaussian
emitters (optionally with a photo-activation transient). The example writes a
synthetic TIFF and runs the full analysis on it:

```bash
uv run python examples/analyze_synthetic.py
```

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # type-check (basedmypy)
uv run pytest              # tests (synthetic ground-truth, end-to-end + units)
uv run pydoctor            # API docs -> site/
```

## Notes

- The reported ON/OFF dwell times **exclude the first and last run** of each
  trace, which are censored (the true switch happened outside the record).
- `reference/avi_microled_reference.py` preserves the original streaming-AVI
  microLED script this package generalised from; it is not part of the package.
