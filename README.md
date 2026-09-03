# wo-blinker

Simple blinking-statistics analysis of TIFF time-series stacks.

Given a multi-page TIFF (pages = frames) of emitters — single molecules, quantum
dots, defect / charge-trap sites, microLED pixels — `blinker` finds the
emitters, extracts a box-summed intensity trace for each, splits every trace
into ON/OFF states, classifies each emitter, and reports the blinking
statistics **per individual emitter** and for **the whole population**:

- **Per emitter** (`emitters.csv`): position, ON/OFF levels, relative
  fluctuation `cv`, ON/OFF contrast and separation, duty cycle, transition
  count, mean ON/OFF dwell, and a class — `static`, `stepper` (one-off switch)
  or `stochastic` (a genuine blinker).
- **Population** (`summary.json`): how many emitters are static / steppers /
  stochastic blinkers, and the pooled ON/OFF dwell-time distributions with
  power-law and exponential fits, plus the contrast / duty-cycle / `cv`
  distributions.

Large files are **streamed** frame-by-frame, so a multi-GB uint8 movie is
analyzed in a few hundred MB of RAM.

## Method

```
TIFF stack ──► detect spots ──► extract traces ──► two-state ──► aggregate stats
 loading.py     detection.py       traces.py        states.py       stats.py
```

| stage | module | what it does |
| --- | --- | --- |
| load | [loading.py](blinker/loading.py) | stream a TIFF as `(T, H, W)` float32 chunks (`open_stack`), or load it fully (`load_stack`) |
| detect | [detection.py](blinker/detection.py) | local maxima of a temporal projection (default: the time **mean**) |
| extract | [traces.py](blinker/traces.py) | sum a small square box (default 5×5) around each emitter, per frame |
| segment | [states.py](blinker/states.py) | Otsu threshold → ON/OFF, median-filter debounce, then classify each trace |
| aggregate | [stats.py](blinker/stats.py) | pool dwell times, fit power-law / exponential, summarize the population |
| orchestrate | [pipeline.py](blinker/pipeline.py) | run all stages and write CSV / JSON / PNG outputs |

An emitter is called a **stochastic blinker** when its two levels are genuinely
separated (`sep >= sep_threshold` *and* `contrast >= contrast_threshold`) and it
switches often enough (`n_transitions >= min_transitions`); a clearly two-state
trace that switches only once or twice is a **stepper**; everything else is
**static**. All parameters live in one immutable
[`AnalysisParams`](blinker/params.py).

## Install

```bash
uv sync
```

## Run script (edit constants, then run)

There is no command-line parser. [run_blinker.py](run_blinker.py) has a `CONFIG`
block at the top with the input/output paths and every analysis parameter as a
plainly visible constant — edit them and run:

```bash
uv run python run_blinker.py
```

`INPUT` may be a single `.tif`/`.tiff` **file** or a **folder** of them. The
constants map one-to-one onto [`AnalysisParams`](blinker/params.py); the
defaults reproduce the reference analysis (mean-projection detection, a 5×5 box,
Otsu + a 5-frame debounce):

```python
INPUT = Path("movie.tiff")  # a file, or a folder of stacks
OUTPUT_DIR = Path("blinker_out")
FRAME_INTERVAL_S = 1.0  # seconds per frame (1.0 => dwell times in frames)
PROJECTION = "mean"  # "mean" | "std" | "max" | "cv"
DETECTION_PERCENTILE = 60.0  # keep local maxima above this percentile of the projection
MIN_SEPARATION = 6  # min pixels between two emitters
ROI_RADIUS = 2  # box half-width (2 => 5×5)
DEBOUNCE_FRAMES = 5  # median-filter window (1 = off)
SEP_THRESHOLD = 2.0  # min level separation (sigma) to be two-state
CONTRAST_THRESHOLD = 0.1  # min contrast to be two-state
MIN_TRANSITIONS = 4  # min switches to be a stochastic blinker
```

> Sparse emitters that are dark most of the time detect better with
> `PROJECTION = "std"` and a higher `DETECTION_PERCENTILE` (e.g. 99). The `mean`
> default suits dense, always-present sites (e.g. an emitter lattice).

Outputs, written under `OUTPUT_DIR/<stack name>/`:

- `emitters.csv` — one row per emitter (the individual-emitter statistics).
- `summary.json` — the population statistics and the parameters used.
- `on_dwell_times.csv`, `off_dwell_times.csv` — pooled dwell times (blinkers only).
- `overview.png` — detection image with emitters marked by class, the strongest
  blinker traces, and the fluctuation / switching-activity histograms.
- `statistics.png` — pooled ON/OFF dwell distributions and their survival curves
  with the exponential and power-law fits (where heavy-tailed kinetics show up).

For a folder, two roll-ups are also written at the top of `OUTPUT_DIR`:
`batch_summary.csv` (one row per stack) and `combined_summary.json` (all stacks
pooled).

## Python API

```python
from blinker import AnalysisParams, analyze_file, save_results

params = AnalysisParams(frame_interval_s=1.0)  # defaults reproduce the reference analysis
result = analyze_file("stack.tiff", params)  # streamed from disk

print(result.stats.n_blinkers, "blinkers of", result.stats.n_spots, "spots")
print("ON dwell median:", result.stats.on.summary.median)
if result.stats.on.power_law:
    print("ON power-law exponent:", result.stats.on.power_law.alpha)

save_results(result, "results")
```

Work with individual stages directly for finer control:

```python
from blinker import open_stack, detect_spots, extract_traces, analyze_traces, aggregate

source = open_stack("stack.tiff", frame_interval_s=1.0)  # streaming source
spots = detect_spots(source, params)
traces = extract_traces(source, spots, params)
emitters = analyze_traces(traces, params)  # per-emitter EmitterResult list
stats = aggregate(emitters, params)  # population AggregateStats
```

## Example (no data required)

`blinker.synthetic` generates a ground-truth stack of telegraph-blinking Gaussian
emitters. The demo writes a synthetic TIFF and runs the full analysis on it:

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
- With `FRAME_INTERVAL_S = 1.0` dwell times are reported in **frames**; set it to
  `1/fps` to report them in seconds.
- `reference/avi_microled_reference.py` preserves the original streaming-AVI
  microLED script this package generalized from; it is not part of the package.
