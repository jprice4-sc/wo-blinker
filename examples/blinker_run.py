from blinker import AnalysisParams, Blinker

# --- input / output ---
STACK_PATH = "/Users/jprice/Downloads/0002-20261019_110955_035.tif"
OUTPUT_DIR = "results_sensitive"

# --- time base ---
FRAME_INTERVAL_S = 0.01  # ImageJ metadata for testblink.tif says 100 fps

# --- detection (defaults: already finds ~6000 spots; "std" projection found fewer blinkers) ---
PROJECTION = "mean"
DETECTION_PERCENTILE = 60.0
MIN_SEPARATION = 6  # px between spots; reduce if emitters are densely packed
ROI_RADIUS = 2  # 5x5 box; keep matched to spot size (bigger adds background noise)

# --- segmentation / classification: keep short, low-contrast blinks ---
DEBOUNCE_FRAMES = 3  # default 5 erases 1-2 frame blinks; 3 keeps >= 2-frame blinks
SEP_THRESHOLD = 1.5  # default 2.0; pure noise scores ~1.35, so don't go lower
CONTRAST_THRESHOLD = 0.05  # default 0.1 was the main cut on subtle blinkers; noise is ~0.02
MIN_TRANSITIONS = 4  # fast blinkers switch often, so no need to lower this

params = AnalysisParams(
    frame_interval_s=FRAME_INTERVAL_S,
    projection=PROJECTION,
    detection_percentile=DETECTION_PERCENTILE,
    min_separation=MIN_SEPARATION,
    roi_radius=ROI_RADIUS,
    debounce_frames=DEBOUNCE_FRAMES,
    sep_threshold=SEP_THRESHOLD,
    contrast_threshold=CONTRAST_THRESHOLD,
    min_transitions=MIN_TRANSITIONS,
)

b = Blinker(params)
b.load(STACK_PATH).analyze()

print(b.summary())  # key metrics as text
on, off = b.dwell_times()  # pooled ON/OFF times, seconds
b.save(OUTPUT_DIR)  # tables + JSON + figures incl. summary.png
