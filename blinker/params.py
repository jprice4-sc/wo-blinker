"""Analysis parameters for the blinking-statistics pipeline.

A single immutable :class:`AnalysisParams` bundles every tunable knob used across
detection, trace extraction, two-state segmentation and the aggregate
statistics. Passing one object (rather than a long argument list) keeps every
stage's signature small and makes a run fully reproducible from the recorded
parameters.

The defaults reproduce the reference analysis: emitters are found as local
maxima of the **time-mean** image, each trace is the sum of a small square box
around the emitter, and every trace is split into ON/OFF with Otsu's threshold
followed by a median-filter debounce. A trace is called a *blinker* when its two
levels are genuinely separated (``sep`` and ``contrast`` cuts) and it switches
often enough (``min_transitions``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Projection = Literal["mean", "std", "max", "cv"]
"""Temporal projection used to build the spot-detection image.

``mean`` (per-pixel time average) is the default: it finds every persistent
emitter, blinking or not, which is what we then classify. Use ``std`` or ``max``
instead when emitters are dark most of the time and only the fluctuating pixels
should be detected.
"""


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    """Immutable configuration for a single-stack analysis.

    Attributes:
        frame_interval_s: Time between successive frames, in seconds. Dwell
            times are reported using this value; leave it at ``1.0`` to report
            dwell times in frames.
        projection: Temporal projection used to locate emitters.
        detection_percentile: Local maxima brighter than this percentile of the
            projection image are kept as candidate emitters.
        min_separation: Minimum allowed distance, in pixels, between two detected
            spots (peak-finding exclusion radius).
        roi_radius: Half-width, in pixels, of the square box summed to form each
            emitter's trace. ``roi_radius=2`` sums a 5x5 box.
        max_spots: Optional cap on the number of emitters kept, ranked by
            projection brightness. ``None`` keeps all of them.
        debounce_frames: Median-filter window (frames) applied to the ON/OFF
            state before counting transitions and dwell times, to remove
            single-frame threshold flicker. ``1`` disables debouncing.
        sep_threshold: Minimum separation of the two levels, in units of the
            pooled within-state standard deviation, for a trace to be considered
            genuinely two-state (bimodal).
        contrast_threshold: Minimum Michelson contrast ``(on-off)/(on+off)`` for
            a trace to be considered genuinely two-state.
        min_transitions: Minimum number of debounced ON/OFF switches for a
            bimodal trace to be classed as a stochastic blinker (fewer switches
            with a clear step is classed as a one-off ``stepper``).
        power_law_xmin_s: Lower cutoff (seconds) for the dwell-time power-law
            fit. ``None`` uses the smallest observed dwell time.
    """

    # --- time base ---
    frame_interval_s: float = 1.0

    # --- spot detection ---
    projection: Projection = "mean"
    detection_percentile: float = 60.0
    min_separation: int = 2
    roi_radius: int = 2
    max_spots: int | None = None

    # --- two-state segmentation & classification ---
    debounce_frames: int = 3
    sep_threshold: float = 2.0
    contrast_threshold: float = 0.1
    min_transitions: int = 4

    # --- aggregate statistics ---
    power_law_xmin_s: float | None = None

    def __post_init__(self) -> None:
        """Validate physically meaningful bounds on the parameters.

        Raises:
            ValueError: If a parameter is outside its allowed range.
        """
        if self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")
        if not 0 < self.detection_percentile < 100:  # noqa: PLR2004
            raise ValueError("detection_percentile must be in (0, 100)")
        if self.roi_radius < 1:
            raise ValueError("roi_radius must be >= 1")
        if self.min_separation < 1:
            raise ValueError("min_separation must be >= 1")
        if self.debounce_frames < 1:
            raise ValueError("debounce_frames must be >= 1")
        if self.sep_threshold < 0 or self.contrast_threshold < 0:
            raise ValueError("sep_threshold and contrast_threshold must be >= 0")
        if self.min_transitions < 1:
            raise ValueError("min_transitions must be >= 1")
        if self.max_spots is not None and self.max_spots < 1:
            raise ValueError("max_spots must be >= 1 or None")

    @property
    def fps(self) -> float:
        """Frame rate in frames per second (reciprocal of the frame interval)."""
        return 1.0 / self.frame_interval_s
