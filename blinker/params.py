"""Analysis parameters for the blinking-statistics pipeline.

A single immutable :class:`AnalysisParams` bundles every tunable knob used across
loading, spot detection, trace extraction, two-state segmentation and the
aggregate statistics. Passing one object (rather than a long argument list)
keeps every stage's signature small and makes a run fully reproducible from the
recorded parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Projection = Literal["std", "max", "mean", "cv"]
"""Temporal projection used to build the spot-detection image.

``std`` (per-pixel standard deviation over time) is the default because blinking
emitters are, by definition, the pixels that vary the most.
"""

StateMethod = Literal["gmm", "kmeans", "otsu"]
"""Method used to split a single intensity trace into two levels."""


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    """Immutable configuration for a single-stack analysis.

    Attributes:
        frame_interval_s: Time between successive frames, in seconds. All dwell
            times are reported in seconds using this value.
        projection: Temporal projection used to locate emitters.
        detection_percentile: Pixels above this percentile of the projection
            image are treated as candidate emitter peaks.
        min_separation: Minimum allowed distance, in pixels, between two detected
            spots (peak-finding exclusion radius).
        roi_radius: Radius, in pixels, of the circular aperture whose pixels are
            summed to form each emitter's intensity trace.
        background_inner: Inner radius, in pixels, of the annulus used to estimate
            local background around each emitter.
        background_outer: Outer radius, in pixels, of the background annulus.
        max_spots: Optional cap on the number of emitters kept, ranked by
            projection brightness. ``None`` keeps all of them.
        state_method: Two-level segmentation method for each trace.
        min_snr: Minimum amplitude-to-noise ratio for a trace to be treated as a
            genuine blinker rather than flat noise.
        min_transitions: Minimum number of ON/OFF transitions for a trace to be
            counted as a blinker.
        min_dwell_frames: Runs shorter than this many frames are merged into their
            neighbour, suppressing single-frame noise flips.
        power_law_xmin_s: Lower cutoff (seconds) for the dwell-time power-law fit.
            ``None`` uses the smallest observed dwell time.
    """

    # --- time base ---
    frame_interval_s: float = 1.0

    # --- spot detection ---
    projection: Projection = "std"
    detection_percentile: float = 99.0
    min_separation: int = 5
    roi_radius: int = 3
    background_inner: int = 5
    background_outer: int = 8
    max_spots: int | None = 2000

    # --- two-state segmentation ---
    state_method: StateMethod = "gmm"
    min_snr: float = 3.0
    min_transitions: int = 2
    min_dwell_frames: int = 1

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
        if self.background_inner <= self.roi_radius:
            raise ValueError("background_inner must exceed roi_radius")
        if self.background_outer <= self.background_inner:
            raise ValueError("background_outer must exceed background_inner")
        if self.max_spots is not None and self.max_spots < 1:
            raise ValueError("max_spots must be >= 1 or None")

    @property
    def fps(self) -> float:
        """Frame rate in frames per second (reciprocal of the frame interval)."""
        return 1.0 / self.frame_interval_s
