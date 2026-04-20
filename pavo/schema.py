"""Pydantic models for validating Pavo Engine timeline JSON files.

Use :func:`validate_timeline_json` to validate a parsed JSON dict before
rendering.  A :class:`pydantic.ValidationError` is translated into a
:class:`ValueError` with a human-readable message by the
:func:`pavo.pavo.render_video` caller.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Regex accepting #RGB, #RRGGBB, or #RRGGBBAA hex color strings.
_HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

# Valid soundtrack effect names.
SUPPORTED_SOUNDTRACK_EFFECTS = {"fadeIn", "fadeOut", "fadeInOut"}

# Valid strip effect names for media assets.
SUPPORTED_MEDIA_EFFECTS = {"zoomIn", "zoomOut", "panLeft", "panRight", "slideUp", "slideDown"}

# Valid text animation names.
SUPPORTED_TEXT_ANIMATIONS = {"typewriter", "fadeIn", "fadeOut", "slideInLeft"}

# Valid asset fit modes.
SUPPORTED_FIT_MODES = {"cover", "contain", "stretch", "none"}


def _validate_color(value: Any, field_name: str = "color") -> Any:
    """Return *value* if it is a valid FFmpeg color string, else raise."""
    if value is None:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    # Hex colors must match the expected pattern; named colors are passed through.
    if value.startswith("#") and not _HEX_COLOR_RE.match(value):
        raise ValueError(
            f"{field_name} hex value '{value}' is invalid; "
            "expected #RGB, #RRGGBB, or #RRGGBBAA format"
        )
    return value


class SoundtrackModel(BaseModel):
    """Optional background music track."""

    src: str = Field(..., description="Path or URL to the audio file.")
    effect: Optional[str] = Field(
        None,
        description=(
            "Audio effect: 'fadeIn', 'fadeOut', or 'fadeInOut'. "
            "Applied to the soundtrack using FFmpeg afade filter."
        ),
    )

    @field_validator("effect", mode="before")
    @classmethod
    def _validate_effect(cls, v: Any) -> Any:
        if v is not None and v not in SUPPORTED_SOUNDTRACK_EFFECTS:
            raise ValueError(
                f"unsupported soundtrack effect '{v}'; "
                f"valid options are: {sorted(SUPPORTED_SOUNDTRACK_EFFECTS)}"
            )
        return v


class PositionModel(BaseModel):
    """Text-overlay position.  ``x`` and ``y`` may be numbers or ``'center'``."""

    x: Union[float, int, str] = 0
    y: Union[float, int, str] = 0

    @field_validator("x", "y", mode="before")
    @classmethod
    def _validate_coordinate(cls, v: Any) -> Any:
        if isinstance(v, str) and v != "center":
            raise ValueError("position coordinate must be a number or 'center'")
        return v


class FiltersModel(BaseModel):
    """Per-strip color grading and visual filter configuration."""

    brightness: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Brightness adjustment in range [-1.0, 1.0]. 0.0 = no change.",
    )
    contrast: Optional[float] = Field(
        None, ge=0.0, le=2.0,
        description="Contrast multiplier in range [0.0, 2.0]. 1.0 = no change.",
    )
    saturation: Optional[float] = Field(
        None, ge=0.0, le=3.0,
        description="Saturation multiplier in range [0.0, 3.0]. 1.0 = no change.",
    )
    blur: Optional[float] = Field(
        None, gt=0.0,
        description="Gaussian blur sigma (radius). Larger values = more blur.",
    )
    grayscale: Optional[bool] = Field(
        False, description="Convert to grayscale when true."
    )
    sepia: Optional[bool] = Field(
        False, description="Apply sepia tone effect when true."
    )
    lut: Optional[str] = Field(
        None,
        description="Path to a .cube LUT file for professional color grading.",
    )


class ShadowModel(BaseModel):
    """Drop-shadow configuration for text overlays."""

    x: int = Field(2, description="Shadow x offset in pixels.")
    y: int = Field(2, description="Shadow y offset in pixels.")
    color: str = Field("black", description="Shadow color (FFmpeg color string or hex).")

    @field_validator("color", mode="before")
    @classmethod
    def _validate_shadow_color(cls, v: Any) -> Any:
        return _validate_color(v, "shadow color")


class AssetModel(BaseModel):
    """A single asset attached to a strip."""

    type: Literal["image", "video", "text", "subtitle", "audio", "watermark"] = Field(
        ...,
        description=(
            "Asset type: 'image', 'video', 'text', 'subtitle', 'audio', or 'watermark'."
        ),
    )
    # Media / audio assets
    src: Optional[str] = Field(
        None,
        description=(
            "File path for image, video, audio, or watermark assets. "
            "For subtitle assets, can point to an .srt or .vtt file."
        ),
    )
    # Text / subtitle fields
    content: Optional[str] = Field(None, description="Text content (text/subtitle strips only).")
    font: Optional[str] = Field(None, description="Path to a TTF/OTF font file.")
    size: Optional[int] = Field(24, ge=1, description="Font size in pixels.")
    color: Optional[str] = Field("white", description="FFmpeg color string or hex.")
    background_color: Optional[str] = Field(
        None, description="Background box color for subtitle strips (FFmpeg color string or hex)."
    )
    position: Optional[PositionModel] = Field(
        None, description="Top-left anchor for text or watermark overlays."
    )
    animation: Optional[str] = Field(
        None,
        description=(
            "Text animation: 'typewriter', 'fadeIn', 'fadeOut', or 'slideInLeft'. "
            "Media animation: 'zoomIn', 'zoomOut', 'panLeft', 'panRight', 'slideUp', 'slideDown'."
        ),
    )
    # Video trimming fields (video type only)
    trim_start: Optional[float] = Field(
        None, ge=0, description="Trim start time in seconds (video only)."
    )
    trim_end: Optional[float] = Field(
        None, gt=0, description="Trim end time in seconds (video only)."
    )
    trim_start_frame: Optional[int] = Field(
        None, ge=0, description="Trim start position in frames (video only)."
    )
    trim_end_frame: Optional[int] = Field(
        None, ge=1, description="Trim end position in frames (video only)."
    )
    # Video looping (video type only)
    loop: Optional[bool] = Field(
        False, description="Loop the video clip when the strip length exceeds the clip duration."
    )
    # Speed control (video type only)
    speed: Optional[float] = Field(
        None, gt=0,
        description="Playback speed multiplier. 2.0 = double speed; 0.5 = half speed (slow motion).",
    )
    # Scale / fit mode (image, video, watermark)
    fit: Optional[str] = Field(
        None,
        description=(
            "How to scale the asset to fit the output frame. "
            "Options: 'cover' (fill and crop), 'contain' (letterbox), "
            "'stretch' (exact dimensions), 'none' (keep original size)."
        ),
    )
    # Color grading / visual filters (image, video, watermark)
    filters: Optional[FiltersModel] = Field(
        None, description="Color grading and visual filter settings."
    )
    # Watermark / opacity (watermark type, or image overlay)
    opacity: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Alpha opacity for watermark or image overlays. 1.0 = fully opaque.",
    )
    # Audio volume (audio type)
    volume: Optional[float] = Field(
        None, gt=0,
        description="Volume multiplier for audio strips. 1.0 = original volume.",
    )
    # Text rendering improvements
    stroke_color: Optional[str] = Field(
        None, description="Text stroke (outline) color (FFmpeg color string or hex)."
    )
    stroke_width: Optional[int] = Field(
        None, ge=1, description="Text stroke (outline) width in pixels."
    )
    shadow: Optional[ShadowModel] = Field(
        None, description="Drop-shadow configuration for text overlays."
    )
    line_spacing: Optional[int] = Field(
        None, ge=0, description="Extra spacing between lines of multi-line text in pixels."
    )

    @field_validator("color", "background_color", "stroke_color", mode="before")
    @classmethod
    def _validate_color_field(cls, v: Any) -> Any:
        return _validate_color(v)

    @field_validator("fit", mode="before")
    @classmethod
    def _validate_fit(cls, v: Any) -> Any:
        if v is not None and v not in SUPPORTED_FIT_MODES:
            raise ValueError(
                f"unsupported fit mode '{v}'; "
                f"valid options are: {sorted(SUPPORTED_FIT_MODES)}"
            )
        return v

    @model_validator(mode="after")
    def _check_required_fields(self) -> "AssetModel":
        # Subtitle requires either content (inline) or src (SRT/VTT file).
        if self.type == "subtitle" and not self.content and not self.src:
            raise ValueError(
                "subtitle assets must include either a 'content' field (inline text) "
                "or a 'src' field pointing to an .srt / .vtt file"
            )
        if self.type == "text" and not self.content:
            raise ValueError("text assets must include a non-empty 'content' field")
        if self.type in ("image", "video", "audio", "watermark") and not self.src:
            raise ValueError(
                f"{self.type} assets must include a 'src' field with the file path"
            )
        # background_color is only valid for subtitle assets
        if self.background_color is not None and self.type != "subtitle":
            raise ValueError("'background_color' is only valid for subtitle assets")
        # Trim fields are only valid for video assets
        trim_fields = (self.trim_start, self.trim_end, self.trim_start_frame, self.trim_end_frame)
        if any(v is not None for v in trim_fields) and self.type != "video":
            raise ValueError(
                "trim fields (trim_start, trim_end, trim_start_frame, trim_end_frame) "
                "are only valid for video assets"
            )
        # Cannot specify both time-based and frame-based start trim
        if self.trim_start is not None and self.trim_start_frame is not None:
            raise ValueError("specify either 'trim_start' (seconds) or 'trim_start_frame' (frames), not both")
        # Cannot specify both time-based and frame-based end trim
        if self.trim_end is not None and self.trim_end_frame is not None:
            raise ValueError("specify either 'trim_end' (seconds) or 'trim_end_frame' (frames), not both")
        # trim_end must be greater than trim_start when both are provided in seconds
        if (
            self.trim_start is not None
            and self.trim_end is not None
            and self.trim_end <= self.trim_start
        ):
            raise ValueError(
                f"trim_end ({self.trim_end}) must be greater than trim_start ({self.trim_start})"
            )
        # loop and speed are only valid for video assets
        if self.loop and self.type != "video":
            raise ValueError("'loop' is only valid for video assets")
        if self.speed is not None and self.type != "video":
            raise ValueError("'speed' is only valid for video assets")
        # volume is only valid for audio assets
        if self.volume is not None and self.type != "audio":
            raise ValueError("'volume' is only valid for audio assets")
        # opacity is only valid for watermark assets (or image overlays)
        if self.opacity is not None and self.type not in ("watermark", "image"):
            raise ValueError("'opacity' is only valid for watermark and image assets")
        # Text-rendering fields only valid for text/subtitle
        text_only = (self.stroke_color, self.stroke_width, self.shadow, self.line_spacing)
        if any(v is not None for v in text_only) and self.type not in ("text", "subtitle"):
            raise ValueError(
                "stroke_color, stroke_width, shadow, and line_spacing are only valid "
                "for text and subtitle assets"
            )
        # filters / fit / opacity not meaningful for text/subtitle/audio
        if self.filters is not None and self.type in ("text", "subtitle", "audio"):
            raise ValueError(
                "'filters' is only valid for image, video, and watermark assets"
            )
        if self.fit is not None and self.type in ("text", "subtitle", "audio"):
            raise ValueError(
                "'fit' is only valid for image, video, and watermark assets"
            )
        return self


class TransitionModel(BaseModel):
    """Optional in/out transition configuration for a strip."""

    in_: Optional[str] = Field(None, alias="in", description="Transition type on enter.")
    out: Optional[str] = Field(None, description="Transition type on exit.")
    duration: Optional[int] = Field(
        5, ge=1, description="Transition length in frames (default 5)."
    )

    model_config = {"populate_by_name": True}

    @field_validator("in_", "out", mode="before")
    @classmethod
    def _validate_transition_type(cls, v: Any) -> Any:
        supported = {"fade", "slide", "wipe", "dissolve"}
        if v is not None and v not in supported:
            raise ValueError(
                f"unsupported transition type '{v}'; "
                f"valid options are: {sorted(supported)}"
            )
        return v


class StripModel(BaseModel):
    """One strip on the timeline (a single media/text item)."""

    asset: AssetModel
    start: int = Field(..., ge=0, description="Start frame index (0-based).")
    length: int = Field(..., ge=1, description="Duration in frames.")
    video_start_frame: int = Field(
        0, ge=0, description="Offset into the source video in frames."
    )
    effect: Optional[str] = Field(None, description="Named effect (e.g. 'zoomIn').")
    transition: Optional[TransitionModel] = Field(
        None, description="In/out transition configuration."
    )


class TrackModel(BaseModel):
    """A single track containing an ordered list of strips."""

    track_id: int = Field(..., ge=0, description="Track index (0 = bottom layer).")
    strips: List[StripModel] = Field(
        ..., min_length=1, description="Ordered list of strips on this track."
    )


class TimelineModel(BaseModel):
    """The top-level 'timeline' object inside the JSON file."""

    n_frames: Optional[int] = Field(
        None, ge=0,
        description=(
            "Total number of frames to render. "
            "If omitted, it is auto-computed as max(start + length) across all strips."
        ),
    )
    background: str = Field(
        "#000000", description="Hex background color, e.g. '#000000'."
    )
    tracks: List[TrackModel] = Field(
        default_factory=list, description="List of tracks (may be empty)."
    )
    soundtrack: Optional[SoundtrackModel] = Field(
        None, description="Optional background music track."
    )

    @field_validator("background", mode="before")
    @classmethod
    def _validate_background(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.startswith("#"):
            raise ValueError(
                f"background must be a hex color string starting with '#', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _check_strip_bounds_and_overlaps(self) -> "TimelineModel":
        """Warn about overlapping strips on the same track; validate bounds when n_frames is set."""
        for track in self.tracks:
            intervals: List[tuple] = []
            for strip in track.strips:
                strip_end = strip.start + strip.length
                # Validate strip bounds against n_frames when provided.
                if self.n_frames is not None and strip_end > self.n_frames:
                    raise ValueError(
                        f"strip on track {track.track_id} ends at frame {strip_end} "
                        f"which exceeds n_frames={self.n_frames}"
                    )
                # Collect intervals for overlap detection (skip audio strips, which can overlap).
                if strip.asset.type not in ("audio",):
                    intervals.append((strip.start, strip_end))

            # Warn about overlapping strips on the same visual track.
            for i, (s1, e1) in enumerate(intervals):
                for s2, e2 in intervals[i + 1:]:
                    if s1 < e2 and s2 < e1:
                        warnings.warn(
                            f"track {track.track_id} has overlapping strips "
                            f"([{s1}, {e1}) and [{s2}, {e2})). "
                            "Overlapping visual strips may produce unexpected results.",
                            UserWarning,
                            stacklevel=4,
                        )
                        break  # Only warn once per track.
        return self


class OutputModel(BaseModel):
    """The optional 'output' configuration block."""

    format: Optional[str] = Field(
        "mp4",
        description="Output container format. Use 'mp4' (default) or 'gif' for animated GIF.",
    )
    fps: Optional[float] = Field(25.0, gt=0, description="Frames per second.")
    width: Optional[int] = Field(None, ge=1, description="Output width in pixels.")
    height: Optional[int] = Field(None, ge=1, description="Output height in pixels.")
    audio_ducking: Optional[bool] = Field(
        False, description="Enable automatic volume ducking during speech."
    )
    ducking_reduction_db: Optional[float] = Field(
        10.0, description="Volume reduction in dB during speech segments."
    )
    workers: Optional[int] = Field(
        1, ge=1,
        description=(
            "Number of parallel worker threads for frame rendering. "
            "Values > 1 enable concurrent rendering for faster throughput."
        ),
    )


class TimelineFileModel(BaseModel):
    """Root model representing an entire Pavo timeline JSON file."""

    timeline: TimelineModel
    output: Optional[OutputModel] = Field(None, description="Output configuration.")


def _auto_compute_n_frames(data: Dict[str, Any]) -> int:
    """Compute n_frames from the maximum (start + length) across all strips.

    Parameters
    ----------
    data:
        The raw timeline dict (as loaded from JSON).

    Returns
    -------
    int
        The computed frame count, or 0 if no strips are present.
    """
    max_frame = 0
    for track in data.get("timeline", {}).get("tracks", []):
        for strip in track.get("strips", []):
            end = strip.get("start", 0) + strip.get("length", 0)
            if end > max_frame:
                max_frame = end
    return max_frame


def validate_timeline_json(data: Dict[str, Any]) -> TimelineFileModel:
    """Validate a parsed timeline JSON dict and return a :class:`TimelineFileModel`.

    Parameters
    ----------
    data:
        Dictionary produced by ``json.load()`` or equivalent.

    Returns
    -------
    TimelineFileModel
        The validated (and lightly coerced) model instance.

    Raises
    ------
    ValueError
        If the data does not conform to the expected schema.  The error message
        contains a human-readable description of every validation failure.

    Notes
    -----
    If ``timeline.n_frames`` is omitted from *data*, it is automatically
    computed as ``max(strip.start + strip.length)`` across all strips.
    """
    from pydantic import ValidationError

    # Auto-compute n_frames when the timeline key exists but n_frames is absent.
    timeline_data = data.get("timeline")
    if isinstance(timeline_data, dict) and timeline_data.get("n_frames") is None:
        computed = _auto_compute_n_frames(data)
        # Mutate a copy so the caller's dict is not modified.
        data = {**data, "timeline": {**timeline_data, "n_frames": computed}}

    try:
        return TimelineFileModel.model_validate(data)
    except ValidationError as exc:
        lines = ["Timeline JSON validation failed:"]
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"]) if err["loc"] else "root"
            lines.append(f"  • {loc}: {err['msg']}")
        raise ValueError("\n".join(lines)) from exc
