"""Pydantic models for validating Pavo Engine timeline JSON files.

Use :func:`validate_timeline_json` to validate a parsed JSON dict before
rendering.  A :class:`pydantic.ValidationError` is translated into a
:class:`ValueError` with a human-readable message by the
:func:`pavo.pavo.render_video` caller.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SoundtrackModel(BaseModel):
    """Optional background music track."""

    src: str = Field(..., description="Path or URL to the audio file.")
    effect: Optional[str] = Field(None, description="Audio effect, e.g. 'fadeOut'.")


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


class AssetModel(BaseModel):
    """A single asset attached to a strip (image, video, or text)."""

    type: Literal["image", "video", "text"] = Field(
        ..., description="Asset type: 'image', 'video', or 'text'."
    )
    # Media assets (image / video)
    src: Optional[str] = Field(
        None, description="File path for image or video assets."
    )
    # Text-only fields
    content: Optional[str] = Field(None, description="Text content (text strips only).")
    font: Optional[str] = Field(None, description="Path to a TTF/OTF font file.")
    size: Optional[int] = Field(24, ge=1, description="Font size in pixels.")
    color: Optional[str] = Field("white", description="FFmpeg color string or hex.")
    position: Optional[PositionModel] = Field(
        None, description="Top-left anchor for the text."
    )
    animation: Optional[str] = Field(
        None, description="Optional animation tag (stored for future use)."
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

    @model_validator(mode="after")
    def _check_required_fields(self) -> "AssetModel":
        if self.type == "text" and not self.content:
            raise ValueError("text assets must include a non-empty 'content' field")
        if self.type in ("image", "video") and not self.src:
            raise ValueError(
                f"{self.type} assets must include a 'src' field with the file path"
            )
        # Trim fields are only valid for video assets
        trim_fields = (self.trim_start, self.trim_end, self.trim_start_frame, self.trim_end_frame)
        if any(v is not None for v in trim_fields) and self.type != "video":
            raise ValueError("trim fields (trim_start, trim_end, trim_start_frame, trim_end_frame) are only valid for video assets")
        # Cannot specify both time-based and frame-based start trim
        if self.trim_start is not None and self.trim_start_frame is not None:
            raise ValueError("specify either 'trim_start' (seconds) or 'trim_start_frame' (frames), not both")
        # Cannot specify both time-based and frame-based end trim
        if self.trim_end is not None and self.trim_end_frame is not None:
            raise ValueError("specify either 'trim_end' (seconds) or 'trim_end_frame' (frames), not both")
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

    n_frames: int = Field(..., ge=0, description="Total number of frames to render.")
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


class OutputModel(BaseModel):
    """The optional 'output' configuration block."""

    format: Optional[str] = Field("mp4", description="Output container format.")
    fps: Optional[float] = Field(25.0, gt=0, description="Frames per second.")
    width: Optional[int] = Field(None, ge=1, description="Output width in pixels.")
    height: Optional[int] = Field(None, ge=1, description="Output height in pixels.")
    audio_ducking: Optional[bool] = Field(
        False, description="Enable automatic volume ducking during speech."
    )
    ducking_reduction_db: Optional[float] = Field(
        10.0, description="Volume reduction in dB during speech segments."
    )


class TimelineFileModel(BaseModel):
    """Root model representing an entire Pavo timeline JSON file."""

    timeline: TimelineModel
    output: Optional[OutputModel] = Field(None, description="Output configuration.")


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
    """
    from pydantic import ValidationError

    try:
        return TimelineFileModel.model_validate(data)
    except ValidationError as exc:
        lines = ["Timeline JSON validation failed:"]
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"]) if err["loc"] else "root"
            lines.append(f"  • {loc}: {err['msg']}")
        raise ValueError("\n".join(lines)) from exc
