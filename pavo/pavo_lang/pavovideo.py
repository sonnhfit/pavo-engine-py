from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from pavo.schema import validate_timeline_json

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*$")


class PavoText:
    """A text node in the Pavo language DSL."""

    def __init__(self, video: "PavoVideo", asset: Dict[str, Any], track_id: int = 0) -> None:
        self._video = video
        self._asset = asset
        self._track_id = track_id

    def animate(
        self,
        from_state: Optional[Dict[str, Any]],
        to_state: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> "PavoText":
        """Append a text strip with inferred animation and advance the playhead."""
        options = options or {}
        duration_frames = self._video.parse_duration(options.get("duration", "0.5s"), min_frames=1)

        asset = dict(self._asset)
        animation = self._infer_animation(from_state or {}, to_state or {})
        if animation:
            asset["animation"] = animation

        self._video._append_strip(
            track_id=self._track_id,
            strip={
                "asset": asset,
                "start": self._video.cursor,
                "length": duration_frames,
            },
        )

        self._video.cursor += duration_frames
        return self

    @staticmethod
    def _infer_animation(from_state: Dict[str, Any], to_state: Dict[str, Any]) -> Optional[str]:
        start_opacity = from_state.get("opacity")
        end_opacity = to_state.get("opacity")
        if start_opacity == 0 and end_opacity == 1:
            return "fadeIn"
        if start_opacity == 1 and end_opacity == 0:
            return "fadeOut"
        if from_state.get("x") is not None and to_state.get("x") == 0:
            return "slideInLeft"
        return None


class PavoVideo:
    """Python DSL for generating valid Pavo timeline JSON."""

    def __init__(
        self,
        *,
        name: str,
        width: int,
        height: int,
        fps: float = 25.0,
        background: str = "#000000",
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.background = background

        self.cursor = 0
        self._tracks: Dict[int, list[Dict[str, Any]]] = {}
        self._soundtrack: Optional[Dict[str, Any]] = None

    def addText(
        self,
        *,
        text: str,
        fontSize: int = 24,
        fontWeight: Optional[int] = None,
        color: str = "white",
        x: Any = "center",
        y: Any = "center",
        trackId: int = 0,
        font: Optional[str] = None,
    ) -> PavoText:
        """Create a text node and return an object that supports ``animate``."""
        asset: Dict[str, Any] = {
            "type": "text",
            "content": text,
            "size": fontSize,
            "color": color,
            "position": {"x": x, "y": y},
        }
        if font:
            asset["font"] = font

        # Keep fontWeight in DSL for parity with VideoFlow; renderer currently
        # controls visual weight primarily through chosen font files.
        if fontWeight is not None:
            asset["font_weight"] = fontWeight

        return PavoText(video=self, asset=asset, track_id=trackId)

    def wait(self, duration: Any) -> "PavoVideo":
        """Move the global playhead forward by ``duration``."""
        self.cursor += self.parse_duration(duration, min_frames=0)
        return self

    def setSoundtrack(self, *, src: str, effect: Optional[str] = None) -> "PavoVideo":
        self._soundtrack = {"src": src}
        if effect:
            self._soundtrack["effect"] = effect
        return self

    def addStrip(self, *, trackId: int, start: Any, length: Any, asset: Dict[str, Any]) -> "PavoVideo":
        """Low-level API for power users to append any valid strip."""
        self._append_strip(
            track_id=trackId,
            strip={
                "asset": asset,
                "start": self.parse_duration(start, min_frames=0),
                "length": self.parse_duration(length, min_frames=1),
            },
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        tracks = [
            {"track_id": track_id, "strips": strips}
            for track_id, strips in sorted(self._tracks.items(), key=lambda item: item[0])
            if strips
        ]

        payload: Dict[str, Any] = {
            "timeline": {
                "background": self.background,
                "tracks": tracks,
            },
            "output": {
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
            },
        }

        if self._soundtrack:
            payload["timeline"]["soundtrack"] = self._soundtrack

        model = validate_timeline_json(payload)
        return model.model_dump(by_alias=True, exclude_none=True)

    def to_json(self, *, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def save_json(self, output: str | Path, *, indent: int = 2, ensure_ascii: bool = False) -> Path:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent, ensure_ascii=ensure_ascii), encoding="utf-8")
        return output_path

    def _append_strip(self, *, track_id: int, strip: Dict[str, Any]) -> None:
        if track_id not in self._tracks:
            self._tracks[track_id] = []
        self._tracks[track_id].append(strip)

    def parse_duration(self, value: Any, *, min_frames: int = 0) -> int:
        if isinstance(value, int):
            frames = value
        elif isinstance(value, float):
            frames = round(value * self.fps)
        elif isinstance(value, str):
            match = _DURATION_RE.match(value)
            if not match:
                raise ValueError(f"Invalid duration value: {value!r}")
            amount = float(match.group(1))
            unit = match.group(2) or "s"
            if unit == "ms":
                frames = round((amount / 1000.0) * self.fps)
            else:
                frames = round(amount * self.fps)
        else:
            raise TypeError("Duration must be int, float, or time string like '1.5s' or '250ms'")

        if frames < min_frames:
            frames = min_frames
        return frames
