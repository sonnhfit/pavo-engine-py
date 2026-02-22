"""
Timeline tools for the video editing agent.

Defines callable tools that manipulate a timeline state dictionary.
Each tool corresponds to an OpenAI function-calling definition in
``TIMELINE_TOOL_DEFINITIONS`` and has a matching method on
:class:`TimelineTools`.
"""

import copy
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# OpenAI function-calling tool definitions
# ---------------------------------------------------------------------------

TIMELINE_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_video_strip",
            "description": "Add a video clip to a track in the timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "integer",
                        "description": "Track ID (0-based). Track 0 is the base layer.",
                    },
                    "src": {
                        "type": "string",
                        "description": "Path or URL to the video file.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start frame (0-based).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Duration in frames.",
                    },
                    "video_start_frame": {
                        "type": "integer",
                        "description": "First frame to read from the source video. Default 0.",
                    },
                    "effect": {
                        "type": "string",
                        "description": (
                            "Visual effect applied to the strip. "
                            "Supported: zoomIn, zoomOut, slideLeft, slideRight, slideUp, slideDown."
                        ),
                    },
                    "transition_in": {
                        "type": "string",
                        "description": "Transition effect at the start (e.g. 'fade').",
                    },
                    "transition_out": {
                        "type": "string",
                        "description": "Transition effect at the end (e.g. 'fade').",
                    },
                },
                "required": ["track_id", "src", "start", "length"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_image_strip",
            "description": "Add an image to a track in the timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "integer",
                        "description": "Track ID (0-based).",
                    },
                    "src": {
                        "type": "string",
                        "description": "Path or URL to the image file.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start frame (0-based).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Duration in frames.",
                    },
                    "effect": {
                        "type": "string",
                        "description": "Visual effect applied to the strip.",
                    },
                    "transition_in": {
                        "type": "string",
                        "description": "Transition effect at the start (e.g. 'fade').",
                    },
                    "transition_out": {
                        "type": "string",
                        "description": "Transition effect at the end (e.g. 'fade').",
                    },
                },
                "required": ["track_id", "src", "start", "length"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_text_strip",
            "description": "Add a text overlay to a track in the timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "integer",
                        "description": "Track ID (0-based).",
                    },
                    "text": {
                        "type": "string",
                        "description": "The text string to display.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start frame (0-based).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Duration in frames.",
                    },
                    "effect": {
                        "type": "string",
                        "description": "Visual effect applied to the strip.",
                    },
                    "transition_in": {
                        "type": "string",
                        "description": "Transition effect at the start.",
                    },
                    "transition_out": {
                        "type": "string",
                        "description": "Transition effect at the end.",
                    },
                },
                "required": ["track_id", "text", "start", "length"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_soundtrack",
            "description": "Set the background music / soundtrack for the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "Path or URL to the audio file.",
                    },
                    "effect": {
                        "type": "string",
                        "description": "Audio effect (e.g. 'fadeOut', 'fadeIn'). Default 'fadeOut'.",
                    },
                },
                "required": ["src"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_output_settings",
            "description": "Configure the output video format, resolution, and FPS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Container format (e.g. 'mp4'). Default 'mp4'.",
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Frames per second. Default 30.",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Output width in pixels.",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Output height in pixels.",
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Preset resolution label (sd, hd, fhd). Default 'sd'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_total_frames",
            "description": "Set the total number of frames (length) of the timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_frames": {
                        "type": "integer",
                        "description": "Total frame count.",
                    },
                },
                "required": ["n_frames"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# TimelineTools implementation
# ---------------------------------------------------------------------------


class TimelineTools:
    """
    Timeline manipulation tools for the video editing agent.

    Maintains a mutable timeline state dictionary and provides methods
    that directly correspond to :data:`TIMELINE_TOOL_DEFINITIONS`.
    An LLM agent calls :meth:`dispatch_tool_call` with the tool name and
    JSON-decoded arguments; the underlying method updates the state in place.

    Example::

        tools = TimelineTools()
        tools.set_total_frames(90)
        tools.set_output_settings(fps=30, width=1920, height=1080)
        tools.add_video_strip(track_id=0, src="clip.mp4", start=0, length=90)
        tools.add_text_strip(track_id=1, text="Hello World", start=0, length=90)
        timeline = tools.get_timeline()
    """

    def __init__(self) -> None:
        """Initialize with a default empty timeline."""
        self._timeline: Dict[str, Any] = {
            "timeline": {
                "n_frames": 0,
                "background": "#000000",
                "tracks": [],
            },
            "output": {
                "format": "mp4",
                "resolution": "sd",
                "fps": 30,
                "width": 1280,
                "height": 720,
            },
        }
        # Mapping from track_id to index in the tracks list
        self._track_index: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_track(self, track_id: int) -> Dict[str, Any]:
        """Return the track dict for *track_id*, creating it if absent."""
        if track_id not in self._track_index:
            track: Dict[str, Any] = {"track_id": track_id, "strips": []}
            self._timeline["timeline"]["tracks"].append(track)
            self._track_index[track_id] = (
                len(self._timeline["timeline"]["tracks"]) - 1
            )
        idx = self._track_index[track_id]
        return self._timeline["timeline"]["tracks"][idx]

    @staticmethod
    def _build_transition(transition_in: str, transition_out: str) -> Dict[str, str]:
        transition: Dict[str, str] = {}
        if transition_in:
            transition["in"] = transition_in
        if transition_out:
            transition["out"] = transition_out
        return transition

    # ------------------------------------------------------------------
    # Public tool methods
    # ------------------------------------------------------------------

    def add_video_strip(
        self,
        track_id: int,
        src: str,
        start: int,
        length: int,
        video_start_frame: int = 0,
        effect: str = "",
        transition_in: str = "",
        transition_out: str = "",
    ) -> Dict[str, Any]:
        """Add a video strip to *track_id*.

        Args:
            track_id: Track ID (0-based).
            src: Path or URL to the video file.
            start: Start frame (0-based).
            length: Duration in frames.
            video_start_frame: First frame to read from the source video.
            effect: Visual effect name (e.g. ``'zoomIn'``).
            transition_in: Transition at the beginning (e.g. ``'fade'``).
            transition_out: Transition at the end (e.g. ``'fade'``).

        Returns:
            The strip dictionary that was appended to the track.
        """
        track = self._get_or_create_track(track_id)
        strip: Dict[str, Any] = {
            "asset": {"type": "video", "src": src},
            "start": start,
            "video_start_frame": video_start_frame,
            "length": length,
            "effect": effect,
            "transition": self._build_transition(transition_in, transition_out),
        }
        track["strips"].append(strip)
        return strip

    def add_image_strip(
        self,
        track_id: int,
        src: str,
        start: int,
        length: int,
        effect: str = "",
        transition_in: str = "",
        transition_out: str = "",
    ) -> Dict[str, Any]:
        """Add an image strip to *track_id*.

        Args:
            track_id: Track ID (0-based).
            src: Path or URL to the image file.
            start: Start frame (0-based).
            length: Duration in frames.
            effect: Visual effect name.
            transition_in: Transition at the beginning.
            transition_out: Transition at the end.

        Returns:
            The strip dictionary that was appended to the track.
        """
        track = self._get_or_create_track(track_id)
        strip: Dict[str, Any] = {
            "asset": {"type": "image", "src": src},
            "start": start,
            "video_start_frame": 0,
            "length": length,
            "effect": effect,
            "transition": self._build_transition(transition_in, transition_out),
        }
        track["strips"].append(strip)
        return strip

    def add_text_strip(
        self,
        track_id: int,
        text: str,
        start: int,
        length: int,
        effect: str = "",
        transition_in: str = "",
        transition_out: str = "",
    ) -> Dict[str, Any]:
        """Add a text overlay strip to *track_id*.

        Args:
            track_id: Track ID (0-based).
            text: The text string to display.
            start: Start frame (0-based).
            length: Duration in frames.
            effect: Visual effect name.
            transition_in: Transition at the beginning.
            transition_out: Transition at the end.

        Returns:
            The strip dictionary that was appended to the track.
        """
        track = self._get_or_create_track(track_id)
        strip: Dict[str, Any] = {
            "asset": {"type": "text", "src": "", "text": text},
            "start": start,
            "video_start_frame": 0,
            "length": length,
            "effect": effect,
            "transition": self._build_transition(transition_in, transition_out),
        }
        track["strips"].append(strip)
        return strip

    def set_soundtrack(self, src: str, effect: str = "fadeOut") -> None:
        """Set the background soundtrack.

        Args:
            src: Path or URL to the audio file.
            effect: Audio effect (e.g. ``'fadeOut'``).
        """
        self._timeline["timeline"]["soundtrack"] = {"src": src, "effect": effect}

    def set_output_settings(
        self,
        format: str = "mp4",
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None,
        resolution: str = "sd",
    ) -> None:
        """Update the output video settings.

        Args:
            format: Container format (e.g. ``'mp4'``).
            fps: Frames per second.
            width: Output width in pixels (optional).
            height: Output height in pixels (optional).
            resolution: Preset resolution label (sd / hd / fhd).
        """
        self._timeline["output"]["format"] = format
        self._timeline["output"]["fps"] = fps
        self._timeline["output"]["resolution"] = resolution
        if width is not None:
            self._timeline["output"]["width"] = width
        if height is not None:
            self._timeline["output"]["height"] = height

    def set_total_frames(self, n_frames: int) -> None:
        """Set the total timeline length in frames.

        Args:
            n_frames: Total frame count.
        """
        self._timeline["timeline"]["n_frames"] = n_frames

    def get_timeline(self) -> Dict[str, Any]:
        """Return a deep copy of the current timeline state.

        Returns:
            A dictionary compatible with :func:`pavo.render_video`.
        """
        return copy.deepcopy(self._timeline)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def dispatch_tool_call(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Any:
        """Dispatch a tool call by name with keyword arguments.

        This is the entry-point used by :class:`~pavo.planner.director.Director`
        when it processes LLM tool-call responses.

        Args:
            tool_name: Name of the tool to invoke.
            tool_args: Keyword arguments decoded from the LLM response JSON.

        Returns:
            The return value of the invoked tool method, or ``None`` for
            void methods.

        Raises:
            ValueError: If *tool_name* is not a recognised tool.
        """
        tool_map = {
            "add_video_strip": self.add_video_strip,
            "add_image_strip": self.add_image_strip,
            "add_text_strip": self.add_text_strip,
            "set_soundtrack": self.set_soundtrack,
            "set_output_settings": self.set_output_settings,
            "set_total_frames": self.set_total_frames,
        }
        if tool_name not in tool_map:
            raise ValueError(
                f"Unknown tool: '{tool_name}'. "
                f"Available tools: {sorted(tool_map.keys())}"
            )
        return tool_map[tool_name](**tool_args)
