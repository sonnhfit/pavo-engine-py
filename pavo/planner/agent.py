"""
Video editing agent that orchestrates the full natural-language-to-video pipeline.

:class:`VideoEditingAgent` accepts a natural language prompt and a list of media
files, generates an edit plan using :class:`~pavo.planner.director.Director`,
and renders the final video via :func:`pavo.render_video`.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from pavo.planner.director import Director
from pavo.planner.tools import TimelineTools


class VideoEditingAgent:
    """
    Autonomous video editing agent driven by natural language.

    Combines the LLM :class:`~pavo.planner.director.Director` with
    :func:`pavo.render_video` to turn a plain-text editing instruction into a
    rendered video file in a single call.

    Attributes:
        model (str): LLM model used for planning.
        fps (int): Default output FPS.

    Example::

        agent = VideoEditingAgent(api_key="sk-...")
        output_path = agent.edit(
            prompt="Create a 10-second slideshow with photo1.jpg and photo2.jpg, fade transitions",
            media_files=["photo1.jpg", "photo2.jpg"],
            output="slideshow.mp4",
        )
        print(f"Video saved to {output_path}")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        fps: int = 30,
    ) -> None:
        """Initialize the VideoEditingAgent.

        Args:
            model: LLM model identifier passed to
                :class:`~pavo.planner.director.Director`. Default ``'gpt-4o'``.
            api_key: OpenAI API key. If ``None``, the ``OPENAI_API_KEY``
                environment variable is used.
            fps: Default frames-per-second for rendered output. Default 30.
        """
        self.model = model
        self.fps = fps
        self._api_key = api_key
        self._director: Optional[Director] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_director(self) -> Director:
        """Return a lazily-initialized :class:`~pavo.planner.director.Director`."""
        if self._director is None:
            self._director = Director(model=self.model, api_key=self._api_key)
        return self._director

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        prompt: str,
        media_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a timeline plan without rendering.

        Useful for inspecting or modifying the plan before committing to a
        potentially slow render step.

        Args:
            prompt: Natural language editing instructions.
            media_files: Available media file paths or URLs.

        Returns:
            Timeline JSON dictionary compatible with :func:`pavo.render_video`.

        Example::

            agent = VideoEditingAgent(api_key="sk-...")
            timeline = agent.plan(
                "5-second clip: logo.png zooms in",
                media_files=["logo.png"],
            )
            import json
            print(json.dumps(timeline, indent=2))
        """
        director = self._get_director()
        return director.plan(prompt, media_files=media_files, fps=self.fps)

    def edit(
        self,
        prompt: str,
        media_files: Optional[List[str]] = None,
        output: str = "output.mp4",
        timeline_save_path: Optional[str] = None,
    ) -> str:
        """Edit a video from a natural language prompt.

        Calls :meth:`plan` to generate a timeline, writes it to a temporary
        JSON file, and then passes it to :func:`pavo.render_video`.

        Args:
            prompt: Natural language description of the desired edit.
            media_files: Media file paths or URLs available to the editor.
            output: Destination path for the rendered video. Default
                ``'output.mp4'``.
            timeline_save_path: If provided, the generated timeline JSON is
                also written to this path for inspection or reuse.

        Returns:
            The *output* path passed in (i.e. the path to the rendered video).

        Raises:
            RuntimeError: If planning or rendering fails.

        Example::

            agent = VideoEditingAgent(api_key="sk-...")
            agent.edit(
                prompt="10-second slideshow: photo1.jpg then photo2.jpg with fade",
                media_files=["photo1.jpg", "photo2.jpg"],
                output="slideshow.mp4",
            )
        """
        # Lazy import to avoid a hard dependency when only planning is needed.
        from pavo import render_video  # noqa: PLC0415

        timeline = self.plan(prompt, media_files=media_files)

        if timeline_save_path:
            with open(timeline_save_path, "w", encoding="utf-8") as f:
                json.dump(timeline, f, indent=2)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(timeline, f, indent=2)
            render_video(tmp_path, output)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return output

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoEditingAgent":
        """Context manager entry — returns *self*."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit — no special teardown required."""
        return False
