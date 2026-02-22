"""
LLM Director module for generating video edit plans from natural language.

The :class:`Director` uses an LLM with OpenAI-compatible function/tool calling
to iteratively build a :data:`~pavo.planner.tools.TIMELINE_TOOL_DEFINITIONS`-
driven timeline by calling the tools provided by
:class:`~pavo.planner.tools.TimelineTools`.
"""

import json
import os
from typing import Any, Dict, List, Optional

from pavo.planner.tools import TIMELINE_TOOL_DEFINITIONS, TimelineTools


class Director:
    """
    LLM-powered director that converts natural language prompts into timeline JSON.

    Uses an LLM (OpenAI by default) with function/tool calling to iteratively
    build a video timeline. The LLM decides which timeline tools to call and
    with what arguments; the :class:`~pavo.planner.tools.TimelineTools` instance
    executes them and maintains state.

    Attributes:
        model (str): LLM model identifier.
        max_iterations (int): Maximum tool-call rounds before giving up.

    Example::

        director = Director(model="gpt-4o", api_key="sk-...")
        timeline = director.plan(
            prompt="Create a 5-second video: logo.png zooms in, title 'Hello World' fades in",
            media_files=["logo.png"],
        )
        import json
        print(json.dumps(timeline, indent=2))
    """

    #: System prompt injected into every LLM conversation.
    SYSTEM_PROMPT: str = (
        "You are a professional video editor AI. "
        "Your job is to create a video timeline from the user's natural language description. "
        "Use the provided tools to build the timeline step by step.\n\n"
        "Available effects: zoomIn, zoomOut, slideLeft, slideRight, slideUp, slideDown\n"
        "Available transitions: fade\n\n"
        "Timeline rules:\n"
        "- Frame numbers start at 0.\n"
        "- Default FPS is 30 (1 second = 30 frames).\n"
        "- Track 0 is the base/background track; higher track IDs overlay on lower ones.\n"
        "- Always call set_total_frames to set the overall timeline length.\n"
        "- Always call set_output_settings to configure the output.\n"
        "- Stop calling tools once the timeline is fully built."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_iterations: int = 20,
    ) -> None:
        """Initialize the Director.

        Args:
            model: OpenAI-compatible model identifier. Default ``'gpt-4o'``.
            api_key: API key for the LLM provider. If ``None``, the
                ``OPENAI_API_KEY`` environment variable is used.
            max_iterations: Maximum number of tool-call rounds to prevent
                infinite loops. Default 20.

        Raises:
            ImportError: If the ``openai`` package is not installed.
        """
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the Director. "
                "Install it with:\n    pip install openai"
            ) from exc

        self.model = model
        self.max_iterations = max_iterations

        self._client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def plan(
        self,
        prompt: str,
        media_files: Optional[List[str]] = None,
        fps: int = 30,
    ) -> Dict[str, Any]:
        """Generate a timeline JSON from a natural language prompt.

        Runs an agentic loop where the LLM repeatedly calls timeline tools
        until it signals that the timeline is complete (no more tool calls).

        Args:
            prompt: Natural language description of the desired video edit.
            media_files: Paths or URLs of media files available for use in the
                timeline. These are listed in the user message so the LLM can
                reference them by path.
            fps: Frames per second for the output video. Passed to the LLM as
                context. Default 30.

        Returns:
            A timeline dictionary compatible with :func:`pavo.render_video`.

        Raises:
            RuntimeError: If the LLM API call fails.

        Example::

            director = Director(api_key="sk-...")
            timeline = director.plan(
                prompt="5-second intro with logo.png zooming in",
                media_files=["logo.png"],
            )
        """
        tools = TimelineTools()

        user_content = prompt
        if media_files:
            files_list = "\n".join(f"  - {f}" for f in media_files)
            user_content = (
                f"{prompt}\n\n"
                f"Available media files:\n{files_list}\n\n"
                f"Output FPS: {fps}"
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        for _ in range(self.max_iterations):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TIMELINE_TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
            except Exception as exc:
                raise RuntimeError(f"LLM API call failed: {exc}") from exc

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                # LLM finished building the timeline
                break

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    result = tools.dispatch_tool_call(tool_name, tool_args)
                    result_content = (
                        json.dumps(result) if result is not None else "success"
                    )
                except Exception as exc:
                    result_content = f"error: {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                )

        return tools.get_timeline()
