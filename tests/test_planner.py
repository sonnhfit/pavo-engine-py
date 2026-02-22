"""
Unit tests for pavo.planner.tools.TimelineTools.

These tests exercise the stateful timeline builder without requiring an LLM
or any external API key.
"""

import copy

import pytest

from pavo.planner.tools import TIMELINE_TOOL_DEFINITIONS, TimelineTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tools() -> TimelineTools:
    return TimelineTools()


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


def test_initial_timeline_structure(tools: TimelineTools) -> None:
    timeline = tools.get_timeline()
    assert "timeline" in timeline
    assert "output" in timeline
    assert timeline["timeline"]["n_frames"] == 0
    assert timeline["timeline"]["tracks"] == []


def test_get_timeline_returns_deep_copy(tools: TimelineTools) -> None:
    t1 = tools.get_timeline()
    t1["timeline"]["n_frames"] = 999
    t2 = tools.get_timeline()
    assert t2["timeline"]["n_frames"] == 0


# ---------------------------------------------------------------------------
# set_total_frames
# ---------------------------------------------------------------------------


def test_set_total_frames(tools: TimelineTools) -> None:
    tools.set_total_frames(300)
    assert tools.get_timeline()["timeline"]["n_frames"] == 300


# ---------------------------------------------------------------------------
# set_output_settings
# ---------------------------------------------------------------------------


def test_set_output_settings_defaults(tools: TimelineTools) -> None:
    tools.set_output_settings()
    output = tools.get_timeline()["output"]
    assert output["format"] == "mp4"
    assert output["fps"] == 30
    assert output["resolution"] == "sd"


def test_set_output_settings_custom(tools: TimelineTools) -> None:
    tools.set_output_settings(format="mp4", fps=60, width=1920, height=1080, resolution="fhd")
    output = tools.get_timeline()["output"]
    assert output["fps"] == 60
    assert output["width"] == 1920
    assert output["height"] == 1080
    assert output["resolution"] == "fhd"


def test_set_output_settings_without_dimensions(tools: TimelineTools) -> None:
    original_width = tools.get_timeline()["output"]["width"]
    tools.set_output_settings()
    assert tools.get_timeline()["output"]["width"] == original_width


# ---------------------------------------------------------------------------
# add_video_strip
# ---------------------------------------------------------------------------


def test_add_video_strip_creates_track(tools: TimelineTools) -> None:
    tools.add_video_strip(track_id=0, src="clip.mp4", start=0, length=90)
    timeline = tools.get_timeline()
    assert len(timeline["timeline"]["tracks"]) == 1
    track = timeline["timeline"]["tracks"][0]
    assert track["track_id"] == 0
    assert len(track["strips"]) == 1


def test_add_video_strip_fields(tools: TimelineTools) -> None:
    tools.add_video_strip(
        track_id=0,
        src="clip.mp4",
        start=10,
        length=30,
        video_start_frame=5,
        effect="zoomIn",
        transition_in="fade",
        transition_out="fade",
    )
    strip = tools.get_timeline()["timeline"]["tracks"][0]["strips"][0]
    assert strip["asset"]["type"] == "video"
    assert strip["asset"]["src"] == "clip.mp4"
    assert strip["start"] == 10
    assert strip["length"] == 30
    assert strip["video_start_frame"] == 5
    assert strip["effect"] == "zoomIn"
    assert strip["transition"]["in"] == "fade"
    assert strip["transition"]["out"] == "fade"


def test_add_video_strip_no_transition(tools: TimelineTools) -> None:
    tools.add_video_strip(track_id=0, src="clip.mp4", start=0, length=30)
    strip = tools.get_timeline()["timeline"]["tracks"][0]["strips"][0]
    assert strip["transition"] == {}


# ---------------------------------------------------------------------------
# add_image_strip
# ---------------------------------------------------------------------------


def test_add_image_strip(tools: TimelineTools) -> None:
    tools.add_image_strip(track_id=0, src="photo.jpg", start=0, length=60, effect="zoomIn")
    strip = tools.get_timeline()["timeline"]["tracks"][0]["strips"][0]
    assert strip["asset"]["type"] == "image"
    assert strip["asset"]["src"] == "photo.jpg"
    assert strip["effect"] == "zoomIn"
    assert strip["video_start_frame"] == 0


# ---------------------------------------------------------------------------
# add_text_strip
# ---------------------------------------------------------------------------


def test_add_text_strip(tools: TimelineTools) -> None:
    tools.add_text_strip(track_id=1, text="Hello World", start=0, length=90)
    timeline = tools.get_timeline()
    # Only one track should have been created (track 1)
    assert len(timeline["timeline"]["tracks"]) == 1
    strip = timeline["timeline"]["tracks"][0]["strips"][0]
    assert strip["asset"]["type"] == "text"
    assert strip["asset"]["text"] == "Hello World"
    assert strip["asset"]["src"] == ""


# ---------------------------------------------------------------------------
# set_soundtrack
# ---------------------------------------------------------------------------


def test_set_soundtrack(tools: TimelineTools) -> None:
    tools.set_soundtrack(src="music.mp3", effect="fadeOut")
    timeline = tools.get_timeline()
    assert timeline["timeline"]["soundtrack"]["src"] == "music.mp3"
    assert timeline["timeline"]["soundtrack"]["effect"] == "fadeOut"


def test_set_soundtrack_default_effect(tools: TimelineTools) -> None:
    tools.set_soundtrack(src="music.mp3")
    assert tools.get_timeline()["timeline"]["soundtrack"]["effect"] == "fadeOut"


# ---------------------------------------------------------------------------
# Multiple tracks & strips
# ---------------------------------------------------------------------------


def test_multiple_tracks(tools: TimelineTools) -> None:
    tools.add_image_strip(track_id=0, src="bg.jpg", start=0, length=90)
    tools.add_text_strip(track_id=1, text="Title", start=0, length=90)
    timeline = tools.get_timeline()
    assert len(timeline["timeline"]["tracks"]) == 2


def test_multiple_strips_same_track(tools: TimelineTools) -> None:
    tools.add_image_strip(track_id=0, src="a.jpg", start=0, length=30)
    tools.add_image_strip(track_id=0, src="b.jpg", start=30, length=30)
    timeline = tools.get_timeline()
    assert len(timeline["timeline"]["tracks"]) == 1
    assert len(timeline["timeline"]["tracks"][0]["strips"]) == 2


def test_track_id_preserved(tools: TimelineTools) -> None:
    tools.add_video_strip(track_id=3, src="clip.mp4", start=0, length=30)
    timeline = tools.get_timeline()
    assert timeline["timeline"]["tracks"][0]["track_id"] == 3


# ---------------------------------------------------------------------------
# dispatch_tool_call
# ---------------------------------------------------------------------------


def test_dispatch_add_video_strip(tools: TimelineTools) -> None:
    tools.dispatch_tool_call(
        "add_video_strip",
        {"track_id": 0, "src": "clip.mp4", "start": 0, "length": 60},
    )
    assert len(tools.get_timeline()["timeline"]["tracks"]) == 1


def test_dispatch_set_total_frames(tools: TimelineTools) -> None:
    tools.dispatch_tool_call("set_total_frames", {"n_frames": 150})
    assert tools.get_timeline()["timeline"]["n_frames"] == 150


def test_dispatch_unknown_tool_raises(tools: TimelineTools) -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.dispatch_tool_call("fly_to_the_moon", {})


# ---------------------------------------------------------------------------
# TIMELINE_TOOL_DEFINITIONS sanity checks
# ---------------------------------------------------------------------------


def test_tool_definitions_is_list() -> None:
    assert isinstance(TIMELINE_TOOL_DEFINITIONS, list)
    assert len(TIMELINE_TOOL_DEFINITIONS) > 0


def test_tool_definitions_have_required_keys() -> None:
    for defn in TIMELINE_TOOL_DEFINITIONS:
        assert defn["type"] == "function"
        assert "function" in defn
        assert "name" in defn["function"]
        assert "parameters" in defn["function"]


def test_tool_definitions_names_match_dispatcher() -> None:
    """Every tool definition name should be dispatchable."""
    tools = TimelineTools()
    dispatchable_names = {
        "add_video_strip",
        "add_image_strip",
        "add_text_strip",
        "set_soundtrack",
        "set_output_settings",
        "set_total_frames",
    }
    defined_names = {d["function"]["name"] for d in TIMELINE_TOOL_DEFINITIONS}
    assert defined_names == dispatchable_names
