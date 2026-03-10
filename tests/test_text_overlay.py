"""Unit tests for text overlay support (asset.type == "text")."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from pavo.sequancer.seq import Strip, Sequence
from pavo.sequancer.render import get_strips_from_json


# ---------------------------------------------------------------------------
# Strip construction and attribute defaults
# ---------------------------------------------------------------------------

class TestTextStripAttributes:
    def test_default_text_attributes(self):
        strip = Strip(type="text")
        assert strip.content is None
        assert strip.font is None
        assert strip.size == 24
        assert strip.color == "white"
        assert strip.position == {"x": 0, "y": 0}
        assert strip.animation is None

    def test_custom_text_attributes(self):
        strip = Strip(
            type="text",
            content="Hello World",
            font="/path/to/font.ttf",
            size=48,
            color="yellow",
            position={"x": 50, "y": 100},
            animation="fadeIn",
        )
        assert strip.content == "Hello World"
        assert strip.font == "/path/to/font.ttf"
        assert strip.size == 48
        assert strip.color == "yellow"
        assert strip.position == {"x": 50, "y": 100}
        assert strip.animation == "fadeIn"

    def test_get_frame_returns_none_for_text(self):
        strip = Strip(type="text", content="Test")
        result = strip.get_frame(0, "/tmp")
        assert result is None


# ---------------------------------------------------------------------------
# Strip.apply_text()
# ---------------------------------------------------------------------------

class TestApplyText:
    def _make_stream(self):
        """Return a mock ffmpeg stream with a trackable .filter() method."""
        stream = MagicMock()
        stream.filter.return_value = stream
        return stream

    def test_apply_text_calls_drawtext_filter(self):
        strip = Strip(type="text", content="Hi", size=32, color="red", position={"x": 10, "y": 20})
        stream = self._make_stream()
        result = strip.apply_text(stream)
        stream.filter.assert_called_once_with(
            "drawtext",
            text="Hi",
            fontsize=32,
            fontcolor="red",
            x="10",
            y="20",
        )
        assert result is stream

    def test_apply_text_includes_fontfile_when_set(self):
        strip = Strip(
            type="text",
            content="Title",
            font="/fonts/arial.ttf",
            size=24,
            color="white",
            position={"x": 0, "y": 0},
        )
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["fontfile"] == "/fonts/arial.ttf"

    def test_apply_text_omits_fontfile_when_none(self):
        strip = Strip(type="text", content="Hi", font=None, position={"x": 0, "y": 0})
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert "fontfile" not in call_kwargs

    def test_apply_text_returns_unchanged_stream_when_no_content(self):
        strip = Strip(type="text", content=None)
        stream = self._make_stream()
        result = strip.apply_text(stream)
        stream.filter.assert_not_called()
        assert result is stream

    def test_apply_text_center_position_x(self):
        strip = Strip(type="text", content="Center", position={"x": "center", "y": 50})
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["x"] == "(w-tw)/2"
        assert call_kwargs["y"] == "50"

    def test_apply_text_center_position_y(self):
        strip = Strip(type="text", content="Center Y", position={"x": 10, "y": "center"})
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["x"] == "10"
        assert call_kwargs["y"] == "(h-th)/2"

    def test_apply_text_defaults_position_when_missing_keys(self):
        strip = Strip(type="text", content="Hello", position={})
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["x"] == "0"
        assert call_kwargs["y"] == "0"


# ---------------------------------------------------------------------------
# get_strips_from_json with text assets
# ---------------------------------------------------------------------------

class TestGetStripsFromJsonText:
    def _make_json(self, include_optional=True, content="Sample Text"):
        asset = {
            "type": "text",
            "content": content,
        }
        if include_optional:
            asset.update({
                "font": "/path/to/font.ttf",
                "size": 40,
                "color": "yellow",
                "position": {"x": 20, "y": 30},
                "animation": "slideIn",
            })
        return {
            "timeline": {
                "n_frames": 10,
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": asset,
                                "start": 0,
                                "length": 10,
                                "effect": None,
                                "video_start_frame": 0,
                            }
                        ],
                    }
                ],
            }
        }

    def test_text_strip_parsed_correctly(self):
        json_data = self._make_json()
        strips = get_strips_from_json(json_data)
        assert len(strips) == 1
        s = strips[0]
        assert s.type == "text"
        assert s.content == "Sample Text"
        assert s.font == "/path/to/font.ttf"
        assert s.size == 40
        assert s.color == "yellow"
        assert s.position == {"x": 20, "y": 30}
        assert s.animation == "slideIn"
        assert s.media_source is None

    def test_text_strip_defaults_applied(self):
        json_data = self._make_json(include_optional=False, content="Default Test")
        strips = get_strips_from_json(json_data)
        s = strips[0]
        assert s.content == "Default Test"
        assert s.font is None
        assert s.size == 24
        assert s.color == "white"
        assert s.position == {"x": 0, "y": 0}
        assert s.animation is None

    def test_text_strip_track_id_and_timing(self):
        json_data = self._make_json()
        json_data["timeline"]["tracks"][0]["track_id"] = 2
        json_data["timeline"]["tracks"][0]["strips"][0]["start"] = 5
        json_data["timeline"]["tracks"][0]["strips"][0]["length"] = 8

        strips = get_strips_from_json(json_data)
        s = strips[0]
        assert s.track_id == 2
        assert s.start_frame == 5
        assert s.length == 8

    def test_non_text_strip_still_parsed_with_src(self):
        json_data = {
            "timeline": {
                "n_frames": 5,
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "image", "src": "docs/media/im1.jpg"},
                                "start": 0,
                                "length": 5,
                                "effect": "zoomIn",
                                "video_start_frame": 0,
                            }
                        ],
                    }
                ],
            }
        }
        strips = get_strips_from_json(json_data)
        assert strips[0].media_source == "docs/media/im1.jpg"
        assert strips[0].type == "image"


# ---------------------------------------------------------------------------
# Sequence.render_strip_list with text strips
# ---------------------------------------------------------------------------

class TestSequenceRenderStripListText:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_text_strip(self, content="Hello", x=10, y=10, track_id=1):
        strip = Strip(
            type="text",
            content=content,
            size=24,
            color="white",
            position={"x": x, "y": y},
            track_id=track_id,
        )
        return strip

    def _make_image_strip(self, track_id=0):
        strip = Strip(type="image", media_source="img.jpg", track_id=track_id)
        return strip

    def test_text_strip_applied_via_apply_text(self):
        """Text strip should call apply_text() on the underlying image stream."""
        image_stream = MagicMock()
        image_stream.filter.return_value = image_stream

        image_strip = self._make_image_strip()
        text_strip = self._make_text_strip()

        seq = Sequence(strips=[], n_frame=1, temp_dir=self._tmpdir)

        # Image first (track_id=0), text second (track_id=1) — natural z-order.
        with patch.object(image_strip, "get_frame", return_value=image_stream):
            result = seq.render_strip_list([image_strip, text_strip], frame=0)

        image_stream.filter.assert_called_once()
        call_args = image_stream.filter.call_args
        assert call_args[0][0] == "drawtext"
        assert call_args[1]["text"] == "Hello"

    def test_text_strip_not_applied_without_base_image(self):
        """Text strip alone (no base image) returns None."""
        text_strip = self._make_text_strip()
        seq = Sequence(strips=[], n_frame=1, temp_dir=self._tmpdir)
        result = seq.render_strip_list([text_strip], frame=0)
        assert result is None

    def test_multiple_text_strips_all_applied(self):
        """Multiple text strips are all applied as drawtext layers."""
        image_stream = MagicMock()
        # Each drawtext call returns a new mock representing the filtered stream.
        after_first_text = MagicMock()
        after_second_text = MagicMock()
        image_stream.filter.return_value = after_first_text
        after_first_text.filter.return_value = after_second_text

        image_strip = self._make_image_strip()
        text1 = self._make_text_strip(content="Line 1", track_id=1)
        text2 = self._make_text_strip(content="Line 2", track_id=2)

        seq = Sequence(strips=[], n_frame=1, temp_dir=self._tmpdir)
        with patch.object(image_strip, "get_frame", return_value=image_stream):
            result = seq.render_strip_list([image_strip, text1, text2], frame=0)

        assert image_stream.filter.call_count == 1
        assert after_first_text.filter.call_count == 1
        assert result is after_second_text
