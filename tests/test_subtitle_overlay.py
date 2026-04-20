"""Unit tests for subtitle overlay support (asset.type == "subtitle")."""

import tempfile
import pytest
from unittest.mock import MagicMock, patch

from pavo.sequancer.seq import Strip, Sequence
from pavo.sequancer.render import get_strips_from_json
from pavo.schema import validate_timeline_json


# ---------------------------------------------------------------------------
# Schema validation – subtitle asset
# ---------------------------------------------------------------------------

def _minimal_subtitle_timeline(**asset_overrides):
    """Return a valid timeline dict containing a subtitle strip."""
    asset = {
        "type": "subtitle",
        "content": "Hello World",
    }
    asset.update(asset_overrides)
    return {
        "timeline": {
            "n_frames": 25,
            "background": "#000000",
            "tracks": [
                {
                    "track_id": 1,
                    "strips": [
                        {"asset": asset, "start": 0, "length": 25},
                    ],
                }
            ],
        },
        "output": {"fps": 25, "width": 1280, "height": 720},
    }


class TestSubtitleSchemaValidation:
    def test_minimal_subtitle_is_valid(self):
        result = validate_timeline_json(_minimal_subtitle_timeline())
        asset = result.timeline.tracks[0].strips[0].asset
        assert asset.type == "subtitle"
        assert asset.content == "Hello World"

    def test_subtitle_with_all_style_fields(self):
        result = validate_timeline_json(
            _minimal_subtitle_timeline(
                font="/fonts/roboto.ttf",
                size=36,
                color="#ffffff",
                background_color="black@0.5",
                position={"x": "center", "y": 600},
            )
        )
        asset = result.timeline.tracks[0].strips[0].asset
        assert asset.font == "/fonts/roboto.ttf"
        assert asset.size == 36
        assert asset.color == "#ffffff"
        assert asset.background_color == "black@0.5"
        assert asset.position.x == "center"
        assert asset.position.y == 600

    def test_subtitle_size_must_be_positive(self):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            validate_timeline_json(_minimal_subtitle_timeline(size=0))

    def test_subtitle_missing_content_raises(self):
        with pytest.raises(ValueError, match="content.*src|src.*content"):
            validate_timeline_json(
                {
                    "timeline": {
                        "n_frames": 10,
                        "background": "#000000",
                        "tracks": [
                            {
                                "track_id": 0,
                                "strips": [
                                    {
                                        "asset": {"type": "subtitle"},
                                        "start": 0,
                                        "length": 10,
                                    }
                                ],
                            }
                        ],
                    }
                }
            )

    # Color validation tests
    def test_valid_hex_color_rrggbb(self):
        result = validate_timeline_json(_minimal_subtitle_timeline(color="#ff0000"))
        assert result.timeline.tracks[0].strips[0].asset.color == "#ff0000"

    def test_valid_hex_color_rgb(self):
        result = validate_timeline_json(_minimal_subtitle_timeline(color="#f00"))
        assert result.timeline.tracks[0].strips[0].asset.color == "#f00"

    def test_valid_hex_color_rrggbbaa(self):
        result = validate_timeline_json(
            _minimal_subtitle_timeline(background_color="#00000080")
        )
        assert result.timeline.tracks[0].strips[0].asset.background_color == "#00000080"

    def test_valid_named_color(self):
        result = validate_timeline_json(_minimal_subtitle_timeline(color="yellow"))
        assert result.timeline.tracks[0].strips[0].asset.color == "yellow"

    def test_invalid_hex_color_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_timeline_json(_minimal_subtitle_timeline(color="#ZZZZZZ"))

    def test_invalid_hex_color_wrong_length(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_timeline_json(_minimal_subtitle_timeline(color="#12345"))

    def test_background_color_on_text_asset_raises(self):
        """background_color is only valid for subtitle assets."""
        with pytest.raises(ValueError, match="background_color"):
            validate_timeline_json(
                {
                    "timeline": {
                        "n_frames": 10,
                        "background": "#000000",
                        "tracks": [
                            {
                                "track_id": 0,
                                "strips": [
                                    {
                                        "asset": {
                                            "type": "text",
                                            "content": "hi",
                                            "background_color": "black",
                                        },
                                        "start": 0,
                                        "length": 10,
                                    }
                                ],
                            }
                        ],
                    }
                }
            )

    def test_position_center(self):
        result = validate_timeline_json(
            _minimal_subtitle_timeline(position={"x": "center", "y": "center"})
        )
        pos = result.timeline.tracks[0].strips[0].asset.position
        assert pos.x == "center"
        assert pos.y == "center"

    def test_position_invalid_string_raises(self):
        with pytest.raises(ValueError):
            validate_timeline_json(
                _minimal_subtitle_timeline(position={"x": "left", "y": 0})
            )


# ---------------------------------------------------------------------------
# Strip construction – subtitle type
# ---------------------------------------------------------------------------

class TestSubtitleStripAttributes:
    def test_default_subtitle_attributes(self):
        strip = Strip(type="subtitle", content="Test")
        assert strip.type == "subtitle"
        assert strip.content == "Test"
        assert strip.font is None
        assert strip.size == 24
        assert strip.color == "white"
        assert strip.background_color is None
        assert strip.position == {"x": 0, "y": 0}
        assert strip.animation is None

    def test_custom_subtitle_attributes(self):
        strip = Strip(
            type="subtitle",
            content="Caption",
            font="/fonts/arial.ttf",
            size=32,
            color="#ffffff",
            background_color="#000000",
            position={"x": "center", "y": 600},
            animation="fadeIn",
        )
        assert strip.background_color == "#000000"
        assert strip.color == "#ffffff"
        assert strip.font == "/fonts/arial.ttf"
        assert strip.size == 32

    def test_get_frame_returns_none_for_subtitle(self):
        strip = Strip(type="subtitle", content="Sub")
        assert strip.get_frame(0, "/tmp") is None


# ---------------------------------------------------------------------------
# Strip.apply_text() – subtitle rendering
# ---------------------------------------------------------------------------

class TestSubtitleApplyText:
    def _make_stream(self):
        stream = MagicMock()
        stream.filter.return_value = stream
        return stream

    def test_subtitle_without_background_color(self):
        strip = Strip(
            type="subtitle", content="Hi", size=28, color="white",
            position={"x": "center", "y": 650},
        )
        stream = self._make_stream()
        result = strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["text"] == "Hi"
        assert call_kwargs["fontcolor"] == "white"
        assert call_kwargs["x"] == "(w-tw)/2"
        assert call_kwargs["y"] == "650"
        assert "box" not in call_kwargs
        assert "boxcolor" not in call_kwargs

    def test_subtitle_with_background_color_adds_box_params(self):
        strip = Strip(
            type="subtitle", content="Sub", size=32, color="#ffffff",
            background_color="black@0.5",
            position={"x": 0, "y": 680},
        )
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["box"] == 1
        assert call_kwargs["boxcolor"] == "black@0.5"

    def test_subtitle_with_hex_background_color(self):
        strip = Strip(
            type="subtitle", content="Caption", background_color="#000000",
            position={"x": 0, "y": 0},
        )
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["box"] == 1
        assert call_kwargs["boxcolor"] == "#000000"

    def test_text_type_does_not_add_box_even_if_background_set(self):
        """For plain text strips, background_color attribute does not add box."""
        strip = Strip(
            type="text", content="Plain", background_color="red",
            position={"x": 0, "y": 0},
        )
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        # text type must not emit box/boxcolor even if background_color is set on object
        assert "box" not in call_kwargs
        assert "boxcolor" not in call_kwargs

    def test_subtitle_returns_unchanged_stream_when_no_content(self):
        strip = Strip(type="subtitle", content=None)
        stream = self._make_stream()
        result = strip.apply_text(stream)
        stream.filter.assert_not_called()
        assert result is stream

    def test_subtitle_fontfile_included_when_set(self):
        strip = Strip(
            type="subtitle", content="Cap", font="/fonts/roboto.ttf",
            position={"x": 0, "y": 0},
        )
        stream = self._make_stream()
        strip.apply_text(stream)
        call_kwargs = stream.filter.call_args[1]
        assert call_kwargs["fontfile"] == "/fonts/roboto.ttf"


# ---------------------------------------------------------------------------
# get_strips_from_json – subtitle type
# ---------------------------------------------------------------------------

class TestGetStripsFromJsonSubtitle:
    def _make_json(self, include_optional=True):
        asset = {"type": "subtitle", "content": "Caption Text"}
        if include_optional:
            asset.update(
                {
                    "font": "/path/to/font.ttf",
                    "size": 36,
                    "color": "#ffffff",
                    "background_color": "black@0.6",
                    "position": {"x": "center", "y": 650},
                    "animation": "fadeIn",
                }
            )
        return {
            "timeline": {
                "n_frames": 25,
                "tracks": [
                    {
                        "track_id": 1,
                        "strips": [
                            {"asset": asset, "start": 0, "length": 25},
                        ],
                    }
                ],
            }
        }

    def test_subtitle_strip_parsed_correctly(self):
        strips = get_strips_from_json(self._make_json())
        assert len(strips) == 1
        s = strips[0]
        assert s.type == "subtitle"
        assert s.content == "Caption Text"
        assert s.font == "/path/to/font.ttf"
        assert s.size == 36
        assert s.color == "#ffffff"
        assert s.background_color == "black@0.6"
        assert s.position == {"x": "center", "y": 650}
        assert s.animation == "fadeIn"
        assert s.media_source is None

    def test_subtitle_strip_defaults_applied(self):
        strips = get_strips_from_json(self._make_json(include_optional=False))
        s = strips[0]
        assert s.content == "Caption Text"
        assert s.font is None
        assert s.size == 24
        assert s.color == "white"
        assert s.background_color is None
        assert s.position == {"x": 0, "y": 0}
        assert s.animation is None


# ---------------------------------------------------------------------------
# Sequence.render_strip_list – subtitle rendering
# ---------------------------------------------------------------------------

class TestSequenceRenderSubtitle:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_subtitle_strip_rendered_with_background_box(self):
        image_stream = MagicMock()
        image_stream.filter.return_value = image_stream

        image_strip = Strip(type="image", media_source="img.jpg", track_id=0)
        subtitle_strip = Strip(
            type="subtitle",
            content="Caption",
            size=32,
            color="white",
            background_color="black",
            position={"x": "center", "y": 650},
            track_id=1,
        )

        seq = Sequence(strips=[], n_frame=1, temp_dir=self._tmpdir)
        with patch.object(image_strip, "get_frame", return_value=image_stream):
            result = seq.render_strip_list([image_strip, subtitle_strip], frame=0)

        image_stream.filter.assert_called_once()
        call_args = image_stream.filter.call_args
        assert call_args[0][0] == "drawtext"
        assert call_args[1]["text"] == "Caption"
        assert call_args[1]["box"] == 1
        assert call_args[1]["boxcolor"] == "black"

    def test_subtitle_strip_without_base_image_returns_none(self):
        subtitle_strip = Strip(type="subtitle", content="Cap", track_id=0)
        seq = Sequence(strips=[], n_frame=1, temp_dir=self._tmpdir)
        result = seq.render_strip_list([subtitle_strip], frame=0)
        assert result is None
