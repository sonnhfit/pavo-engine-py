"""Unit tests for Pydantic-based timeline JSON schema validation (pavo/schema.py)."""
import pytest

from pavo.schema import validate_timeline_json, TimelineFileModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_timeline(**overrides):
    """Return a valid minimal timeline dict, optionally overriding top-level keys."""
    data = {
        "timeline": {
            "n_frames": 10,
            "background": "#000000",
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "image", "src": "img.jpg"},
                            "start": 0,
                            "length": 10,
                        }
                    ],
                }
            ],
        },
        "output": {"fps": 25, "width": 640, "height": 480},
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Valid documents
# ---------------------------------------------------------------------------

class TestValidDocuments:
    def test_minimal_image_timeline(self):
        result = validate_timeline_json(_minimal_timeline())
        assert isinstance(result, TimelineFileModel)
        assert result.timeline.n_frames == 10

    def test_text_strip(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "text", "content": "Hello"},
            "start": 0,
            "length": 5,
        }
        result = validate_timeline_json(data)
        strip = result.timeline.tracks[0].strips[0]
        assert strip.asset.content == "Hello"

    def test_text_strip_with_all_optional_fields(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {
                "type": "text",
                "content": "World",
                "font": "/fonts/roboto.ttf",
                "size": 48,
                "color": "#ff0000",
                "position": {"x": "center", "y": 50},
                "animation": "fadeIn",
            },
            "start": 0,
            "length": 10,
        }
        result = validate_timeline_json(data)
        asset = result.timeline.tracks[0].strips[0].asset
        assert asset.color == "#ff0000"
        assert asset.position.x == "center"
        assert asset.position.y == 50

    def test_video_strip(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4"},
            "start": 0,
            "length": 25,
            "video_start_frame": 5,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].video_start_frame == 5

    def test_video_strip_trim_start_seconds(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_start": 2.5},
            "start": 0,
            "length": 25,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.trim_start == 2.5
        assert result.timeline.tracks[0].strips[0].asset.trim_end is None

    def test_video_strip_trim_end_seconds(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_end": 10.0},
            "start": 0,
            "length": 25,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.trim_end == 10.0
        assert result.timeline.tracks[0].strips[0].asset.trim_start is None

    def test_video_strip_trim_start_and_end_seconds(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_start": 1.0, "trim_end": 5.0},
            "start": 0,
            "length": 25,
        }
        result = validate_timeline_json(data)
        asset = result.timeline.tracks[0].strips[0].asset
        assert asset.trim_start == 1.0
        assert asset.trim_end == 5.0

    def test_video_strip_trim_start_frame(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_start_frame": 30},
            "start": 0,
            "length": 25,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.trim_start_frame == 30

    def test_video_strip_trim_end_frame(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_end_frame": 150},
            "start": 0,
            "length": 25,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.trim_end_frame == 150

    def test_video_strip_trim_start_frame_and_end_frame(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_start_frame": 25, "trim_end_frame": 100},
            "start": 0,
            "length": 25,
        }
        result = validate_timeline_json(data)
        asset = result.timeline.tracks[0].strips[0].asset
        assert asset.trim_start_frame == 25
        assert asset.trim_end_frame == 100

    def test_transition_fields(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["transition"] = {
            "in": "fade",
            "out": "slide",
            "duration": 8,
        }
        result = validate_timeline_json(data)
        t = result.timeline.tracks[0].strips[0].transition
        assert t.in_ == "fade"
        assert t.out == "slide"
        assert t.duration == 8

    def test_soundtrack(self):
        data = _minimal_timeline()
        data["timeline"]["soundtrack"] = {"src": "music.mp3", "effect": "fadeOut"}
        result = validate_timeline_json(data)
        assert result.timeline.soundtrack.src == "music.mp3"

    def test_output_defaults(self):
        data = _minimal_timeline()
        data.pop("output", None)
        result = validate_timeline_json(data)
        assert result.output is None

    def test_audio_ducking_fields(self):
        data = _minimal_timeline()
        data["output"]["audio_ducking"] = True
        data["output"]["ducking_reduction_db"] = 15.0
        result = validate_timeline_json(data)
        assert result.output.audio_ducking is True
        assert result.output.ducking_reduction_db == 15.0
        assert result.output.ducking_reduction_db == 15.0

    def test_multiple_tracks(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"].append(
            {
                "track_id": 1,
                "strips": [
                    {
                        "asset": {"type": "text", "content": "Track 2"},
                        "start": 0,
                        "length": 5,
                    }
                ],
            }
        )
        result = validate_timeline_json(data)
        assert len(result.timeline.tracks) == 2

    def test_position_center_both_axes(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {
                "type": "text",
                "content": "Centered",
                "position": {"x": "center", "y": "center"},
            },
            "start": 0,
            "length": 5,
        }
        result = validate_timeline_json(data)
        pos = result.timeline.tracks[0].strips[0].asset.position
        assert pos.x == "center"
        assert pos.y == "center"


# ---------------------------------------------------------------------------
# Invalid documents – missing required fields
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    def test_missing_timeline_key(self):
        with pytest.raises(ValueError, match="timeline"):
            validate_timeline_json({"output": {}})

    def test_missing_n_frames(self):
        data = _minimal_timeline()
        del data["timeline"]["n_frames"]
        with pytest.raises(ValueError, match="n_frames"):
            validate_timeline_json(data)

    def test_missing_tracks(self):
        # tracks defaults to empty list when omitted
        data = _minimal_timeline()
        del data["timeline"]["tracks"]
        result = validate_timeline_json(data)
        assert result.timeline.tracks == []

    def test_empty_tracks_list(self):
        # Empty tracks list is allowed
        data = _minimal_timeline()
        data["timeline"]["tracks"] = []
        result = validate_timeline_json(data)
        assert result.timeline.tracks == []

    def test_missing_strip_start(self):
        data = _minimal_timeline()
        del data["timeline"]["tracks"][0]["strips"][0]["start"]
        with pytest.raises(ValueError, match="start"):
            validate_timeline_json(data)

    def test_missing_strip_length(self):
        data = _minimal_timeline()
        del data["timeline"]["tracks"][0]["strips"][0]["length"]
        with pytest.raises(ValueError, match="length"):
            validate_timeline_json(data)

    def test_text_asset_missing_content(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "text"},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="content"):
            validate_timeline_json(data)

    def test_image_asset_missing_src(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "image"},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="src"):
            validate_timeline_json(data)

    def test_video_asset_missing_src(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video"},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="src"):
            validate_timeline_json(data)


# ---------------------------------------------------------------------------
# Invalid documents – wrong field values
# ---------------------------------------------------------------------------

class TestInvalidFieldValues:
    def test_n_frames_zero(self):
        # n_frames=0 is allowed (empty render)
        data = _minimal_timeline()
        data["timeline"]["n_frames"] = 0
        result = validate_timeline_json(data)
        assert result.timeline.n_frames == 0

    def test_n_frames_negative(self):
        data = _minimal_timeline()
        data["timeline"]["n_frames"] = -5
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_strip_length_zero(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["length"] = 0
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_strip_start_negative(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["start"] = -1
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_invalid_asset_type(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["type"] = "audio"
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_invalid_transition_in(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["transition"] = {"in": "zoom"}
        with pytest.raises(ValueError, match="unsupported transition type"):
            validate_timeline_json(data)

    def test_invalid_transition_out(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["transition"] = {"out": "rotate"}
        with pytest.raises(ValueError, match="unsupported transition type"):
            validate_timeline_json(data)

    def test_transition_duration_zero(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0]["transition"] = {"duration": 0}
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_fps_zero(self):
        data = _minimal_timeline()
        data["output"]["fps"] = 0
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_fps_negative(self):
        data = _minimal_timeline()
        data["output"]["fps"] = -10
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_output_width_zero(self):
        data = _minimal_timeline()
        data["output"]["width"] = 0
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_output_height_negative(self):
        data = _minimal_timeline()
        data["output"]["height"] = -1
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_background_without_hash(self):
        data = _minimal_timeline()
        data["timeline"]["background"] = "000000"
        with pytest.raises(ValueError, match="background"):
            validate_timeline_json(data)

    def test_position_invalid_string(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {
                "type": "text",
                "content": "Hi",
                "position": {"x": "left", "y": 0},
            },
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="center"):
            validate_timeline_json(data)

    def test_text_asset_size_zero(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "text", "content": "Hi", "size": 0},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_track_id_negative(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["track_id"] = -1
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_error_message_is_descriptive(self):
        """The raised ValueError should contain 'Timeline JSON validation failed'."""
        data = _minimal_timeline()
        data["timeline"]["n_frames"] = -1
        with pytest.raises(ValueError, match="Timeline JSON validation failed"):
            validate_timeline_json(data)

    def test_trim_fields_on_non_video_asset(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "image", "src": "img.jpg", "trim_start": 1.0},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="trim"):
            validate_timeline_json(data)

    def test_trim_fields_on_text_asset(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "text", "content": "Hi", "trim_start_frame": 5},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="trim"):
            validate_timeline_json(data)

    def test_trim_start_negative_seconds(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_start": -1.0},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_trim_end_zero_seconds(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_end": 0},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_trim_start_frame_negative(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_start_frame": -1},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_trim_end_frame_zero(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {"type": "video", "src": "clip.mp4", "trim_end_frame": 0},
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_trim_start_and_trim_start_frame_both_set(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {
                "type": "video",
                "src": "clip.mp4",
                "trim_start": 1.0,
                "trim_start_frame": 25,
            },
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="trim_start"):
            validate_timeline_json(data)

    def test_trim_end_and_trim_end_frame_both_set(self):
        data = _minimal_timeline()
        data["timeline"]["tracks"][0]["strips"][0] = {
            "asset": {
                "type": "video",
                "src": "clip.mp4",
                "trim_end": 5.0,
                "trim_end_frame": 125,
            },
            "start": 0,
            "length": 5,
        }
        with pytest.raises(ValueError, match="trim_end"):
            validate_timeline_json(data)


# ---------------------------------------------------------------------------
# Animation preset validation
# ---------------------------------------------------------------------------

class TestAnimationPresets:
    def _text_strip_with_animation(self, animation):
        data = {
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
                                    "content": "Hello",
                                    "animation": animation,
                                },
                                "start": 0,
                                "length": 10,
                            }
                        ],
                    }
                ],
            }
        }
        return data

    def test_fadein_preset_is_valid(self):
        data = self._text_strip_with_animation("fadeIn")
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.animation == "fadeIn"

    def test_slideup_preset_is_valid(self):
        data = self._text_strip_with_animation("slideUp")
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.animation == "slideUp"

    def test_typewriter_preset_is_valid(self):
        data = self._text_strip_with_animation("typewriter")
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.animation == "typewriter"

    def test_animation_none_is_valid(self):
        data = self._text_strip_with_animation(None)
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.animation is None

    def test_unknown_preset_is_rejected(self):
        data = self._text_strip_with_animation("spinIn")
        with pytest.raises(ValueError, match="unsupported animation preset"):
            validate_timeline_json(data)

    def test_unknown_preset_error_lists_valid_options(self):
        data = self._text_strip_with_animation("zoomIn")
        with pytest.raises(ValueError, match="fadeIn"):
            validate_timeline_json(data)

    def test_empty_string_preset_is_rejected(self):
        data = self._text_strip_with_animation("")
        with pytest.raises(ValueError, match="unsupported animation preset"):
            validate_timeline_json(data)

