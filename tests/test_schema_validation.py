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
