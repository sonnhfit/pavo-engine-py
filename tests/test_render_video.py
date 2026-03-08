"""Unit tests for the render_video public API (pavo/pavo.py)."""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pavo.pavo import render_video, _add_audio_to_video, _create_background_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    with open(path, "w") as fh:
        json.dump(data, fh)


MINIMAL_TIMELINE = {
    "timeline": {
        "n_frames": 5,
        "background": "#000000",
        "tracks": [
            {
                "track_id": 0,
                "strips": [
                    {
                        "asset": {"type": "image", "src": "fake.jpg"},
                        "start": 0,
                        "video_start_frame": 0,
                        "length": 5,
                        "effect": None,
                        "transition": {},
                    }
                ],
            }
        ],
    },
    "output": {"format": "mp4", "fps": 25, "width": 320, "height": 240},
}


# ---------------------------------------------------------------------------
# render_video – error handling
# ---------------------------------------------------------------------------

class TestRenderVideoErrors:
    def test_missing_json_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            render_video("/nonexistent/path/timeline.json", "/tmp/out.mp4")

    def test_invalid_json_content_raises_value_error(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("this is not json {{{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            render_video(str(bad_json), str(tmp_path / "out.mp4"))

    def test_json_missing_timeline_key_raises_value_error(self, tmp_path):
        no_timeline = tmp_path / "no_timeline.json"
        _write_json(str(no_timeline), {"output": {}})
        with pytest.raises(ValueError, match="'timeline'"):
            render_video(str(no_timeline), str(tmp_path / "out.mp4"))


# ---------------------------------------------------------------------------
# render_video – valid JSON (mocked rendering)
# ---------------------------------------------------------------------------

class TestRenderVideoValid:
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_calls_render_and_render_strips(self, mock_strips, mock_render, tmp_path):
        """render_video should call render() and render_video_from_strips()."""
        mock_render.return_value = [None] * 5

        json_file = tmp_path / "timeline.json"
        _write_json(str(json_file), MINIMAL_TIMELINE)
        output = str(tmp_path / "out.mp4")

        render_video(str(json_file), output)

        assert mock_render.called
        assert mock_strips.called

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_fps_and_background_forwarded(self, mock_strips, mock_render, tmp_path):
        """fps and background colour from the JSON are forwarded correctly."""
        mock_render.return_value = []

        timeline = {
            "timeline": {"n_frames": 2, "background": "#ff0000", "tracks": []},
            "output": {"fps": 30, "width": 640, "height": 480},
        }
        json_file = tmp_path / "timeline.json"
        _write_json(str(json_file), timeline)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        _, kwargs = mock_strips.call_args
        assert kwargs.get("fps") == 30
        assert kwargs.get("background") == "#ff0000"

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_temp_dir_cleaned_up(self, mock_strips, mock_render, tmp_path):
        """Temporary directories created during rendering are removed afterwards."""
        mock_render.return_value = []

        json_file = tmp_path / "timeline.json"
        _write_json(str(json_file), MINIMAL_TIMELINE)
        output = str(tmp_path / "out.mp4")

        render_video(str(json_file), output)

        # No pavo_ temp directories should remain under tmp_path
        leftovers = [
            d for d in os.listdir(str(tmp_path)) if d.startswith("pavo_")
        ]
        assert leftovers == []


# ---------------------------------------------------------------------------
# render_video – audio handling
# ---------------------------------------------------------------------------

class TestRenderVideoAudio:
    @patch("pavo.pavo._add_audio_to_video")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_soundtrack_triggers_audio_merge(
        self, mock_strips, mock_render, mock_audio, tmp_path
    ):
        """When the JSON has a soundtrack, _add_audio_to_video must be called."""
        mock_render.return_value = []

        timeline = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "soundtrack": {"src": "music.mp3", "effect": "fadeOut"},
                "tracks": [],
            },
            "output": {"fps": 25, "width": 320, "height": 240},
        }
        json_file = tmp_path / "with_audio.json"
        _write_json(str(json_file), timeline)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_audio.assert_called_once()
        args, _ = mock_audio.call_args
        assert args[1] == "music.mp3"

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_no_soundtrack_skips_audio_merge(self, mock_strips, mock_render, tmp_path):
        """When the JSON has no soundtrack, _add_audio_to_video must NOT be called."""
        mock_render.return_value = []

        json_file = tmp_path / "no_audio.json"
        _write_json(str(json_file), MINIMAL_TIMELINE)

        with patch("pavo.pavo._add_audio_to_video") as mock_audio:
            render_video(str(json_file), str(tmp_path / "out.mp4"))
            mock_audio.assert_not_called()


# ---------------------------------------------------------------------------
# render_video – edge cases
# ---------------------------------------------------------------------------

class TestRenderVideoEdgeCases:
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_empty_tracks_renders_without_error(
        self, mock_strips, mock_render, tmp_path
    ):
        """An empty tracks list should not raise any exception."""
        mock_render.return_value = []

        timeline = {
            "timeline": {
                "n_frames": 0,
                "background": "#000000",
                "tracks": [],
            },
            "output": {"fps": 25},
        }
        json_file = tmp_path / "empty.json"
        _write_json(str(json_file), timeline)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_missing_output_defaults_used(self, mock_strips, mock_render, tmp_path):
        """Missing 'output' key should fall back to defaults (no exception)."""
        mock_render.return_value = []

        timeline = {
            "timeline": {
                "n_frames": 1,
                "background": "#000000",
                "tracks": [],
            }
        }
        json_file = tmp_path / "no_output.json"
        _write_json(str(json_file), timeline)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        _, kwargs = mock_strips.call_args
        assert kwargs.get("fps") == 25  # default

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_render_error_cleans_up_temp(self, mock_strips, mock_render, tmp_path):
        """Even when rendering raises, temp directories should be cleaned up."""
        mock_render.side_effect = RuntimeError("render failed")

        json_file = tmp_path / "timeline.json"
        _write_json(str(json_file), MINIMAL_TIMELINE)

        with pytest.raises(RuntimeError):
            render_video(str(json_file), str(tmp_path / "out.mp4"))

        leftovers = [
            d for d in os.listdir(str(tmp_path)) if d.startswith("pavo_")
        ]
        assert leftovers == []
