"""Unit tests for video trimming support in Strip and get_strips_from_json."""
import pytest
from unittest.mock import MagicMock, patch, call

from pavo.sequancer.seq import Strip
from pavo.sequancer.render import get_strips_from_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_data_with_video_asset(asset_extra=None, fps=25.0):
    """Return minimal JSON data dict with a single video strip."""
    asset = {"type": "video", "src": "clip.mp4"}
    if asset_extra:
        asset.update(asset_extra)
    return {
        "timeline": {
            "n_frames": 50,
            "background": "#000000",
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": asset,
                            "start": 0,
                            "length": 25,
                        }
                    ],
                }
            ],
        },
        "output": {"fps": fps, "width": 640, "height": 480},
    }


# ---------------------------------------------------------------------------
# Strip class – trim attribute tests
# ---------------------------------------------------------------------------

class TestStripTrimAttributes:
    def test_default_trim_values_are_none(self):
        strip = Strip(type="video", media_source="clip.mp4")
        assert strip.trim_start is None
        assert strip.trim_end is None

    def test_trim_start_stored(self):
        strip = Strip(type="video", media_source="clip.mp4", trim_start=3.5)
        assert strip.trim_start == 3.5

    def test_trim_end_stored(self):
        strip = Strip(type="video", media_source="clip.mp4", trim_end=10.0)
        assert strip.trim_end == 10.0

    def test_trim_start_and_end_stored(self):
        strip = Strip(type="video", media_source="clip.mp4", trim_start=2.0, trim_end=8.0)
        assert strip.trim_start == 2.0
        assert strip.trim_end == 8.0

    def test_trim_start_zero(self):
        strip = Strip(type="video", media_source="clip.mp4", trim_start=0.0)
        assert strip.trim_start == 0.0


# ---------------------------------------------------------------------------
# Strip.read_video_by_frame – FFmpeg input kwargs tests
# ---------------------------------------------------------------------------

class TestReadVideoByFrameTrim:
    def _setup_ffmpeg_mock_chain(self, mock_ffmpeg):
        """Wire up a minimal ffmpeg mock chain and return the input mock."""
        input_mock = MagicMock()
        filter_mock = MagicMock()
        output_mock = MagicMock()
        run_mock = MagicMock(return_value=(b"\xff\xd8\xff\xe0test", b""))

        mock_ffmpeg.input.return_value = input_mock
        input_mock.filter.return_value = filter_mock
        filter_mock.output.return_value = output_mock
        output_mock.run.return_value = (b"\xff\xd8\xff\xe0test", b"")

        return input_mock

    @patch("pavo.sequancer.seq.ffmpeg")
    def test_no_trim_calls_input_without_ss_or_to(self, mock_ffmpeg, tmp_path):
        self._setup_ffmpeg_mock_chain(mock_ffmpeg)
        strip = Strip(type="video", media_source="clip.mp4", track_id=0)
        strip.read_video_by_frame(0, temp_dir=str(tmp_path))
        mock_ffmpeg.input.assert_any_call("clip.mp4")

    @patch("pavo.sequancer.seq.ffmpeg")
    def test_trim_start_passes_ss_to_input(self, mock_ffmpeg, tmp_path):
        self._setup_ffmpeg_mock_chain(mock_ffmpeg)
        strip = Strip(type="video", media_source="clip.mp4", trim_start=5.0, track_id=0)
        strip.read_video_by_frame(0, temp_dir=str(tmp_path))
        mock_ffmpeg.input.assert_any_call("clip.mp4", ss=5.0)

    @patch("pavo.sequancer.seq.ffmpeg")
    def test_trim_end_passes_to_to_input(self, mock_ffmpeg, tmp_path):
        self._setup_ffmpeg_mock_chain(mock_ffmpeg)
        strip = Strip(type="video", media_source="clip.mp4", trim_end=12.0, track_id=0)
        strip.read_video_by_frame(0, temp_dir=str(tmp_path))
        mock_ffmpeg.input.assert_any_call("clip.mp4", to=12.0)

    @patch("pavo.sequancer.seq.ffmpeg")
    def test_trim_start_and_end_passes_both(self, mock_ffmpeg, tmp_path):
        self._setup_ffmpeg_mock_chain(mock_ffmpeg)
        strip = Strip(
            type="video", media_source="clip.mp4", trim_start=2.0, trim_end=8.0, track_id=0
        )
        strip.read_video_by_frame(0, temp_dir=str(tmp_path))
        mock_ffmpeg.input.assert_any_call("clip.mp4", ss=2.0, to=8.0)


# ---------------------------------------------------------------------------
# get_strips_from_json – trim field propagation tests
# ---------------------------------------------------------------------------

class TestGetStripsFromJsonTrim:
    def test_no_trim_fields_gives_none_trim(self):
        json_data = _json_data_with_video_asset()
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_start is None
        assert strips[0].trim_end is None

    def test_trim_start_seconds_propagated(self):
        json_data = _json_data_with_video_asset({"trim_start": 3.0})
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_start == 3.0
        assert strips[0].trim_end is None

    def test_trim_end_seconds_propagated(self):
        json_data = _json_data_with_video_asset({"trim_end": 9.0})
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_end == 9.0
        assert strips[0].trim_start is None

    def test_trim_start_and_end_seconds_propagated(self):
        json_data = _json_data_with_video_asset({"trim_start": 1.5, "trim_end": 7.5})
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_start == 1.5
        assert strips[0].trim_end == 7.5

    def test_trim_start_frame_converted_to_seconds(self):
        # 50 frames at 25 fps = 2.0 seconds
        json_data = _json_data_with_video_asset({"trim_start_frame": 50}, fps=25.0)
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_start == pytest.approx(2.0)
        assert strips[0].trim_end is None

    def test_trim_end_frame_converted_to_seconds(self):
        # 150 frames at 30 fps = 5.0 seconds
        json_data = _json_data_with_video_asset({"trim_end_frame": 150}, fps=30.0)
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_end == pytest.approx(5.0)
        assert strips[0].trim_start is None

    def test_trim_start_frame_and_trim_end_frame_converted(self):
        # trim_start_frame=25 at 25fps=1.0s; trim_end_frame=100 at 25fps=4.0s
        json_data = _json_data_with_video_asset(
            {"trim_start_frame": 25, "trim_end_frame": 100}, fps=25.0
        )
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_start == pytest.approx(1.0)
        assert strips[0].trim_end == pytest.approx(4.0)

    def test_trim_start_seconds_takes_priority_over_frame(self):
        """trim_start (seconds) is used as-is; trim_start_frame is only converted
        when trim_start is absent.  Here both are passed directly to the asset
        dict, bypassing schema validation, to verify the render pipeline precedence
        logic in get_strips_from_json.
        """
        json_data = _json_data_with_video_asset(
            {"trim_start": 2.0, "trim_start_frame": 100}, fps=25.0
        )
        strips = get_strips_from_json(json_data)
        # trim_start (seconds) wins; render.py only converts frames when seconds is absent
        assert strips[0].trim_start == 2.0

    def test_text_strip_has_no_trim_attributes(self):
        json_data = {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "text", "content": "Hi"},
                                "start": 0,
                                "length": 10,
                            }
                        ],
                    }
                ],
            },
            "output": {"fps": 25},
        }
        strips = get_strips_from_json(json_data)
        assert strips[0].trim_start is None
        assert strips[0].trim_end is None

    def test_fps_defaults_to_25_when_output_missing(self):
        """trim_start_frame conversion falls back to 25 fps when output block absent."""
        json_data = {
            "timeline": {
                "n_frames": 50,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {
                                    "type": "video",
                                    "src": "clip.mp4",
                                    "trim_start_frame": 25,
                                },
                                "start": 0,
                                "length": 25,
                            }
                        ],
                    }
                ],
            },
        }
        strips = get_strips_from_json(json_data)
        # 25 frames / 25 fps = 1.0 s
        assert strips[0].trim_start == pytest.approx(1.0)
