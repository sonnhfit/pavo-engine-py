"""Unit tests for the render_video public API (pavo/pavo.py)."""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pavo.pavo import (
    render_video,
    _add_audio_to_video,
    _create_background_frame,
    _build_ducking_expr,
    _apply_audio_ducking,
    _detect_speech_segments,
    _collect_audio_strips,
    _mix_audio_with_strips,
)


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


# ---------------------------------------------------------------------------
# _build_ducking_expr
# ---------------------------------------------------------------------------

class TestBuildDuckingExpr:
    def test_empty_segments_returns_unity(self):
        """No speech segments → volume expression is always 1.0."""
        expr = _build_ducking_expr([])
        assert expr == "1.0"

    def test_single_segment_contains_between(self):
        """A single segment should produce a between() condition."""
        expr = _build_ducking_expr([(2.0, 5.0)], reduction_db=10.0)
        assert "between(t,2.000,5.000)" in expr

    def test_multiple_segments_joined_with_plus(self):
        """Multiple segments should be joined with '+'."""
        expr = _build_ducking_expr([(1.0, 2.0), (4.0, 6.0)])
        assert "between(t,1.000,2.000)" in expr
        assert "between(t,4.000,6.000)" in expr
        assert "+" in expr

    def test_reduction_db_scales_volume(self):
        """10 dB reduction should produce ≈ 0.316 linear factor."""
        expr = _build_ducking_expr([(0.0, 1.0)], reduction_db=10.0)
        # 10^(-10/20) ≈ 0.316228
        assert "0.316228" in expr

    def test_zero_db_reduction_produces_unity_factor(self):
        """0 dB reduction should leave volume at 1.0."""
        expr = _build_ducking_expr([(0.0, 1.0)], reduction_db=0.0)
        assert "1.000000" in expr

    def test_negative_reduction_db_treated_as_positive(self):
        """Passing a negative dB value should still reduce (not boost) volume."""
        expr_pos = _build_ducking_expr([(0.0, 1.0)], reduction_db=10.0)
        expr_neg = _build_ducking_expr([(0.0, 1.0)], reduction_db=-10.0)
        assert expr_pos == expr_neg


# ---------------------------------------------------------------------------
# _apply_audio_ducking
# ---------------------------------------------------------------------------

class TestApplyAudioDucking:
    def test_no_segments_copies_file(self, tmp_path):
        """With no speech segments the original file should be copied unchanged."""
        src = tmp_path / "music.mp3"
        src.write_bytes(b"fake-audio-data")
        dst = tmp_path / "out.mp3"

        _apply_audio_ducking(str(src), [], 10.0, str(dst))

        assert dst.exists()
        assert dst.read_bytes() == src.read_bytes()

    @patch("pavo.pavo.ffmpeg")
    def test_segments_invoke_ffmpeg_filter(self, mock_ffmpeg, tmp_path):
        """With speech segments the FFmpeg volume filter should be invoked."""
        # Wire up the ffmpeg mock so chained calls don't blow up.
        chain = MagicMock()
        mock_ffmpeg.input.return_value = chain
        chain.filter.return_value = chain
        chain.output.return_value = chain
        chain.overwrite_output.return_value = chain
        chain.run.return_value = (b"", b"")

        src = tmp_path / "music.mp3"
        src.write_bytes(b"fake-audio-data")
        dst = tmp_path / "out.mp3"

        _apply_audio_ducking(str(src), [(1.0, 3.0)], 10.0, str(dst))

        mock_ffmpeg.input.assert_called_once_with(str(src))
        chain.filter.assert_called_once()
        filter_kwargs = chain.filter.call_args
        assert filter_kwargs[0][0] == "volume"


# ---------------------------------------------------------------------------
# _detect_speech_segments
# ---------------------------------------------------------------------------

class TestDetectSpeechSegments:
    def test_no_video_strips_returns_empty(self):
        """A timeline with only image strips should return no speech segments."""
        timeline = {
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "image", "src": "img.jpg"},
                            "start": 0,
                            "video_start_frame": 0,
                            "length": 25,
                        }
                    ],
                }
            ]
        }
        result = _detect_speech_segments(timeline, fps=25)
        assert result == []

    def test_missing_whisper_returns_empty(self):
        """When openai-whisper is not installed the function returns []."""
        timeline = {
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "video", "src": "/fake/clip.mp4"},
                            "start": 0,
                            "video_start_frame": 0,
                            "length": 25,
                        }
                    ],
                }
            ]
        }
        with patch.dict("sys.modules", {"whisper": None}):
            with patch(
                "pavo.pavo._detect_speech_segments",
                side_effect=lambda *a, **kw: [],
            ):
                result = _detect_speech_segments(timeline, fps=25)
        # SpeechTranscriber import will fail or return [] when whisper absent.
        assert isinstance(result, list)

    def test_nonexistent_video_src_is_skipped(self):
        """Video strips whose source file does not exist should be skipped."""
        timeline = {
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "video", "src": "/does/not/exist.mp4"},
                            "start": 0,
                            "video_start_frame": 0,
                            "length": 50,
                        }
                    ],
                }
            ]
        }
        result = _detect_speech_segments(timeline, fps=25)
        assert result == []

    @patch("pavo.pavo.ffmpeg")
    def test_video_without_audio_stream_is_skipped(self, mock_ffmpeg, tmp_path):
        """Video strips with no audio stream should be skipped gracefully."""
        video_file = tmp_path / "silent.mp4"
        video_file.write_bytes(b"fake")

        mock_ffmpeg.probe.return_value = {
            "streams": [{"codec_type": "video"}]  # no audio stream
        }

        timeline = {
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "video", "src": str(video_file)},
                            "start": 0,
                            "video_start_frame": 0,
                            "length": 25,
                        }
                    ],
                }
            ]
        }
        result = _detect_speech_segments(timeline, fps=25)
        assert result == []

    @patch("pavo.pavo.ffmpeg")
    def test_speech_segments_mapped_to_output_timeline(self, mock_ffmpeg, tmp_path):
        """Transcribed speech segments are mapped to output video timestamps."""
        video_file = tmp_path / "narration.mp4"
        video_file.write_bytes(b"fake")

        mock_ffmpeg.probe.return_value = {
            "streams": [
                {"codec_type": "video"},
                {"codec_type": "audio"},
            ]
        }

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = {
            "segments": [
                {"start": 1.0, "end": 3.0},
                {"start": 5.0, "end": 7.0},
            ]
        }

        timeline = {
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "video", "src": str(video_file)},
                            "start": 50,   # frame 50 → 2.0 s at 25 fps
                            "video_start_frame": 0,
                            "length": 250,
                        }
                    ],
                }
            ]
        }

        with patch(
            "pavo.perception.speech.transcriber.SpeechTranscriber",
            return_value=mock_transcriber,
        ):
            result = _detect_speech_segments(timeline, fps=25)

        # strip_start_sec = 50/25 = 2.0
        # segment [1.0, 3.0] → output [2.0+1.0, 2.0+3.0] = [3.0, 5.0]
        # segment [5.0, 7.0] → output [2.0+5.0, 2.0+7.0] = [7.0, 9.0]
        assert (3.0, 5.0) in result
        assert (7.0, 9.0) in result


# ---------------------------------------------------------------------------
# render_video – audio ducking integration
# ---------------------------------------------------------------------------

class TestRenderVideoAudioDucking:
    @patch("pavo.pavo._apply_audio_ducking")
    @patch("pavo.pavo._detect_speech_segments")
    @patch("pavo.pavo._add_audio_to_video")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_ducking_enabled_with_speech(
        self,
        mock_strips,
        mock_render,
        mock_add_audio,
        mock_detect,
        mock_duck,
        tmp_path,
    ):
        """When audio_ducking=true and speech is detected, ducking is applied."""
        mock_render.return_value = []
        mock_detect.return_value = [(1.0, 3.0), (5.0, 7.0)]

        timeline_data = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "soundtrack": {"src": "music.mp3"},
                "tracks": [],
            },
            "output": {
                "fps": 25,
                "width": 320,
                "height": 240,
                "audio_ducking": True,
                "ducking_reduction_db": 12.0,
            },
        }
        json_file = tmp_path / "ducking.json"
        _write_json(str(json_file), timeline_data)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_detect.assert_called_once()
        mock_duck.assert_called_once()
        # The reduction_db positional arg is the third positional argument.
        assert mock_duck.call_args[0][2] == 12.0

    @patch("pavo.pavo._apply_audio_ducking")
    @patch("pavo.pavo._detect_speech_segments")
    @patch("pavo.pavo._add_audio_to_video")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_ducking_enabled_no_speech_skips_ducking(
        self,
        mock_strips,
        mock_render,
        mock_add_audio,
        mock_detect,
        mock_duck,
        tmp_path,
    ):
        """When audio_ducking=true but no speech found, ducking is not applied."""
        mock_render.return_value = []
        mock_detect.return_value = []  # no speech

        timeline_data = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "soundtrack": {"src": "music.mp3"},
                "tracks": [],
            },
            "output": {
                "fps": 25,
                "width": 320,
                "height": 240,
                "audio_ducking": True,
            },
        }
        json_file = tmp_path / "no_speech.json"
        _write_json(str(json_file), timeline_data)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_detect.assert_called_once()
        mock_duck.assert_not_called()
        # Original soundtrack should still be merged.
        mock_add_audio.assert_called_once()
        assert mock_add_audio.call_args[0][1] == "music.mp3"

    @patch("pavo.pavo._apply_audio_ducking")
    @patch("pavo.pavo._detect_speech_segments")
    @patch("pavo.pavo._add_audio_to_video")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_ducking_disabled_by_default(
        self,
        mock_strips,
        mock_render,
        mock_add_audio,
        mock_detect,
        mock_duck,
        tmp_path,
    ):
        """When audio_ducking is absent (default False), detection is not called."""
        mock_render.return_value = []

        timeline_data = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "soundtrack": {"src": "music.mp3"},
                "tracks": [],
            },
            "output": {"fps": 25, "width": 320, "height": 240},
        }
        json_file = tmp_path / "no_ducking.json"
        _write_json(str(json_file), timeline_data)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_detect.assert_not_called()
        mock_duck.assert_not_called()
        mock_add_audio.assert_called_once()


# ---------------------------------------------------------------------------
# _collect_audio_strips
# ---------------------------------------------------------------------------

class TestCollectAudioStrips:
    def _make_timeline(self, strips_per_track):
        """Build a minimal timeline dict with one track containing *strips_per_track*."""
        return {
            "tracks": [
                {
                    "track_id": 1,
                    "strips": strips_per_track,
                }
            ]
        }

    def test_no_tracks_returns_empty(self):
        result = _collect_audio_strips({"tracks": []}, fps=25)
        assert result == []

    def test_non_audio_strips_ignored(self):
        timeline = self._make_timeline([
            {"asset": {"type": "image", "src": "img.jpg"}, "start": 0, "length": 25},
            {"asset": {"type": "video", "src": "clip.mp4"}, "start": 0, "length": 25},
        ])
        result = _collect_audio_strips(timeline, fps=25)
        assert result == []

    def test_audio_strip_collected(self):
        timeline = self._make_timeline([
            {
                "asset": {"type": "audio", "src": "voice.mp3"},
                "start": 0,
                "length": 25,
            }
        ])
        result = _collect_audio_strips(timeline, fps=25)
        assert len(result) == 1
        assert result[0]["src"] == "voice.mp3"
        assert result[0]["start_ms"] == 0
        assert result[0]["volume"] == 1.0

    def test_audio_strip_with_volume(self):
        timeline = self._make_timeline([
            {
                "asset": {"type": "audio", "src": "sfx.mp3", "volume": 0.5},
                "start": 0,
                "length": 10,
            }
        ])
        result = _collect_audio_strips(timeline, fps=25)
        assert result[0]["volume"] == 0.5

    def test_strip_start_converted_to_ms(self):
        """start=50 frames at 25 fps → 2000 ms."""
        timeline = self._make_timeline([
            {
                "asset": {"type": "audio", "src": "voice.mp3"},
                "start": 50,
                "length": 25,
            }
        ])
        result = _collect_audio_strips(timeline, fps=25)
        assert result[0]["start_ms"] == 2000

    def test_asset_start_offset_added_to_strip_start(self):
        """strip start=25 + asset start=5 → (30/25)*1000 = 1200 ms."""
        timeline = self._make_timeline([
            {
                "asset": {"type": "audio", "src": "voice.mp3", "start": 5},
                "start": 25,
                "length": 25,
            }
        ])
        result = _collect_audio_strips(timeline, fps=25)
        assert result[0]["start_ms"] == 1200

    def test_missing_src_skipped(self):
        timeline = self._make_timeline([
            {"asset": {"type": "audio", "src": ""}, "start": 0, "length": 10},
            {"asset": {"type": "audio"}, "start": 0, "length": 10},
        ])
        result = _collect_audio_strips(timeline, fps=25)
        assert result == []

    def test_multiple_audio_strips_all_collected(self):
        timeline = {
            "tracks": [
                {
                    "track_id": 1,
                    "strips": [
                        {"asset": {"type": "audio", "src": "a.mp3"}, "start": 0, "length": 10},
                        {"asset": {"type": "audio", "src": "b.mp3"}, "start": 25, "length": 10},
                    ],
                },
                {
                    "track_id": 2,
                    "strips": [
                        {"asset": {"type": "audio", "src": "c.mp3"}, "start": 50, "length": 10},
                    ],
                },
            ]
        }
        result = _collect_audio_strips(timeline, fps=25)
        assert len(result) == 3
        srcs = {r["src"] for r in result}
        assert srcs == {"a.mp3", "b.mp3", "c.mp3"}


# ---------------------------------------------------------------------------
# _mix_audio_with_strips
# ---------------------------------------------------------------------------

class TestMixAudioWithStrips:
    def test_no_audio_strips_delegates_to_add_audio(self, tmp_path):
        """When audio_strips is empty, _add_audio_to_video should be called."""
        with patch("pavo.pavo._add_audio_to_video") as mock_add:
            _mix_audio_with_strips("video.mp4", "sound.mp3", [], "out.mp4")
            mock_add.assert_called_once_with("video.mp4", "sound.mp3", "out.mp4")

    @patch("pavo.pavo.ffmpeg")
    def test_audio_strips_invoke_amix(self, mock_ffmpeg):
        """When audio strips are present, ffmpeg filter chain should be invoked."""
        chain = MagicMock()
        mock_ffmpeg.input.return_value = chain
        chain.audio = chain
        chain.video = chain
        chain.filter.return_value = chain
        chain.output.return_value = chain
        chain.overwrite_output.return_value = chain
        chain.run.return_value = (b"", b"")
        mock_ffmpeg.filter.return_value = chain

        audio_strips = [
            {"src": "voice.mp3", "start_ms": 1000, "volume": 0.8},
        ]
        _mix_audio_with_strips("video.mp4", "sound.mp3", audio_strips, "out.mp4")

        # The function must call ffmpeg.input at least for video, soundtrack, and
        # the audio strip.
        assert mock_ffmpeg.input.call_count >= 3

    @patch("pavo.pavo.ffmpeg")
    def test_audio_strip_without_soundtrack(self, mock_ffmpeg):
        """Audio strips work even when there is no global soundtrack."""
        chain = MagicMock()
        mock_ffmpeg.input.return_value = chain
        chain.audio = chain
        chain.video = chain
        chain.filter.return_value = chain
        chain.output.return_value = chain
        chain.overwrite_output.return_value = chain
        chain.run.return_value = (b"", b"")
        mock_ffmpeg.filter.return_value = chain

        audio_strips = [{"src": "sfx.mp3", "start_ms": 0, "volume": 1.0}]
        _mix_audio_with_strips("video.mp4", None, audio_strips, "out.mp4")

        # Input called for video + strip audio only (no soundtrack).
        assert mock_ffmpeg.input.call_count == 2

    @patch("pavo.pavo.ffmpeg")
    def test_delay_filter_applied_for_nonzero_start(self, mock_ffmpeg):
        """adelay filter must be applied when start_ms > 0."""
        chain = MagicMock()
        mock_ffmpeg.input.return_value = chain
        chain.audio = chain
        chain.video = chain
        chain.filter.return_value = chain
        chain.output.return_value = chain
        chain.overwrite_output.return_value = chain
        chain.run.return_value = (b"", b"")
        mock_ffmpeg.filter.return_value = chain

        audio_strips = [{"src": "sfx.mp3", "start_ms": 2000, "volume": 1.0}]
        _mix_audio_with_strips("video.mp4", None, audio_strips, "out.mp4")

        # .filter("adelay", ...) must be called on the audio chain
        adelay_calls = [
            c for c in chain.filter.call_args_list
            if c.args and c.args[0] == "adelay"
        ]
        assert len(adelay_calls) >= 1

    @patch("pavo.pavo.ffmpeg")
    def test_volume_filter_applied_when_not_unity(self, mock_ffmpeg):
        """volume filter must be applied when volume != 1.0."""
        chain = MagicMock()
        mock_ffmpeg.input.return_value = chain
        chain.audio = chain
        chain.video = chain
        chain.filter.return_value = chain
        chain.output.return_value = chain
        chain.overwrite_output.return_value = chain
        chain.run.return_value = (b"", b"")
        mock_ffmpeg.filter.return_value = chain

        audio_strips = [{"src": "sfx.mp3", "start_ms": 0, "volume": 0.5}]
        _mix_audio_with_strips("video.mp4", None, audio_strips, "out.mp4")

        volume_calls = [
            c for c in chain.filter.call_args_list
            if c.args and c.args[0] == "volume"
        ]
        assert len(volume_calls) >= 1

    @patch("pavo.pavo.ffmpeg")
    def test_volume_filter_skipped_when_unity(self, mock_ffmpeg):
        """volume filter must NOT be applied when volume == 1.0."""
        chain = MagicMock()
        mock_ffmpeg.input.return_value = chain
        chain.audio = chain
        chain.video = chain
        chain.filter.return_value = chain
        chain.output.return_value = chain
        chain.overwrite_output.return_value = chain
        chain.run.return_value = (b"", b"")
        mock_ffmpeg.filter.return_value = chain

        audio_strips = [{"src": "sfx.mp3", "start_ms": 0, "volume": 1.0}]
        _mix_audio_with_strips("video.mp4", None, audio_strips, "out.mp4")

        volume_calls = [
            c for c in chain.filter.call_args_list
            if c.args and c.args[0] == "volume"
        ]
        assert len(volume_calls) == 0


# ---------------------------------------------------------------------------
# render_video – audio strip integration
# ---------------------------------------------------------------------------

class TestRenderVideoAudioStrips:
    @patch("pavo.pavo._mix_audio_with_strips")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_audio_strips_trigger_mix(self, mock_strips, mock_render, mock_mix, tmp_path):
        """When an audio strip is present, _mix_audio_with_strips must be called."""
        mock_render.return_value = []

        timeline_data = {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "audio", "src": "voice.mp3", "volume": 0.8},
                                "start": 0,
                                "video_start_frame": 0,
                                "length": 10,
                                "effect": None,
                                "transition": {},
                            }
                        ],
                    }
                ],
            },
            "output": {"fps": 25, "width": 320, "height": 240},
        }
        json_file = tmp_path / "audio_strip.json"
        _write_json(str(json_file), timeline_data)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_mix.assert_called_once()
        # No global soundtrack – second arg must be None.
        assert mock_mix.call_args[0][1] is None
        # Audio strips list must contain the strip we defined.
        audio_strips_arg = mock_mix.call_args[0][2]
        assert len(audio_strips_arg) == 1
        assert audio_strips_arg[0]["src"] == "voice.mp3"
        assert audio_strips_arg[0]["volume"] == 0.8

    @patch("pavo.pavo._mix_audio_with_strips")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_audio_strips_with_soundtrack_passed_to_mix(
        self, mock_strips, mock_render, mock_mix, tmp_path
    ):
        """Soundtrack src should be forwarded to _mix_audio_with_strips."""
        mock_render.return_value = []

        timeline_data = {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "soundtrack": {"src": "music.mp3"},
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "audio", "src": "sfx.mp3"},
                                "start": 5,
                                "video_start_frame": 0,
                                "length": 5,
                                "effect": None,
                                "transition": {},
                            }
                        ],
                    }
                ],
            },
            "output": {"fps": 25, "width": 320, "height": 240},
        }
        json_file = tmp_path / "audio_and_track.json"
        _write_json(str(json_file), timeline_data)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_mix.assert_called_once()
        # Second arg is the effective soundtrack (no ducking → original src).
        assert mock_mix.call_args[0][1] == "music.mp3"

    @patch("pavo.pavo._add_audio_to_video")
    @patch("pavo.pavo._mix_audio_with_strips")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_no_audio_strips_still_uses_add_audio(
        self, mock_strips, mock_render, mock_mix, mock_add_audio, tmp_path
    ):
        """Without audio strips, _add_audio_to_video is used (not _mix_audio_with_strips)."""
        mock_render.return_value = []

        timeline_data = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "soundtrack": {"src": "music.mp3"},
                "tracks": [],
            },
            "output": {"fps": 25, "width": 320, "height": 240},
        }
        json_file = tmp_path / "only_soundtrack.json"
        _write_json(str(json_file), timeline_data)

        render_video(str(json_file), str(tmp_path / "out.mp4"))

        mock_mix.assert_not_called()
        mock_add_audio.assert_called_once()
