"""Tests for the 15 new JSON-to-video features added to pavo-engine-py.

Covers:
  1.  Effects & Animation (image effects + text animations)
  2.  Audio strips on the timeline
  3.  Soundtrack effects (schema validation)
  4.  Auto-compute n_frames
  5.  Scale & Fit per strip
  6.  Color grading / filters per strip
  7.  Video looping
  8.  Speed control
  9.  Subtitle import from SRT/VTT files
  10. Progress callback in render_video()
  11. GIF output schema field
  12. Watermark / logo overlay
  13. Improved text rendering (stroke, shadow, line_spacing)
  14. Schema validation improvements
  15. Parallel frame rendering
"""

from __future__ import annotations

import os
import tempfile
import warnings
from unittest.mock import MagicMock, patch, call

import pytest

from pavo.schema import validate_timeline_json, TimelineFileModel
from pavo.sequancer.seq import Strip, Sequence, _parse_srt, _parse_vtt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal(n_frames=10, **track_overrides):
    """Return a valid minimal timeline dict."""
    return {
        "timeline": {
            "n_frames": n_frames,
            "background": "#000000",
            "tracks": [
                {
                    "track_id": 0,
                    "strips": [
                        {
                            "asset": {"type": "image", "src": "img.jpg"},
                            "start": 0,
                            "length": n_frames,
                        }
                    ],
                }
            ],
        },
        "output": {"fps": 25, "width": 640, "height": 480},
    }


def _make_stream():
    """Return a chainable mock ffmpeg stream."""
    s = MagicMock()
    s.filter.return_value = s
    s.overlay.return_value = s
    return s


# ---------------------------------------------------------------------------
# Feature 1 – Effects & Animation
# ---------------------------------------------------------------------------

class TestMediaEffects:
    """Image / video motion effects: zoomIn, zoomOut, panLeft, panRight, slideUp, slideDown."""

    @pytest.mark.parametrize("effect", ["zoomIn", "zoomOut", "panLeft", "panRight", "slideUp", "slideDown"])
    def test_effect_calls_filter(self, effect):
        """_apply_media_effect must call filter() on the stream for each effect."""
        strip = Strip(type="image", media_source="img.jpg", start_frame=0, length=20, effect=effect)
        stream = _make_stream()
        result = strip._apply_media_effect(stream, frame=10)
        # At least one filter should have been applied.
        assert stream.filter.called

    def test_no_effect_returns_unchanged_stream(self):
        strip = Strip(type="image", media_source="img.jpg", start_frame=0, length=20)
        stream = _make_stream()
        result = strip._apply_media_effect(stream, frame=0)
        stream.filter.assert_not_called()
        assert result is stream

    def test_progress_calculation(self):
        strip = Strip(start_frame=0, length=11)
        assert strip._get_progress(0) == pytest.approx(0.0)
        assert strip._get_progress(5) == pytest.approx(0.5)
        assert strip._get_progress(10) == pytest.approx(1.0)

    def test_progress_clamped_to_0_1(self):
        strip = Strip(start_frame=5, length=10)
        assert strip._get_progress(4) == pytest.approx(0.0)  # before strip start
        assert strip._get_progress(20) == pytest.approx(1.0)  # beyond strip end


class TestTextAnimations:
    """Text animations: typewriter, fadeIn, fadeOut, slideInLeft."""

    def _strip(self, animation):
        return Strip(
            type="text",
            content="Hello World",
            start_frame=0,
            length=10,
            animation=animation,
            color="white",
            size=24,
        )

    def test_typewriter_at_start_shows_partial_content(self):
        strip = self._strip("typewriter")
        stream = _make_stream()
        # At frame 0 (progress≈0), only 1 character should be shown.
        strip.apply_text(stream, frame=0)
        args, kwargs = stream.filter.call_args
        # The text drawn should be a prefix of the original.
        assert kwargs["text"] in "Hello World"
        assert len(kwargs["text"]) >= 1

    def test_typewriter_at_end_shows_full_content(self):
        strip = self._strip("typewriter")
        stream = _make_stream()
        strip.apply_text(stream, frame=9)
        _, kwargs = stream.filter.call_args
        assert kwargs["text"] == "Hello World"

    def test_fadein_at_start_has_low_alpha(self):
        strip = self._strip("fadeIn")
        stream = _make_stream()
        strip.apply_text(stream, frame=0)
        _, kwargs = stream.filter.call_args
        assert "alpha" in kwargs
        assert kwargs["alpha"] < 0.15

    def test_fadein_at_end_has_no_alpha_kwarg(self):
        """At frame=9 (progress=1.0), alpha=1.0 so the kwarg is NOT added."""
        strip = self._strip("fadeIn")
        stream = _make_stream()
        strip.apply_text(stream, frame=9)
        _, kwargs = stream.filter.call_args
        assert "alpha" not in kwargs

    def test_fadeout_at_start_has_no_alpha_kwarg(self):
        """At frame=0 (progress=0.0, 1-progress=1.0), alpha is 1.0 → not added."""
        strip = self._strip("fadeOut")
        stream = _make_stream()
        strip.apply_text(stream, frame=0)
        _, kwargs = stream.filter.call_args
        assert "alpha" not in kwargs

    def test_fadeout_at_end_has_low_alpha(self):
        strip = self._strip("fadeOut")
        stream = _make_stream()
        strip.apply_text(stream, frame=9)
        _, kwargs = stream.filter.call_args
        assert "alpha" in kwargs
        assert kwargs["alpha"] < 0.15

    def test_slideinleft_at_start_has_offset_x(self):
        strip = self._strip("slideInLeft")
        stream = _make_stream()
        strip.apply_text(stream, frame=0)
        _, kwargs = stream.filter.call_args
        # x expression should encode a negative offset when progress ~ 0.
        assert "w *" in kwargs["x"] or "w*" in kwargs["x"]

    def test_apply_text_without_content_returns_img(self):
        strip = Strip(type="text", content=None, start_frame=0, length=5)
        stream = _make_stream()
        result = strip.apply_text(stream, frame=0)
        assert result is stream
        stream.filter.assert_not_called()


# ---------------------------------------------------------------------------
# Feature 2 – Audio strips
# ---------------------------------------------------------------------------

class TestAudioStrips:
    def test_audio_type_is_valid_in_schema(self):
        data = {
            "timeline": {
                "n_frames": 50,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "audio", "src": "voice.mp3"},
                                "start": 0,
                                "length": 50,
                            }
                        ],
                    }
                ],
            },
            "output": {"fps": 25},
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.type == "audio"

    def test_audio_strip_volume_in_schema(self):
        data = {
            "timeline": {
                "n_frames": 50,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "audio", "src": "sfx.mp3", "volume": 0.5},
                                "start": 10,
                                "length": 25,
                            }
                        ],
                    }
                ],
            },
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.volume == 0.5

    def test_audio_strip_not_rendered_as_video_frame(self, tmp_path):
        """Audio strips should be excluded from the visual frame cache."""
        strip = Strip(type="audio", media_source="voice.mp3", start_frame=0, length=10)
        img_strip = Strip(type="image", media_source="img.jpg", start_frame=0, length=10)
        seq = Sequence(strips=[strip, img_strip], n_frame=10, temp_dir=str(tmp_path))
        seq.init_strip_dict_by_frame()
        # Only the image strip should appear in the frame dict.
        for frame_strips in seq.strips_dict_by_frame.values():
            assert all(s.type != "audio" for s in frame_strips)

    def test_audio_strip_volume_not_valid_on_image(self):
        data = _minimal(n_frames=10)
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["volume"] = 0.5
        with pytest.raises(ValueError, match="volume"):
            validate_timeline_json(data)

    def test_get_audio_strips_from_json(self):
        from pavo.sequancer.render import get_audio_strips_from_json
        json_data = {
            "timeline": {
                "n_frames": 50,
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {"asset": {"type": "image", "src": "img.jpg"}, "start": 0, "length": 25},
                            {"asset": {"type": "audio", "src": "bg.mp3"}, "start": 0, "length": 50},
                        ],
                    }
                ],
            },
        }
        audio_strips = get_audio_strips_from_json(json_data)
        assert len(audio_strips) == 1
        assert audio_strips[0].type == "audio"
        assert audio_strips[0].media_source == "bg.mp3"


# ---------------------------------------------------------------------------
# Feature 3 – Soundtrack effects
# ---------------------------------------------------------------------------

class TestSoundtrackEffects:
    def test_fadein_effect_valid(self):
        data = _minimal()
        data["timeline"]["soundtrack"] = {"src": "music.mp3", "effect": "fadeIn"}
        result = validate_timeline_json(data)
        assert result.timeline.soundtrack.effect == "fadeIn"

    def test_fadeout_effect_valid(self):
        data = _minimal()
        data["timeline"]["soundtrack"] = {"src": "music.mp3", "effect": "fadeOut"}
        result = validate_timeline_json(data)
        assert result.timeline.soundtrack.effect == "fadeOut"

    def test_fadeinout_effect_valid(self):
        data = _minimal()
        data["timeline"]["soundtrack"] = {"src": "music.mp3", "effect": "fadeInOut"}
        result = validate_timeline_json(data)
        assert result.timeline.soundtrack.effect == "fadeInOut"

    def test_invalid_effect_raises(self):
        data = _minimal()
        data["timeline"]["soundtrack"] = {"src": "music.mp3", "effect": "echo"}
        with pytest.raises(ValueError, match="unsupported soundtrack effect"):
            validate_timeline_json(data)

    @patch("pavo.pavo.ffmpeg")
    def test_apply_soundtrack_effect_copies_when_unknown_effect(self, mock_ffmpeg):
        """For an unrecognised effect, the file should be copied unchanged."""
        import shutil
        from pavo.pavo import _apply_soundtrack_effect
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as src_f:
            src_f.write(b"dummy")
            src_path = src_f.name
        try:
            with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as dst_f:
                dst_path = dst_f.name
            _apply_soundtrack_effect(src_path, "unknown", dst_path)
            with open(dst_path, "rb") as f:
                assert f.read() == b"dummy"
        finally:
            os.unlink(src_path)
            if os.path.exists(dst_path):
                os.unlink(dst_path)


# ---------------------------------------------------------------------------
# Feature 4 – Auto-compute n_frames
# ---------------------------------------------------------------------------

class TestAutoNFrames:
    def test_n_frames_auto_computed_from_strips(self):
        data = {
            "timeline": {
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {"asset": {"type": "image", "src": "img.jpg"}, "start": 0, "length": 30},
                            {"asset": {"type": "image", "src": "img.jpg"}, "start": 30, "length": 20},
                        ],
                    }
                ],
            },
        }
        result = validate_timeline_json(data)
        assert result.timeline.n_frames == 50

    def test_n_frames_auto_computed_empty_tracks(self):
        data = {"timeline": {"background": "#000000", "tracks": []}}
        result = validate_timeline_json(data)
        assert result.timeline.n_frames == 0

    def test_n_frames_explicit_value_respected(self):
        data = _minimal(n_frames=42)
        result = validate_timeline_json(data)
        assert result.timeline.n_frames == 42

    def test_render_auto_n_frames(self):
        from pavo.sequancer.render import get_strips_from_json, _auto_n_frames
        json_data = {
            "timeline": {
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {"asset": {"type": "image", "src": "a.jpg"}, "start": 5, "length": 15},
                        ],
                    }
                ],
            },
        }
        n = _auto_n_frames(json_data)
        assert n == 20  # start=5, length=15 → end=20


# ---------------------------------------------------------------------------
# Feature 5 – Scale & Fit
# ---------------------------------------------------------------------------

class TestFitMode:
    def test_fit_field_valid_in_schema(self):
        for fit in ("cover", "contain", "stretch", "none"):
            data = _minimal()
            data["timeline"]["tracks"][0]["strips"][0]["asset"]["fit"] = fit
            result = validate_timeline_json(data)
            assert result.timeline.tracks[0].strips[0].asset.fit == fit

    def test_fit_invalid_value_raises(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["fit"] = "fill"
        with pytest.raises(ValueError, match="unsupported fit mode"):
            validate_timeline_json(data)

    def test_fit_not_valid_on_text(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "text",
            "content": "Hi",
            "fit": "cover",
        }
        with pytest.raises(ValueError, match="fit"):
            validate_timeline_json(data)

    def test_apply_fit_stretch_calls_scale(self):
        strip = Strip(type="image", media_source="img.jpg", fit="stretch")
        stream = _make_stream()
        result = strip._apply_fit(stream, output_width=640, output_height=480)
        assert stream.filter.called
        args, _ = stream.filter.call_args
        assert "scale" in args

    def test_apply_fit_cover_calls_filters(self):
        strip = Strip(type="image", media_source="img.jpg", fit="cover")
        stream = _make_stream()
        strip._apply_fit(stream, output_width=640, output_height=480)
        assert stream.filter.call_count >= 1

    def test_apply_fit_contain_calls_pad(self):
        strip = Strip(type="image", media_source="img.jpg", fit="contain")
        stream = _make_stream()
        strip._apply_fit(stream, output_width=640, output_height=480)
        call_names = [c.args[0] for c in stream.filter.call_args_list]
        assert "pad" in call_names

    def test_apply_fit_none_returns_unchanged(self):
        strip = Strip(type="image", media_source="img.jpg", fit="none")
        stream = _make_stream()
        result = strip._apply_fit(stream, output_width=640, output_height=480)
        stream.filter.assert_not_called()
        assert result is stream

    def test_apply_fit_without_dimensions_returns_unchanged(self):
        strip = Strip(type="image", media_source="img.jpg", fit="stretch")
        stream = _make_stream()
        result = strip._apply_fit(stream, output_width=None, output_height=None)
        stream.filter.assert_not_called()


# ---------------------------------------------------------------------------
# Feature 6 – Color grading / filters
# ---------------------------------------------------------------------------

class TestColorFilters:
    def test_filters_schema_valid(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["filters"] = {
            "brightness": 0.1,
            "contrast": 1.2,
            "saturation": 0.8,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.filters.brightness == pytest.approx(0.1)

    def test_filters_grayscale_schema_valid(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["filters"] = {"grayscale": True}
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.filters.grayscale is True

    def test_filters_sepia_schema_valid(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["filters"] = {"sepia": True}
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.filters.sepia is True

    def test_filters_brightness_out_of_range_raises(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["filters"] = {"brightness": 2.0}
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_filters_not_valid_on_text(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "text",
            "content": "Hi",
            "filters": {"grayscale": True},
        }
        with pytest.raises(ValueError, match="filters"):
            validate_timeline_json(data)

    def test_apply_filters_brightness_calls_eq(self):
        strip = Strip(type="image", filters={"brightness": 0.2})
        stream = _make_stream()
        strip._apply_filters(stream)
        call_names = [c.args[0] for c in stream.filter.call_args_list]
        assert "eq" in call_names

    def test_apply_filters_grayscale_calls_hue(self):
        strip = Strip(type="image", filters={"grayscale": True})
        stream = _make_stream()
        strip._apply_filters(stream)
        call_names = [c.args[0] for c in stream.filter.call_args_list]
        assert "hue" in call_names

    def test_apply_filters_sepia_calls_colorchannelmixer(self):
        strip = Strip(type="image", filters={"sepia": True})
        stream = _make_stream()
        strip._apply_filters(stream)
        call_names = [c.args[0] for c in stream.filter.call_args_list]
        assert "colorchannelmixer" in call_names

    def test_apply_filters_blur_calls_gblur(self):
        strip = Strip(type="image", filters={"blur": 3.0})
        stream = _make_stream()
        strip._apply_filters(stream)
        call_names = [c.args[0] for c in stream.filter.call_args_list]
        assert "gblur" in call_names

    def test_apply_filters_none_returns_unchanged(self):
        strip = Strip(type="image", filters=None)
        stream = _make_stream()
        result = strip._apply_filters(stream)
        stream.filter.assert_not_called()
        assert result is stream


# ---------------------------------------------------------------------------
# Feature 7 – Video looping
# ---------------------------------------------------------------------------

class TestVideoLooping:
    def test_loop_valid_in_schema(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "video",
            "src": "clip.mp4",
            "loop": True,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.loop is True

    def test_loop_not_valid_on_image(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["loop"] = True
        with pytest.raises(ValueError, match="loop"):
            validate_timeline_json(data)

    def test_strip_stores_loop(self):
        strip = Strip(type="video", loop=True)
        assert strip.loop is True

    def test_strip_loop_defaults_false(self):
        strip = Strip(type="video")
        assert strip.loop is False


# ---------------------------------------------------------------------------
# Feature 8 – Speed control
# ---------------------------------------------------------------------------

class TestSpeedControl:
    def test_speed_valid_in_schema(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "video",
            "src": "clip.mp4",
            "speed": 2.0,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.speed == pytest.approx(2.0)

    def test_speed_must_be_positive(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "video",
            "src": "clip.mp4",
            "speed": 0.0,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_speed_not_valid_on_image(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["speed"] = 1.5
        with pytest.raises(ValueError, match="speed"):
            validate_timeline_json(data)

    def test_strip_stores_speed(self):
        strip = Strip(type="video", speed=0.5)
        assert strip.speed == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Feature 9 – SRT/VTT subtitle parsing
# ---------------------------------------------------------------------------

class TestSubtitleParsing:
    SRT_TEXT = """1
00:00:01,000 --> 00:00:04,000
Hello from SRT.

2
00:00:05,000 --> 00:00:08,500
Second line.
"""

    VTT_TEXT = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello from VTT.

00:00:05.000 --> 00:00:08.500
Second line.
"""

    def test_parse_srt_returns_correct_entries(self):
        entries = _parse_srt(self.SRT_TEXT)
        assert len(entries) == 2
        assert entries[0] == pytest.approx((1.0, 4.0, "Hello from SRT."), abs=0.01)
        assert entries[1][2] == "Second line."

    def test_parse_vtt_returns_correct_entries(self):
        entries = _parse_vtt(self.VTT_TEXT)
        assert len(entries) == 2
        assert entries[0][0] == pytest.approx(1.0, abs=0.01)
        assert entries[0][2] == "Hello from VTT."

    def test_srt_subtitle_file_valid_in_schema(self):
        data = {
            "timeline": {
                "n_frames": 250,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "image", "src": "img.jpg"},
                                "start": 0,
                                "length": 250,
                            },
                        ],
                    },
                    {
                        "track_id": 1,
                        "strips": [
                            {
                                "asset": {"type": "subtitle", "src": "captions.srt"},
                                "start": 0,
                                "length": 250,
                            },
                        ],
                    },
                ],
            },
            "output": {"fps": 25},
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[1].strips[0].asset.src == "captions.srt"

    def test_get_subtitle_content_returns_active_line(self):
        strip = Strip(type="subtitle", start_frame=0, length=250)
        strip._subtitle_entries = [(1.0, 4.0, "Hello"), (5.0, 8.5, "World")]
        # Frame 50 at 25fps = 2.0 seconds → "Hello" is active
        content = strip.get_subtitle_content(frame=50, fps=25.0)
        assert content == "Hello"

    def test_get_subtitle_content_returns_none_outside_range(self):
        strip = Strip(type="subtitle", start_frame=0, length=250)
        strip._subtitle_entries = [(5.0, 8.5, "Hello")]
        # Frame 0 at 25fps = 0.0 seconds → no subtitle active
        content = strip.get_subtitle_content(frame=0, fps=25.0)
        assert content is None

    def test_subtitle_file_loaded_and_used_in_apply_text(self, tmp_path):
        srt_file = tmp_path / "subs.srt"
        srt_file.write_text(self.SRT_TEXT, encoding="utf-8")
        strip = Strip(
            type="subtitle",
            media_source=str(srt_file),
            start_frame=0,
            length=250,
            color="white",
            size=24,
        )
        stream = _make_stream()
        # Frame 50 at 25fps = t=2.0s → "Hello from SRT." is active
        strip.apply_text(stream, frame=50, fps=25.0)
        _, kwargs = stream.filter.call_args
        assert kwargs["text"] == "Hello from SRT."


# ---------------------------------------------------------------------------
# Feature 10 – Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_on_progress_passed_to_render_video_from_strips(
        self, mock_strips, mock_render, tmp_path
    ):
        from pavo.pavo import render_video
        import json

        mock_render.return_value = []
        timeline = {
            "timeline": {"n_frames": 5, "background": "#000000", "tracks": []},
            "output": {"fps": 25},
        }
        json_file = tmp_path / "t.json"
        json_file.write_text(json.dumps(timeline))

        progress_calls = []
        render_video(str(json_file), str(tmp_path / "out.mp4"),
                     on_progress=lambda p: progress_calls.append(p))

        _, kwargs = mock_strips.call_args
        assert kwargs.get("on_progress") is not None


# ---------------------------------------------------------------------------
# Feature 11 – GIF output
# ---------------------------------------------------------------------------

class TestGifOutput:
    def test_gif_format_valid_in_schema(self):
        data = _minimal()
        data["output"]["format"] = "gif"
        result = validate_timeline_json(data)
        assert result.output.format == "gif"

    @patch("pavo.pavo._render_gif")
    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_gif_output_calls_render_gif(self, mock_strips, mock_render, mock_gif, tmp_path):
        from pavo.pavo import render_video
        import json

        mock_render.return_value = []
        timeline = {
            "timeline": {"n_frames": 5, "background": "#000000", "tracks": []},
            "output": {"fps": 15, "format": "gif", "width": 320},
        }
        json_file = tmp_path / "t.json"
        json_file.write_text(json.dumps(timeline))

        render_video(str(json_file), str(tmp_path / "out.gif"))

        mock_gif.assert_called_once()
        _, kwargs = mock_gif.call_args
        assert kwargs.get("fps") == 15 or mock_gif.call_args[0][2] == 15


# ---------------------------------------------------------------------------
# Feature 12 – Watermark / logo overlay
# ---------------------------------------------------------------------------

class TestWatermark:
    def test_watermark_type_valid_in_schema(self):
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
                                    "type": "watermark",
                                    "src": "logo.png",
                                    "opacity": 0.7,
                                    "position": {"x": 10, "y": 10},
                                },
                                "start": 0,
                                "length": 10,
                            }
                        ],
                    }
                ],
            },
        }
        result = validate_timeline_json(data)
        asset = result.timeline.tracks[0].strips[0].asset
        assert asset.type == "watermark"
        assert asset.opacity == pytest.approx(0.7)

    def test_watermark_requires_src(self):
        data = {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "watermark"},
                                "start": 0,
                                "length": 10,
                            }
                        ],
                    }
                ],
            },
        }
        with pytest.raises(ValueError, match="src"):
            validate_timeline_json(data)

    def test_opacity_not_valid_on_video(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "video",
            "src": "clip.mp4",
            "opacity": 0.5,
        }
        with pytest.raises(ValueError, match="opacity"):
            validate_timeline_json(data)

    def test_apply_opacity_calls_colorchannelmixer(self):
        strip = Strip(type="watermark", opacity=0.5)
        stream = _make_stream()
        strip._apply_opacity(stream)
        call_names = [c.args[0] for c in stream.filter.call_args_list]
        assert "colorchannelmixer" in call_names

    def test_apply_opacity_1_returns_unchanged(self):
        strip = Strip(type="watermark", opacity=1.0)
        stream = _make_stream()
        result = strip._apply_opacity(stream)
        assert result is stream
        stream.filter.assert_not_called()

    def test_watermark_strip_get_frame_returns_stream(self):
        strip = Strip(type="watermark", media_source="logo.png", opacity=0.8, start_frame=0, length=10)
        with patch.object(strip, "read_image", return_value=_make_stream()):
            result = strip.get_frame(frame=0)
        assert result is not None


# ---------------------------------------------------------------------------
# Feature 13 – Improved text rendering (stroke, shadow, line_spacing)
# ---------------------------------------------------------------------------

class TestImprovedTextRendering:
    def test_stroke_color_in_schema(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "text",
            "content": "Hi",
            "stroke_color": "red",
            "stroke_width": 3,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.stroke_color == "red"
        assert result.timeline.tracks[0].strips[0].asset.stroke_width == 3

    def test_shadow_in_schema(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "text",
            "content": "Hi",
            "shadow": {"x": 3, "y": 3, "color": "black"},
        }
        result = validate_timeline_json(data)
        shadow = result.timeline.tracks[0].strips[0].asset.shadow
        assert shadow.x == 3
        assert shadow.color == "black"

    def test_line_spacing_in_schema(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "text",
            "content": "Line 1\nLine 2",
            "line_spacing": 8,
        }
        result = validate_timeline_json(data)
        assert result.timeline.tracks[0].strips[0].asset.line_spacing == 8

    def test_stroke_not_valid_on_image(self):
        data = _minimal()
        data["timeline"]["tracks"][0]["strips"][0]["asset"]["stroke_color"] = "red"
        with pytest.raises(ValueError, match="stroke"):
            validate_timeline_json(data)

    def test_apply_text_includes_border_for_stroke(self):
        strip = Strip(
            type="text",
            content="Hello",
            start_frame=0,
            length=10,
            stroke_color="red",
            stroke_width=2,
            color="white",
            size=24,
        )
        stream = _make_stream()
        strip.apply_text(stream, frame=0)
        _, kwargs = stream.filter.call_args
        assert kwargs.get("bordercolor") == "red"
        assert kwargs.get("borderw") == 2

    def test_apply_text_includes_shadow_kwargs(self):
        strip = Strip(
            type="text",
            content="Hello",
            start_frame=0,
            length=10,
            shadow={"x": 2, "y": 3, "color": "gray"},
            color="white",
            size=24,
        )
        stream = _make_stream()
        strip.apply_text(stream, frame=0)
        _, kwargs = stream.filter.call_args
        assert kwargs.get("shadowx") == 2
        assert kwargs.get("shadowy") == 3
        assert kwargs.get("shadowcolor") == "gray"

    def test_apply_text_includes_line_spacing(self):
        strip = Strip(
            type="text",
            content="Hello\nWorld",
            start_frame=0,
            length=10,
            line_spacing=10,
            color="white",
            size=24,
        )
        stream = _make_stream()
        strip.apply_text(stream, frame=0)
        _, kwargs = stream.filter.call_args
        assert kwargs.get("line_spacing") == 10


# ---------------------------------------------------------------------------
# Feature 14 – Schema validation improvements
# ---------------------------------------------------------------------------

class TestSchemaValidationImprovements:
    def test_strip_exceeds_n_frames_raises(self):
        data = _minimal(n_frames=10)
        data["timeline"]["tracks"][0]["strips"][0]["length"] = 15  # 0+15=15 > 10
        with pytest.raises(ValueError, match="exceeds n_frames"):
            validate_timeline_json(data)

    def test_strip_exactly_at_n_frames_boundary_is_valid(self):
        data = _minimal(n_frames=10)
        # start=0, length=10 → end=10, n_frames=10 → 10 > 10 is False → valid
        result = validate_timeline_json(data)
        assert result.timeline.n_frames == 10

    def test_trim_end_less_than_trim_start_raises(self):
        data = _minimal(n_frames=50)
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "video",
            "src": "clip.mp4",
            "trim_start": 5.0,
            "trim_end": 3.0,
        }
        with pytest.raises(ValueError, match="trim_end.*trim_start|trim_end must be greater"):
            validate_timeline_json(data)

    def test_trim_end_equal_to_trim_start_raises(self):
        data = _minimal(n_frames=50)
        data["timeline"]["tracks"][0]["strips"][0]["asset"] = {
            "type": "video",
            "src": "clip.mp4",
            "trim_start": 3.0,
            "trim_end": 3.0,
        }
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_overlapping_strips_on_same_track_warns(self):
        data = {
            "timeline": {
                "n_frames": 30,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {"asset": {"type": "image", "src": "a.jpg"}, "start": 0, "length": 20},
                            {"asset": {"type": "image", "src": "b.jpg"}, "start": 10, "length": 20},
                        ],
                    }
                ],
            },
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_timeline_json(data)
            overlap_warnings = [x for x in w if issubclass(x.category, UserWarning)
                                and "overlapping" in str(x.message).lower()]
            assert len(overlap_warnings) >= 1

    def test_non_overlapping_strips_no_warning(self):
        data = {
            "timeline": {
                "n_frames": 30,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {"asset": {"type": "image", "src": "a.jpg"}, "start": 0, "length": 15},
                            {"asset": {"type": "image", "src": "b.jpg"}, "start": 15, "length": 15},
                        ],
                    }
                ],
            },
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_timeline_json(data)
            overlap_warnings = [x for x in w if "overlapping" in str(getattr(x, "message", "")).lower()]
            assert len(overlap_warnings) == 0


# ---------------------------------------------------------------------------
# Feature 15 – Parallel frame rendering
# ---------------------------------------------------------------------------

class TestParallelRendering:
    def test_workers_field_in_schema(self):
        data = _minimal()
        data["output"]["workers"] = 4
        result = validate_timeline_json(data)
        assert result.output.workers == 4

    def test_workers_must_be_positive(self):
        data = _minimal()
        data["output"]["workers"] = 0
        with pytest.raises(ValueError):
            validate_timeline_json(data)

    def test_render_sequence_accepts_workers_param(self):
        """Sequence.render_sequence(workers=N) should not raise."""
        seq = Sequence(strips=[], n_frame=0, temp_dir=tempfile.mkdtemp())
        result = seq.render_sequence(workers=2)
        assert result == []

    def test_parallel_render_produces_same_frame_count(self):
        """With workers>1, all frames should still be populated."""
        strips = [
            Strip(type="image", media_source="img.jpg", start_frame=0, length=5, track_id=0)
        ]
        seq = Sequence(strips=strips, n_frame=5, temp_dir=tempfile.mkdtemp())
        seq.sort_strips_by_start_frame()
        seq.init_strip_dict_by_frame()

        # Patch get_frame to return a dummy value per frame.
        with patch.object(seq, "get_frame", side_effect=lambda f: f) as mock_gf:
            result = seq.render_sequence(workers=2)

        assert len(result) == 5
        assert mock_gf.call_count == 5
        # All frame indices should appear.
        assert set(result) == {0, 1, 2, 3, 4}
