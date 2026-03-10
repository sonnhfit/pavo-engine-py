"""Unit tests for custom transition effects (fade, slide, wipe, dissolve)."""

import tempfile
import pytest
from unittest.mock import MagicMock, call, patch

from pavo.sequancer.seq import Strip, Sequence, SUPPORTED_TRANSITIONS
from pavo.sequancer.render import get_strips_from_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream():
    """Return a mock ffmpeg stream that supports chaining."""
    stream = MagicMock()
    stream.filter.return_value = stream
    stream.overlay.return_value = stream
    return stream


def _make_strip(**kwargs):
    defaults = dict(
        type="image",
        media_source="img.jpg",
        track_id=0,
        start_frame=0,
        length=20,
        transition_duration=5,
    )
    defaults.update(kwargs)
    return Strip(**defaults)


def _make_json(transition=None, asset_type="image"):
    """Build a minimal timeline JSON dict with one strip."""
    asset = {"type": asset_type, "src": "img.jpg"} if asset_type != "text" else {
        "type": "text",
        "content": "Hello",
    }
    strip_data = {
        "asset": asset,
        "start": 0,
        "length": 20,
        "effect": None,
        "video_start_frame": 0,
    }
    if transition is not None:
        strip_data["transition"] = transition
    return {
        "timeline": {
            "n_frames": 20,
            "tracks": [{"track_id": 0, "strips": [strip_data]}],
        }
    }


# ---------------------------------------------------------------------------
# SUPPORTED_TRANSITIONS constant
# ---------------------------------------------------------------------------

class TestSupportedTransitions:
    def test_all_required_types_present(self):
        assert SUPPORTED_TRANSITIONS == {"fade", "slide", "wipe", "dissolve"}


# ---------------------------------------------------------------------------
# Strip transition attribute defaults
# ---------------------------------------------------------------------------

class TestStripTransitionAttributes:
    def test_defaults_are_none_and_five(self):
        strip = Strip()
        assert strip.transition_in is None
        assert strip.transition_out is None
        assert strip.transition_duration == 5

    def test_custom_values_stored(self):
        strip = Strip(
            transition_in="fade",
            transition_out="slide",
            transition_duration=10,
        )
        assert strip.transition_in == "fade"
        assert strip.transition_out == "slide"
        assert strip.transition_duration == 10

    def test_all_transition_types_accepted(self):
        for t in SUPPORTED_TRANSITIONS:
            strip = Strip(transition_in=t, transition_out=t)
            assert strip.transition_in == t
            assert strip.transition_out == t


# ---------------------------------------------------------------------------
# Strip._get_active_transition
# ---------------------------------------------------------------------------

class TestGetActiveTransition:
    # --- No transitions ---------------------------------------------------

    def test_returns_none_when_no_transitions_set(self):
        strip = _make_strip(start_frame=0, length=20)
        assert strip._get_active_transition(5) is None

    # --- Transition-in ---------------------------------------------------

    def test_in_transition_active_at_first_frame(self):
        strip = _make_strip(transition_in="fade", start_frame=0, length=20, transition_duration=5)
        result = strip._get_active_transition(0)
        assert result is not None
        direction, ttype, progress = result
        assert direction == "in"
        assert ttype == "fade"
        assert progress == pytest.approx(0.0)

    def test_in_transition_progress_increases(self):
        strip = _make_strip(transition_in="fade", start_frame=0, length=20, transition_duration=5)
        prog_0 = strip._get_active_transition(0)[2]
        prog_2 = strip._get_active_transition(2)[2]
        assert prog_2 > prog_0

    def test_in_transition_inactive_after_duration(self):
        strip = _make_strip(transition_in="fade", start_frame=0, length=20, transition_duration=5)
        # Frame 5 is one past the end of the 5-frame in-transition
        assert strip._get_active_transition(5) is None

    def test_in_transition_with_nonzero_start(self):
        strip = _make_strip(
            transition_in="slide", start_frame=10, length=20, transition_duration=5
        )
        result = strip._get_active_transition(10)
        assert result is not None
        assert result[0] == "in"
        assert result[2] == pytest.approx(0.0)

    def test_in_transition_type_propagated(self):
        for t in SUPPORTED_TRANSITIONS:
            strip = _make_strip(transition_in=t, start_frame=0, length=20, transition_duration=5)
            _, ttype, _ = strip._get_active_transition(0)
            assert ttype == t

    # --- Transition-out --------------------------------------------------

    def test_out_transition_active_at_last_frame(self):
        strip = _make_strip(transition_out="fade", start_frame=0, length=20, transition_duration=5)
        result = strip._get_active_transition(19)  # last frame
        assert result is not None
        direction, ttype, progress = result
        assert direction == "out"
        assert ttype == "fade"
        assert progress == pytest.approx(0.0)

    def test_out_transition_progress_decreases_towards_end(self):
        strip = _make_strip(transition_out="fade", start_frame=0, length=20, transition_duration=5)
        prog_start = strip._get_active_transition(15)[2]
        prog_end = strip._get_active_transition(19)[2]
        assert prog_start > prog_end

    def test_out_transition_inactive_before_window(self):
        strip = _make_strip(transition_out="fade", start_frame=0, length=20, transition_duration=5)
        # Frame 14 is one before the 5-frame out window (frames 15-19)
        assert strip._get_active_transition(14) is None

    def test_out_transition_type_propagated(self):
        for t in SUPPORTED_TRANSITIONS:
            strip = _make_strip(transition_out=t, start_frame=0, length=20, transition_duration=5)
            _, ttype, _ = strip._get_active_transition(19)
            assert ttype == t

    # --- Duration clamping -----------------------------------------------

    def test_duration_clamped_to_half_strip_length(self):
        """Transition duration must not exceed half of strip length."""
        strip = _make_strip(
            transition_in="fade",
            transition_out="fade",
            start_frame=0,
            length=6,
            transition_duration=10,  # exceeds half (3)
        )
        # With dur clamped to 3 (= length//2), in-transition covers frames 0-2
        # and out-transition covers frames 3-5; they are adjacent but do not overlap.
        in_result = strip._get_active_transition(0)
        assert in_result is not None
        assert in_result[0] == "in"

        # Frame 2 is the last in-transition frame
        assert strip._get_active_transition(2)[0] == "in"

        # Frame 3 is the first out-transition frame (no overlap)
        out_result = strip._get_active_transition(3)
        assert out_result is not None
        assert out_result[0] == "out"

    def test_progress_clamped_to_0_1(self):
        strip = _make_strip(transition_in="fade", start_frame=0, length=20, transition_duration=5)
        for frame in range(0, 20):
            result = strip._get_active_transition(frame)
            if result:
                _, _, progress = result
                assert 0.0 <= progress <= 1.0


# ---------------------------------------------------------------------------
# Strip._apply_fade
# ---------------------------------------------------------------------------

class TestApplyFade:
    def test_format_rgba_called(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()
        rgba_stream = MagicMock()
        rgba_stream.filter.return_value = rgba_stream
        strip_frame.filter.return_value = rgba_stream

        strip._apply_fade(base, strip_frame, 0.5)

        strip_frame.filter.assert_called_with("format", "rgba")

    def test_colorchannelmixer_called_with_alpha(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = MagicMock()
        rgba_stream = MagicMock()
        rgba_stream.filter.return_value = rgba_stream
        rgba_stream.overlay.return_value = rgba_stream
        strip_frame.filter.return_value = rgba_stream

        strip._apply_fade(base, strip_frame, 0.75)

        rgba_stream.filter.assert_called_with("colorchannelmixer", aa=0.75)

    def test_overlay_called_with_format_auto(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = MagicMock()
        rgba_stream = MagicMock()
        rgba_stream.filter.return_value = rgba_stream
        base.overlay = MagicMock(return_value=base)
        strip_frame.filter.return_value = rgba_stream

        strip._apply_fade(base, strip_frame, 0.5)

        base.overlay.assert_called_once()
        _, kwargs = base.overlay.call_args
        assert kwargs.get("format") == "auto"

    def test_alpha_clamped_to_0_1(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = MagicMock()
        rgba_stream = MagicMock()
        rgba_stream.filter.return_value = rgba_stream
        rgba_stream.overlay.return_value = rgba_stream
        strip_frame.filter.return_value = rgba_stream

        # Should not raise even with out-of-range values
        strip._apply_fade(base, strip_frame, -0.5)
        strip._apply_fade(base, strip_frame, 1.5)


# ---------------------------------------------------------------------------
# Strip._apply_slide
# ---------------------------------------------------------------------------

class TestApplySlide:
    def test_slide_in_uses_negative_overlay_w_x(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        strip._apply_slide(base, strip_frame, 0.0, "in")

        base.overlay.assert_called_once()
        _, kwargs = base.overlay.call_args
        assert "overlay_w" in kwargs["x"]
        assert kwargs["y"] == "0"

    def test_slide_in_at_full_progress_x_is_zero(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        strip._apply_slide(base, strip_frame, 1.0, "in")

        _, kwargs = base.overlay.call_args
        # offset = 1 - progress = 0.0 → x = "-overlay_w*0.000000"
        assert "0.000000" in kwargs["x"]

    def test_slide_out_uses_main_w_x(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        strip._apply_slide(base, strip_frame, 0.5, "out")

        _, kwargs = base.overlay.call_args
        assert "main_w" in kwargs["x"]
        assert kwargs["y"] == "0"

    def test_slide_out_at_zero_progress_x_is_main_w(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        strip._apply_slide(base, strip_frame, 0.0, "out")

        _, kwargs = base.overlay.call_args
        # in_progress = 1 - 0 = 1.0 → x = "main_w*1.000000"
        assert "1.000000" in kwargs["x"]


# ---------------------------------------------------------------------------
# Strip._apply_wipe
# ---------------------------------------------------------------------------

class TestApplyWipe:
    def test_wipe_applies_crop_filter(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        strip._apply_wipe(base, strip_frame, 0.5, "in")

        strip_frame.filter.assert_called_once()
        args, kwargs = strip_frame.filter.call_args
        assert args[0] == "crop"
        assert "iw" in kwargs["w"]

    def test_wipe_at_zero_progress_returns_base(self):
        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        result = strip._apply_wipe(base, strip_frame, 0.0, "in")

        strip_frame.filter.assert_not_called()
        assert result is base

    def test_wipe_overlay_at_origin(self):
        strip = _make_strip()
        base = _make_stream()
        cropped = _make_stream()
        strip_frame = MagicMock()
        strip_frame.filter.return_value = cropped

        strip._apply_wipe(base, strip_frame, 0.5, "in")

        base.overlay.assert_called_once_with(cropped, x="0", y="0")


# ---------------------------------------------------------------------------
# Strip._apply_dissolve
# ---------------------------------------------------------------------------

class TestApplyDissolve:
    def test_dissolve_calls_blend_filter(self):
        import ffmpeg as _ffmpeg

        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()
        blended = _make_stream()

        with patch.object(_ffmpeg, "filter", return_value=blended) as mock_filter:
            result = strip._apply_dissolve(base, strip_frame, 0.5)

        mock_filter.assert_called_once()
        args, kwargs = mock_filter.call_args
        assert args[1] == "blend"
        assert "all_expr" in kwargs
        assert result is blended

    def test_dissolve_blend_expr_uses_alpha(self):
        import ffmpeg as _ffmpeg

        strip = _make_strip()
        base = _make_stream()
        strip_frame = _make_stream()

        with patch.object(_ffmpeg, "filter", return_value=_make_stream()) as mock_filter:
            strip._apply_dissolve(base, strip_frame, 0.4)

        _, kwargs = mock_filter.call_args
        assert "0.400000" in kwargs["all_expr"]


# ---------------------------------------------------------------------------
# Strip.apply_transition_overlay – integration
# ---------------------------------------------------------------------------

class TestApplyTransitionOverlay:
    def test_no_transition_falls_back_to_plain_overlay(self):
        strip = _make_strip(start_frame=0, length=20)  # no transitions set
        base = _make_stream()
        strip_frame = _make_stream()

        result = strip.apply_transition_overlay(base, strip_frame, 5)

        base.overlay.assert_called_once_with(strip_frame)

    def test_fade_in_dispatches_to_apply_fade(self):
        strip = _make_strip(
            transition_in="fade", start_frame=0, length=20, transition_duration=5
        )
        base = _make_stream()
        strip_frame = _make_stream()

        with patch.object(strip, "_apply_fade", return_value=base) as mock_fade:
            strip.apply_transition_overlay(base, strip_frame, 0)  # first frame → in-transition

        mock_fade.assert_called_once()

    def test_slide_out_dispatches_to_apply_slide(self):
        strip = _make_strip(
            transition_out="slide", start_frame=0, length=20, transition_duration=5
        )
        base = _make_stream()
        strip_frame = _make_stream()

        with patch.object(strip, "_apply_slide", return_value=base) as mock_slide:
            strip.apply_transition_overlay(base, strip_frame, 19)  # last frame → out-transition

        mock_slide.assert_called_once()
        args = mock_slide.call_args[0]
        assert args[3] == "out"  # direction argument (index 3 after self, base, frame, progress)

    def test_wipe_in_dispatches_to_apply_wipe(self):
        strip = _make_strip(
            transition_in="wipe", start_frame=0, length=20, transition_duration=5
        )
        base = _make_stream()
        strip_frame = _make_stream()

        with patch.object(strip, "_apply_wipe", return_value=base) as mock_wipe:
            strip.apply_transition_overlay(base, strip_frame, 2)

        mock_wipe.assert_called_once()

    def test_dissolve_out_dispatches_to_apply_dissolve(self):
        strip = _make_strip(
            transition_out="dissolve", start_frame=0, length=20, transition_duration=5
        )
        base = _make_stream()
        strip_frame = _make_stream()

        with patch.object(strip, "_apply_dissolve", return_value=base) as mock_dissolve:
            strip.apply_transition_overlay(base, strip_frame, 18)

        mock_dissolve.assert_called_once()

    def test_unknown_type_falls_back_to_plain_overlay(self):
        strip = _make_strip(
            transition_in="unknown_type", start_frame=0, length=20, transition_duration=5
        )
        base = _make_stream()
        strip_frame = _make_stream()

        result = strip.apply_transition_overlay(base, strip_frame, 0)

        base.overlay.assert_called_once_with(strip_frame)


# ---------------------------------------------------------------------------
# JSON parsing – transition fields
# ---------------------------------------------------------------------------

class TestGetStripsFromJsonTransitions:
    def test_fade_in_and_out_parsed(self):
        json_data = _make_json({"in": "fade", "out": "fade"})
        strips = get_strips_from_json(json_data)
        s = strips[0]
        assert s.transition_in == "fade"
        assert s.transition_out == "fade"

    def test_slide_transition_parsed(self):
        json_data = _make_json({"in": "slide", "out": "slide"})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_in == "slide"
        assert strips[0].transition_out == "slide"

    def test_wipe_transition_parsed(self):
        json_data = _make_json({"in": "wipe", "out": "wipe"})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_in == "wipe"
        assert strips[0].transition_out == "wipe"

    def test_dissolve_transition_parsed(self):
        json_data = _make_json({"in": "dissolve", "out": "dissolve"})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_in == "dissolve"
        assert strips[0].transition_out == "dissolve"

    def test_custom_duration_parsed(self):
        json_data = _make_json({"in": "fade", "out": "fade", "duration": 8})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_duration == 8

    def test_default_duration_when_omitted(self):
        json_data = _make_json({"in": "fade"})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_duration == 5

    def test_missing_transition_field_uses_defaults(self):
        json_data = _make_json(transition=None)
        strips = get_strips_from_json(json_data)
        s = strips[0]
        assert s.transition_in is None
        assert s.transition_out is None
        assert s.transition_duration == 5

    def test_empty_transition_dict_uses_defaults(self):
        json_data = _make_json(transition={})
        strips = get_strips_from_json(json_data)
        s = strips[0]
        assert s.transition_in is None
        assert s.transition_out is None

    def test_only_out_transition_specified(self):
        json_data = _make_json({"out": "wipe"})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_in is None
        assert strips[0].transition_out == "wipe"

    def test_text_strip_transition_parsed(self):
        json_data = _make_json({"in": "fade", "out": "dissolve"}, asset_type="text")
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_in == "fade"
        assert strips[0].transition_out == "dissolve"

    def test_invalid_duration_string_falls_back_to_default(self):
        json_data = _make_json({"in": "fade", "duration": "not_a_number"})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_duration == 5

    def test_invalid_duration_none_falls_back_to_default(self):
        json_data = _make_json({"in": "fade", "duration": None})
        strips = get_strips_from_json(json_data)
        assert strips[0].transition_duration == 5


# ---------------------------------------------------------------------------
# Sequence.render_strip_list – transition integration
# ---------------------------------------------------------------------------

class TestSequenceRenderStripListTransitions:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_seq(self):
        return Sequence(strips=[], n_frame=20, temp_dir=self._tmpdir)

    def test_overlay_strip_without_transition_uses_plain_overlay(self):
        """A strip with no transition should call overlay() directly."""
        base_stream = MagicMock()
        base_stream.overlay.return_value = base_stream
        overlay_stream = MagicMock()

        base_strip = Strip(type="image", media_source="base.jpg", track_id=0)
        overlay_strip = Strip(type="image", media_source="over.jpg", track_id=1,
                               start_frame=0, length=20)

        seq = self._make_seq()

        with patch.object(base_strip, "get_frame", return_value=base_stream), \
             patch.object(overlay_strip, "get_frame", return_value=overlay_stream):
            seq.render_strip_list([base_strip, overlay_strip], frame=5)

        base_stream.overlay.assert_called_once_with(overlay_stream)

    def test_overlay_strip_with_fade_in_uses_apply_transition_overlay(self):
        """A strip with fade-in should route through apply_transition_overlay."""
        base_stream = MagicMock()
        base_stream.overlay.return_value = base_stream
        overlay_stream = MagicMock()

        base_strip = Strip(type="image", media_source="base.jpg", track_id=0)
        overlay_strip = Strip(
            type="image", media_source="over.jpg", track_id=1,
            start_frame=0, length=20,
            transition_in="fade", transition_duration=5,
        )

        seq = self._make_seq()

        with patch.object(base_strip, "get_frame", return_value=base_stream), \
             patch.object(overlay_strip, "get_frame", return_value=overlay_stream), \
             patch.object(overlay_strip, "apply_transition_overlay",
                          return_value=base_stream) as mock_tov:
            seq.render_strip_list([base_strip, overlay_strip], frame=0)

        mock_tov.assert_called_once_with(base_stream, overlay_stream, 0)
