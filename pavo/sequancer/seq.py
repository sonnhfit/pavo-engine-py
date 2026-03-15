from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Optional,
    Any,
    Callable,
    TypeVar,
    Generic,
    Type,
)
# import numpy as np
# import cv2 as cv
import ffmpeg
import os
import sys
import time
import glob

from pavo.schema import SUPPORTED_ANIMATION_PRESETS


class Effect:
    def __init__(self):
        self.type = None


# một element trong sequence
SUPPORTED_TRANSITIONS = {"fade", "slide", "wipe", "dissolve"}

# Extra drawtext kwargs applied for each animation preset.
# "slideUp" is handled separately inside apply_text() because it requires
# modifying the computed y expression rather than adding a new keyword.
ANIMATION_PRESET_FILTERS: Dict[str, Dict[str, str]] = {
    "fadeIn": {"alpha": "if(lt(t,0.5),t/0.5,1)"},
    "slideUp": {},
    "typewriter": {"alpha": "min(1,2*t)"},
}


class Strip:
    def __init__(
        self,
        type=None,
        media_source=None,
        track_id=0,
        start_frame=0,
        length=0,
        effect=None,
        video_start_frame=0,
        content=None,
        font=None,
        size=24,
        color="white",
        position=None,
        animation=None,
        transition_in=None,
        transition_out=None,
        transition_duration=5,
        trim_start=None,
        trim_end=None,
    ):
        self.type = type
        self.media_source = media_source

        self.track_id = track_id
        self.start_frame = start_frame
        self.length = length
        self.effect = effect

        self.video_start_frame = video_start_frame

        self.video_info = None

        # Text overlay attributes
        self.content = content
        self.font = font
        self.size = size
        self.color = color
        self.position = position if position is not None else {"x": 0, "y": 0}
        self.animation = animation

        # Transition attributes
        self.transition_in = transition_in
        self.transition_out = transition_out
        self.transition_duration = transition_duration

        # Video-only trim attributes (in seconds; None means no trim on that end)
        self.trim_start = trim_start
        self.trim_end = trim_end

    def load_media_source(self):
        pass

    def read_image(self):
        img = ffmpeg.input(self.media_source)
        return img

    def read_video(self):
        pass

    def read_video_by_frame(self, frame: int, temp_dir: str = "./temp"):
        input_kwargs = {}
        if self.trim_start is not None:
            input_kwargs["ss"] = self.trim_start
        if self.trim_end is not None:
            input_kwargs["to"] = self.trim_end

        out, err = (
            ffmpeg
            .input(self.media_source, **input_kwargs)
            .filter('select', 'gte(n,{})'.format(frame))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True)
        )
        path_n = f"{temp_dir}/temp-{frame}-{self.track_id}-{int(time.time())}.jpg"
        with open(path_n, "wb") as binary_file:
            binary_file.write(out)

        out = ffmpeg.input(path_n)
        return out

    def init_strip(self):
        pass

    def get_frame(self, frame: int, temp_dir: str = "./temp"):
        if self.type == "image":
            im = self.read_image()
            return im
        elif self.type == "video":
            vid = self.read_video_by_frame(frame, temp_dir)
            return vid
        elif self.type == "text":
            return None
        else:
            return None

    def apply_text(self, img):
        """Apply this text strip's content to an existing FFmpeg stream via drawtext.

        Parameters
        ----------
        img : ffmpeg stream
            The base FFmpeg stream to draw text onto.

        Returns
        -------
        ffmpeg stream
            The stream with the ``drawtext`` filter applied.
        """
        if not self.content:
            return img

        pos = self.position if isinstance(self.position, dict) else {}
        raw_x = pos.get("x", 0)
        raw_y = pos.get("y", 0)

        # Support named positions ("center") as FFmpeg expressions.
        x = "(w-tw)/2" if str(raw_x).lower() == "center" else str(raw_x)
        y = "(h-th)/2" if str(raw_y).lower() == "center" else str(raw_y)

        # slideUp: offset y so the text starts below its final position and
        # slides upward into place over the first 0.5 seconds.
        if self.animation == "slideUp":
            y = f"({y})+50*(1-min(1,t/0.5))"

        kwargs = {
            "text": self.content,
            "fontsize": self.size,
            "fontcolor": self.color,
            "x": x,
            "y": y,
        }

        if self.font:
            kwargs["fontfile"] = self.font

        # Merge any extra drawtext kwargs defined for this animation preset.
        if self.animation and self.animation in ANIMATION_PRESET_FILTERS:
            kwargs.update(ANIMATION_PRESET_FILTERS[self.animation])

        return img.filter("drawtext", **kwargs)

    def read_video_info(self, in_filename):
        probe = ffmpeg.probe(in_filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        self.video_info = video_stream

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _get_active_transition(self, frame):
        """Return the active transition state for *frame*, or ``None``.

        Returns a ``(direction, type, progress)`` tuple when a transition is
        active:

        * ``direction`` – ``'in'`` or ``'out'``
        * ``type`` – transition name (``'fade'``, ``'slide'``, ``'wipe'``,
          ``'dissolve'``)
        * ``progress`` – float in ``[0.0, 1.0]``; ``0.0`` means the strip is
          fully hidden / off-screen, ``1.0`` means fully visible.

        The effective transition duration is clamped to at most half the
        strip's length so that *in* and *out* transitions never overlap.
        """
        max_dur = max(1, self.length // 2)
        dur = min(self.transition_duration, max_dur)
        if dur <= 0:
            return None

        frame_in_strip = frame - self.start_frame

        if self.transition_in and frame_in_strip < dur:
            progress = frame_in_strip / max(dur - 1, 1)
            return ("in", self.transition_in, max(0.0, min(1.0, progress)))

        frames_from_end = (self.start_frame + self.length) - frame
        if self.transition_out and frames_from_end <= dur:
            progress = (frames_from_end - 1) / max(dur - 1, 1)
            return ("out", self.transition_out, max(0.0, min(1.0, progress)))

        return None

    def _apply_fade(self, base_img, strip_frame, progress):
        """Fade transition: modulate strip alpha then overlay on *base_img*."""
        alpha = max(0.0, min(1.0, progress))
        faded = (
            strip_frame
            .filter("format", "rgba")
            .filter("colorchannelmixer", aa=alpha)
        )
        return base_img.overlay(faded, format="auto")

    def _apply_slide(self, base_img, strip_frame, progress, direction):
        """Slide transition: strip enters from / exits to the left edge."""
        if direction == "in":
            # Slide in from left: x goes from −overlay_w to 0.
            offset = 1.0 - progress
            x = f"-overlay_w*{offset:.6f}"
        else:
            # Slide out to right: x goes from 0 to +main_w.
            out_offset = 1.0 - progress
            x = f"main_w*{out_offset:.6f}"
        return base_img.overlay(strip_frame, x=x, y="0")

    def _apply_wipe(self, base_img, strip_frame, progress, direction):  # noqa: ARG002
        """Wipe transition: reveal / hide the strip left-to-right via crop."""
        alpha = max(0.0, min(1.0, progress))
        if alpha <= 0.0:
            return base_img
        cropped = strip_frame.filter(
            "crop", w=f"iw*{alpha:.6f}", h="ih", x="0", y="0"
        )
        return base_img.overlay(cropped, x="0", y="0")

    def _apply_dissolve(self, base_img, strip_frame, progress):
        """Dissolve transition: cross-blend strip with base image."""
        alpha = max(0.0, min(1.0, progress))
        blended = ffmpeg.filter(
            [base_img, strip_frame],
            "blend",
            all_expr=f"A*(1-{alpha:.6f})+B*{alpha:.6f}",
        )
        return blended

    def apply_transition_overlay(self, base_img, strip_frame, frame):
        """Overlay *strip_frame* on *base_img* using the active transition.

        Falls back to a plain :func:`overlay` when no transition is active at
        *frame*.

        Parameters
        ----------
        base_img : ffmpeg stream
            The composited base image stream.
        strip_frame : ffmpeg stream
            The strip's frame stream to overlay.
        frame : int
            The absolute frame number being rendered (used to compute
            transition progress).

        Returns
        -------
        ffmpeg stream
            The composited stream with the transition applied.
        """
        active = self._get_active_transition(frame)
        if active is None:
            return base_img.overlay(strip_frame)

        direction, transition_type, progress = active

        if transition_type == "fade":
            return self._apply_fade(base_img, strip_frame, progress)
        if transition_type == "slide":
            return self._apply_slide(base_img, strip_frame, progress, direction)
        if transition_type == "wipe":
            return self._apply_wipe(base_img, strip_frame, progress, direction)
        if transition_type == "dissolve":
            return self._apply_dissolve(base_img, strip_frame, progress)
        return base_img.overlay(strip_frame)


# một sequence
class Sequence:
    def __init__(self, strips: List[Strip] = [], n_frame: int = 0, temp_dir="temp"):
        self.n_frame = n_frame
        self.fps = 30  # 30 frame per second
        self.final_frame_cache = []
        self.strips: List[Strip] = strips
        self.temp_dir = temp_dir

        self.strips_dict_by_frame = {}
        self.img_dict_by_frame = {}
        self.init_temp()

    def init_temp(self):
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
            return
        files = glob.glob(f'{self.temp_dir}/*')
        for f in files:
            os.remove(f)

    def add_strip(self, strip: Strip):
        self.strips.append(strip)

    def sort_strips_by_start_frame(self):
        self.strips.sort(key=lambda x: x.start_frame)

    def get_strips_by_frame(self, frame: int):
        list_strips = []
        for strip in self.strips:
            if strip.start_frame <= frame < strip.start_frame + strip.length:
                list_strips.append(strip)
        return list_strips

    def init_strip_dict_by_frame(self):
        for strip in self.strips:
            for frame in range(strip.start_frame, strip.start_frame + strip.length):
                if frame in self.strips_dict_by_frame:
                    self.strips_dict_by_frame[frame].append(strip)
                else:
                    self.strips_dict_by_frame[frame] = [strip]

        for frame in self.strips_dict_by_frame:
            self.strips_dict_by_frame[frame] = self.sort_strips_by_track_id(
                self.strips_dict_by_frame[frame]
            )

    def get_strips_by_frame_dict(self, frame: int):
        if frame in self.strips_dict_by_frame:
            return self.strips_dict_by_frame[frame]
        else:
            list_strips = self.get_strips_by_frame(frame)
            self.strips_dict_by_frame[frame] = list_strips
            return list_strips

    def sort_strips_by_track_id(self, strips: List[Strip]):
        strips.sort(key=lambda x: x.track_id)
        return strips

    def overlay(self, img1, img2):
        return img1.overlay(img2)

    def render_strip_list(self, strips: List[Strip], frame: int):
        img = None
        for strip in strips:
            if strip.type == "text":
                if img is not None:
                    img = strip.apply_text(img)
            elif img is None:
                img = strip.get_frame(frame, self.temp_dir)
            else:
                img = strip.apply_transition_overlay(
                    img, strip.get_frame(frame, self.temp_dir), frame
                )
        return img

    def render_strips(self, strips: List[Strip], frame: int):
        if frame in self.img_dict_by_frame:
            return self.img_dict_by_frame[frame]
        else:
            img = self.render_strip_list(strips, frame)
            self.img_dict_by_frame[frame] = img
            return img

    def get_frame(self, frame: int):
        strips = self.get_strips_by_frame_dict(frame)
        img = self.render_strips(strips, frame)
        return img

    def render_sequence(self):
        self.sort_strips_by_start_frame()
        self.init_strip_dict_by_frame()

        self.final_frame_cache = [None] * self.n_frame
        
        for frame in range(self.n_frame):
            self.final_frame_cache[frame] = self.get_frame(frame)
        return self.final_frame_cache
