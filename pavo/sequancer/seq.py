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
import re
import sys
import time
import glob
import threading


class Effect:
    def __init__(self):
        self.type = None


# một element trong sequence
SUPPORTED_TRANSITIONS = {"fade", "slide", "wipe", "dissolve"}


def _parse_srt(text: str) -> List[Tuple[float, float, str]]:
    """Parse SRT subtitle text into a list of (start_sec, end_sec, content) tuples."""
    pattern = re.compile(
        r"\d+\s*\n"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n"
        r"(.*?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    results = []
    for m in pattern.finditer(text):
        h1, m1, s1, ms1 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        h2, m2, s2, ms2 = int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        content = m.group(9).strip()
        if content:
            results.append((start, end, content))
    return results


def _parse_vtt(text: str) -> List[Tuple[float, float, str]]:
    """Parse WebVTT subtitle text into a list of (start_sec, end_sec, content) tuples."""
    lines = text.splitlines()
    results = []
    i = 0
    # Skip WEBVTT header
    if lines and lines[0].startswith("WEBVTT"):
        i = 1
    while i < len(lines):
        # Find a line containing " --> "
        if "-->" in lines[i]:
            timing = lines[i]
            parts = timing.split("-->")
            def _vtt_time(ts: str) -> float:
                ts = ts.strip().split()[0]  # ignore settings
                ts = ts.replace(",", ".")
                segments = ts.split(":")
                seconds = 0.0
                for seg in segments:
                    seconds = seconds * 60 + float(seg)
                return seconds
            start = _vtt_time(parts[0])
            end = _vtt_time(parts[1])
            i += 1
            content_lines = []
            while i < len(lines) and lines[i].strip():
                content_lines.append(lines[i])
                i += 1
            content = " ".join(content_lines).strip()
            if content:
                results.append((start, end, content))
        else:
            i += 1
    return results


def _load_subtitle_file(src: str) -> List[Tuple[float, float, str]]:
    """Load and parse an SRT or VTT subtitle file."""
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    if src.lower().endswith(".vtt"):
        return _parse_vtt(text)
    return _parse_srt(text)


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
        background_color=None,
        position=None,
        animation=None,
        transition_in=None,
        transition_out=None,
        transition_duration=5,
        trim_start=None,
        trim_end=None,
        # New fields
        loop=False,
        speed=None,
        fit=None,
        filters=None,
        opacity=None,
        volume=None,
        stroke_color=None,
        stroke_width=None,
        shadow=None,
        line_spacing=None,
    ):
        self.type = type
        self.media_source = media_source

        self.track_id = track_id
        self.start_frame = start_frame
        self.length = length
        self.effect = effect

        self.video_start_frame = video_start_frame

        self.video_info = None

        # Text / subtitle overlay attributes
        self.content = content
        self.font = font
        self.size = size
        self.color = color
        self.background_color = background_color
        self.position = position if position is not None else {"x": 0, "y": 0}
        self.animation = animation

        # Transition attributes
        self.transition_in = transition_in
        self.transition_out = transition_out
        self.transition_duration = transition_duration

        # Video-only trim attributes (in seconds; None means no trim on that end)
        self.trim_start = trim_start
        self.trim_end = trim_end

        # New feature attributes
        self.loop = loop or False
        self.speed = speed
        self.fit = fit
        self.filters = filters
        self.opacity = opacity
        self.volume = volume
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.shadow = shadow
        self.line_spacing = line_spacing

        # Subtitle file cache: list of (start_sec, end_sec, content)
        self._subtitle_entries: Optional[List[Tuple[float, float, str]]] = None

        # Thread-local storage for unique temp file naming in parallel renders.
        self._thread_local = threading.local()

    # ------------------------------------------------------------------
    # Subtitle file support
    # ------------------------------------------------------------------

    def load_subtitle_file(self):
        """Parse the SRT/VTT file referenced by *self.media_source* and cache the entries."""
        if self._subtitle_entries is None and self.media_source:
            try:
                self._subtitle_entries = _load_subtitle_file(self.media_source)
            except Exception:
                self._subtitle_entries = []

    def get_subtitle_content(self, frame: int, fps: float = 25.0) -> Optional[str]:
        """Return the subtitle text active at *frame*, or ``None`` if none is active.

        Parameters
        ----------
        frame:
            Absolute frame number being rendered.
        fps:
            Frames per second of the output video.
        """
        if self._subtitle_entries is None:
            return None
        t = (frame - self.start_frame) / fps
        for start, end, text in self._subtitle_entries:
            if start <= t < end:
                return text
        return None

    # ------------------------------------------------------------------
    # Media reading helpers
    # ------------------------------------------------------------------

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

        # For looping video, use the loop input filter.
        if self.loop:
            video_stream = ffmpeg.input(self.media_source, **input_kwargs)
            video_stream = video_stream.filter("loop", loop=-1, size=32767)
        else:
            video_stream = ffmpeg.input(self.media_source, **input_kwargs)

        # Apply speed adjustment via setpts.
        if self.speed and self.speed != 1.0:
            pts_factor = 1.0 / self.speed
            video_stream = video_stream.filter("setpts", f"{pts_factor:.6f}*PTS")

        thread_id = threading.get_ident()
        out, err = (
            video_stream
            .filter('select', 'gte(n,{})'.format(frame))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True)
        )
        path_n = f"{temp_dir}/temp-{frame}-{self.track_id}-{thread_id}.jpg"
        with open(path_n, "wb") as binary_file:
            binary_file.write(out)

        out = ffmpeg.input(path_n)
        return out

    def init_strip(self):
        pass

    # ------------------------------------------------------------------
    # Effect helpers
    # ------------------------------------------------------------------

    def _get_progress(self, frame: int) -> float:
        """Return the strip's animation progress in ``[0.0, 1.0]`` at *frame*."""
        if self.length <= 1:
            return 1.0
        return max(0.0, min(1.0, (frame - self.start_frame) / (self.length - 1)))

    def _apply_media_effect(self, img, frame: int):
        """Apply Ken-Burns-style motion effects to an image or video stream.

        Supports: ``zoomIn``, ``zoomOut``, ``panLeft``, ``panRight``,
        ``slideUp``, ``slideDown``.
        """
        if not self.effect:
            return img

        progress = self._get_progress(frame)

        if self.effect == "zoomIn":
            sf = 1.0 + 0.3 * progress
            img = img.filter("scale", f"iw*{sf:.4f}", f"ih*{sf:.4f}")
            img = img.filter(
                "crop",
                f"iw/{sf:.4f}", f"ih/{sf:.4f}",
                f"(iw-iw/{sf:.4f})/2", f"(ih-ih/{sf:.4f})/2",
            )

        elif self.effect == "zoomOut":
            sf = 1.3 - 0.3 * progress  # 1.3 → 1.0
            img = img.filter("scale", f"iw*{sf:.4f}", f"ih*{sf:.4f}")
            img = img.filter(
                "crop",
                f"iw/{sf:.4f}", f"ih/{sf:.4f}",
                f"(iw-iw/{sf:.4f})/2", f"(ih-ih/{sf:.4f})/2",
            )

        elif self.effect == "panLeft":
            # Scale image wider, then crop moving from right to left.
            scale = 1.3
            # x goes from iw*(1-1/scale) → 0 as progress 0 → 1
            x_frac = (1.0 - 1.0 / scale) * (1.0 - progress)
            img = img.filter("scale", f"iw*{scale:.4f}", "ih")
            img = img.filter(
                "crop", f"iw/{scale:.4f}", "ih",
                f"iw*{x_frac:.4f}", "0",
            )

        elif self.effect == "panRight":
            # Scale image wider, then crop moving from left to right.
            scale = 1.3
            x_frac = (1.0 - 1.0 / scale) * progress
            img = img.filter("scale", f"iw*{scale:.4f}", "ih")
            img = img.filter(
                "crop", f"iw/{scale:.4f}", "ih",
                f"iw*{x_frac:.4f}", "0",
            )

        elif self.effect == "slideUp":
            # Scale image taller, then crop moving upward.
            scale = 1.3
            y_frac = (1.0 - 1.0 / scale) * (1.0 - progress)
            img = img.filter("scale", "iw", f"ih*{scale:.4f}")
            img = img.filter(
                "crop", "iw", f"ih/{scale:.4f}",
                "0", f"ih*{y_frac:.4f}",
            )

        elif self.effect == "slideDown":
            # Scale image taller, then crop moving downward.
            scale = 1.3
            y_frac = (1.0 - 1.0 / scale) * progress
            img = img.filter("scale", "iw", f"ih*{scale:.4f}")
            img = img.filter(
                "crop", "iw", f"ih/{scale:.4f}",
                "0", f"ih*{y_frac:.4f}",
            )

        return img

    # ------------------------------------------------------------------
    # Fit / scale helpers
    # ------------------------------------------------------------------

    def _apply_fit(self, img, output_width: Optional[int], output_height: Optional[int]):
        """Scale the stream according to *self.fit* and the output dimensions."""
        if not self.fit or self.fit == "none":
            return img
        if not output_width or not output_height:
            return img

        w, h = output_width, output_height

        if self.fit == "stretch":
            img = img.filter("scale", str(w), str(h))

        elif self.fit == "cover":
            # Scale so that both dimensions are ≥ output size, then crop to exact size.
            img = img.filter(
                "scale",
                f"if(gt(iw/ih,{w}/{h}),{w}*ih/oh,-1)".replace("oh", str(h)),
                str(h),
            )
            img = img.filter(
                "scale",
                f"if(gte(iw,{w}),iw,{w}*ih/oh)".replace("oh", str(h)),
                "-1",
            )
            # Simpler two-step: scale to fill the smallest dimension, then crop.
            img = img.filter("scale", f"-1:{h}")
            img = img.filter("scale", f"{w}:-1")
            img = img.filter("crop", str(w), str(h))

        elif self.fit == "contain":
            # Scale to fit within output, preserving aspect ratio (pillarbox / letterbox).
            img = img.filter(
                "scale",
                str(w), str(h),
                force_original_aspect_ratio="decrease",
            )
            img = img.filter(
                "pad", str(w), str(h),
                "(ow-iw)/2", "(oh-ih)/2",
            )

        return img

    # ------------------------------------------------------------------
    # Color grading / filter helpers
    # ------------------------------------------------------------------

    def _apply_filters(self, img):
        """Apply color grading and visual filters from *self.filters*."""
        if not self.filters:
            return img

        # brightness / contrast / saturation via the eq filter.
        eq_params = {}
        if self.filters.get("brightness") is not None:
            eq_params["brightness"] = self.filters["brightness"]
        if self.filters.get("contrast") is not None:
            eq_params["contrast"] = self.filters["contrast"]
        if self.filters.get("saturation") is not None and not self.filters.get("grayscale"):
            eq_params["saturation"] = self.filters["saturation"]
        if eq_params:
            img = img.filter("eq", **eq_params)

        # Grayscale (desaturate).
        if self.filters.get("grayscale"):
            img = img.filter("hue", s=0)
        elif self.filters.get("sepia"):
            # Sepia tone via colorchannelmixer.
            img = img.filter(
                "colorchannelmixer",
                rr=0.393, rg=0.769, rb=0.189,
                gr=0.349, gg=0.686, gb=0.168,
                br=0.272, bg=0.534, bb=0.131,
            )

        # Gaussian blur.
        if self.filters.get("blur"):
            img = img.filter("gblur", sigma=self.filters["blur"])

        # .cube LUT file.
        if self.filters.get("lut"):
            img = img.filter("lut3d", file=self.filters["lut"])

        return img

    # ------------------------------------------------------------------
    # Opacity helper
    # ------------------------------------------------------------------

    def _apply_opacity(self, img):
        """Apply *self.opacity* to the stream as an alpha channel multiplier."""
        if self.opacity is None or self.opacity >= 1.0:
            return img
        return (
            img
            .filter("format", "rgba")
            .filter("colorchannelmixer", aa=max(0.0, min(1.0, self.opacity)))
        )

    # ------------------------------------------------------------------
    # Frame getter
    # ------------------------------------------------------------------

    def get_frame(
        self,
        frame: int,
        temp_dir: str = "./temp",
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
    ):
        if self.type == "image":
            img = self.read_image()
            img = self._apply_fit(img, output_width, output_height)
            img = self._apply_media_effect(img, frame)
            img = self._apply_filters(img)
            img = self._apply_opacity(img)
            return img
        elif self.type == "video":
            vid = self.read_video_by_frame(frame, temp_dir)
            vid = self._apply_fit(vid, output_width, output_height)
            vid = self._apply_filters(vid)
            return vid
        elif self.type == "watermark":
            img = self.read_image()
            img = self._apply_fit(img, output_width, output_height)
            img = self._apply_filters(img)
            img = self._apply_opacity(img)
            return img
        elif self.type in ("text", "subtitle", "audio"):
            return None
        else:
            return None

    # ------------------------------------------------------------------
    # Text overlay
    # ------------------------------------------------------------------

    def apply_text(
        self,
        img,
        frame: int = 0,
        fps: float = 25.0,
    ):
        """Apply this text/subtitle strip's content to an existing FFmpeg stream via drawtext.

        Parameters
        ----------
        img : ffmpeg stream
            The base FFmpeg stream to draw text onto.
        frame : int
            The absolute frame number being rendered (used for animations and
            per-frame subtitle lookup).
        fps : float
            Frames per second of the output video (used for time calculations).

        Returns
        -------
        ffmpeg stream
            The stream with the ``drawtext`` filter applied.

        Notes
        -----
        For ``type == "subtitle"`` strips with a *src* (SRT/VTT file), the
        active subtitle line at *frame* is looked up automatically.

        Supported animations (``asset.animation``):

        * ``typewriter`` – characters are revealed progressively over the strip.
        * ``fadeIn`` – text fades in from transparent over the strip duration.
        * ``fadeOut`` – text fades out to transparent over the strip duration.
        * ``slideInLeft`` – text slides in from off the left edge.
        """
        # Determine the text to display.
        display_content = self.content

        # SRT/VTT subtitle: look up active line for this frame.
        if self.type == "subtitle" and self.media_source:
            self.load_subtitle_file()
            display_content = self.get_subtitle_content(frame, fps)
            if not display_content:
                return img  # No active subtitle at this frame.

        if not display_content:
            return img

        pos = self.position if isinstance(self.position, dict) else {}
        raw_x = pos.get("x", 0)
        raw_y = pos.get("y", 0)

        # Support named positions ("center") as FFmpeg expressions.
        x_expr = "(w-tw)/2" if str(raw_x).lower() == "center" else str(raw_x)
        y_expr = "(h-th)/2" if str(raw_y).lower() == "center" else str(raw_y)

        # Animation modifiers.
        progress = self._get_progress(frame)
        alpha = 1.0

        if self.animation == "typewriter":
            # Reveal characters progressively.
            n_chars = max(1, round(len(display_content) * progress))
            display_content = display_content[:n_chars]

        elif self.animation == "fadeIn":
            alpha = progress

        elif self.animation == "fadeOut":
            alpha = 1.0 - progress

        elif self.animation == "slideInLeft":
            # Text slides in from the left edge. At progress=0: fully off-screen left.
            if progress < 1.0:
                slide_offset = -(1.0 - progress)
                x_expr = f"(w-tw)/2 + (w * {slide_offset:.4f})"

        kwargs = {
            "text": display_content,
            "fontsize": self.size,
            "fontcolor": self.color,
            "x": x_expr,
            "y": y_expr,
        }

        if alpha < 1.0:
            kwargs["alpha"] = max(0.0, min(1.0, alpha))

        if self.font:
            kwargs["fontfile"] = self.font

        # Subtitle type: render a background box when background_color is set.
        if self.type == "subtitle" and self.background_color:
            kwargs["box"] = 1
            kwargs["boxcolor"] = self.background_color

        # Text stroke (outline).
        if self.stroke_color:
            kwargs["bordercolor"] = self.stroke_color
            kwargs["borderw"] = self.stroke_width if self.stroke_width else 1

        # Drop shadow.
        if self.shadow:
            shadow_cfg = self.shadow if isinstance(self.shadow, dict) else {}
            kwargs["shadowx"] = shadow_cfg.get("x", 2)
            kwargs["shadowy"] = shadow_cfg.get("y", 2)
            kwargs["shadowcolor"] = shadow_cfg.get("color", "black")

        # Line spacing.
        if self.line_spacing is not None:
            kwargs["line_spacing"] = self.line_spacing

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
    def __init__(
        self,
        strips: List[Strip] = [],
        n_frame: int = 0,
        temp_dir: str = "temp",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: float = 25.0,
    ):
        self.n_frame = n_frame
        self.fps = fps
        self.width = width
        self.height = height
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
            # Audio strips are not rendered as video frames.
            if strip.type == "audio":
                continue
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
            if strip.type in ("text", "subtitle"):
                if img is not None:
                    img = strip.apply_text(img, frame=frame, fps=self.fps)
            elif strip.type == "audio":
                continue  # Audio strips are handled separately.
            elif img is None:
                img = strip.get_frame(
                    frame, self.temp_dir,
                    output_width=self.width,
                    output_height=self.height,
                )
            else:
                img = strip.apply_transition_overlay(
                    img,
                    strip.get_frame(
                        frame, self.temp_dir,
                        output_width=self.width,
                        output_height=self.height,
                    ),
                    frame,
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

    def render_sequence(self, workers: int = 1):
        """Render all frames, optionally in parallel.

        Parameters
        ----------
        workers:
            Number of parallel worker threads.  Values > 1 enable concurrent
            frame rendering for faster throughput on multi-core machines.
        """
        self.sort_strips_by_start_frame()
        self.init_strip_dict_by_frame()

        self.final_frame_cache = [None] * self.n_frame

        if workers > 1 and self.n_frame > 0:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_frame = {
                    executor.submit(self.get_frame, f): f
                    for f in range(self.n_frame)
                }
                for future in as_completed(future_to_frame):
                    f = future_to_frame[future]
                    self.final_frame_cache[f] = future.result()
        else:
            for frame in range(self.n_frame):
                self.final_frame_cache[frame] = self.get_frame(frame)

        return self.final_frame_cache
