import json
import os
import shutil
import tempfile
from typing import Callable, List, Optional, Tuple

import ffmpeg
from tqdm import tqdm

from pavo.schema import validate_timeline_json
from pavo.sequancer.render import render, get_audio_strips_from_json


def clear_temp(temp_dir="temp"):
    for f in os.listdir(temp_dir):
        if f.endswith(".jpg"):
            os.remove(f"{temp_dir}/{f}")


def _create_background_frame(color, width, height, output_path):
    """Generate a solid-color JPEG frame using FFmpeg."""
    hex_color = color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    (
        ffmpeg
        .input(f"color=c={r}:{g}:{b}:size={width}x{height}:rate=1", f="lavfi", t=1)
        .output(output_path, vframes=1)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def render_video_from_strips(
    list_strip,
    output="output.mp4",
    temp_dir="render_temp",
    fps=25,
    background="#000000",
    width=None,
    height=None,
    on_progress: Optional[Callable[[float], None]] = None,
):
    """Render a list of FFmpeg strip objects to an MP4 video file.

    Parameters
    ----------
    list_strip:
        List of FFmpeg filter-chain objects (one per frame).
    output:
        Destination path for the rendered video.
    temp_dir:
        Directory for intermediate JPEG frames.
    fps:
        Frames per second of the output video.
    background:
        Hex color string for empty/background frames.
    width, height:
        Output frame dimensions.
    on_progress:
        Optional callback invoked with a float in ``[0.0, 1.0]`` after each
        frame is written, allowing callers to track rendering progress.
    """
    bg_path = os.path.join(temp_dir, "bg.jpg")
    bg_created = False
    total = len(list_strip)

    for i, strip in enumerate(tqdm(list_strip, desc="Rendering frames")):
        frame_path = f"{temp_dir}/im-{i:06d}.jpg"
        if os.path.exists(frame_path):
            os.remove(frame_path)

        if strip is not None:
            strip.output(frame_path).run(
                capture_stdout=True, capture_stderr=True
            )
        else:
            if not bg_created and width and height:
                _create_background_frame(background, width, height, bg_path)
                bg_created = True
            if bg_created:
                shutil.copy(bg_path, frame_path)
            else:
                ffmpeg.input(
                    f"color=black:size=640x480:rate=1", f="lavfi", t=1
                ).output(frame_path, vframes=1).overwrite_output().run(
                    capture_stdout=True, capture_stderr=True
                )

        if on_progress is not None and total > 0:
            on_progress((i + 1) / total)

    (
        ffmpeg.input(
            f"{temp_dir}/im-*.jpg", pattern_type="glob", framerate=fps
        )
        .output(output)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    clear_temp(temp_dir=temp_dir)


def _add_audio_to_video(video_path, audio_path, output_path):
    """Merge audio track into a video file, trimming audio to video length."""
    video_in = ffmpeg.input(video_path)
    audio_in = ffmpeg.input(audio_path)
    (
        ffmpeg
        .output(
            video_in.video,
            audio_in.audio,
            output_path,
            vcodec="copy",
            acodec="aac",
            shortest=None,
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def _build_ducking_expr(speech_segments, reduction_db=10.0):
    """Build an FFmpeg ``volume`` filter expression that ducks audio during speech.

    Parameters
    ----------
    speech_segments : list of (float, float)
        Each tuple is *(start_seconds, end_seconds)* of a detected speech segment.
    reduction_db : float
        How many decibels to reduce the volume during speech (positive number).

    Returns
    -------
    str
        An FFmpeg volume expression string suitable for the ``volume`` filter.
    """
    if not speech_segments:
        return "1.0"

    linear_factor = 10 ** (-abs(reduction_db) / 20.0)
    conditions = [f"between(t,{start:.3f},{end:.3f})" for start, end in speech_segments]
    combined = "+".join(conditions)
    return f"if({combined},{linear_factor:.6f},1.0)"


def _apply_audio_ducking(audio_path, speech_segments, reduction_db, output_path):
    """Create a ducked copy of *audio_path* with reduced volume during speech segments.

    Parameters
    ----------
    audio_path : str
        Path to the original soundtrack file.
    speech_segments : list of (float, float)
        Speech time intervals in seconds (output-video timeline).
    reduction_db : float
        Volume reduction in dB during speech (positive number means reduction).
    output_path : str
        Destination path for the processed audio file.
    """
    if not speech_segments:
        shutil.copy(audio_path, output_path)
        return

    expr = _build_ducking_expr(speech_segments, reduction_db)
    (
        ffmpeg
        .input(audio_path)
        .filter("volume", volume=expr, eval="frame")
        .output(output_path)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def _apply_soundtrack_effect(audio_path: str, effect: str, output_path: str, fps: float = 25.0):
    """Apply a fade effect to a soundtrack using FFmpeg's ``afade`` filter.

    Parameters
    ----------
    audio_path:
        Path to the input audio file.
    effect:
        One of ``'fadeIn'``, ``'fadeOut'``, or ``'fadeInOut'``.
    output_path:
        Destination path for the processed audio file.
    fps:
        Frames per second (used to determine default fade duration).
    """
    if effect not in ("fadeIn", "fadeOut", "fadeInOut"):
        shutil.copy(audio_path, output_path)
        return

    try:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe.get("format", {}).get("duration", 0))
    except Exception:
        duration = 0.0

    fade_dur = min(1.0, duration * 0.1) if duration > 0 else 1.0

    stream = ffmpeg.input(audio_path)
    audio = stream.audio

    if effect in ("fadeIn", "fadeInOut"):
        audio = audio.filter("afade", type="in", start_time=0, duration=fade_dur)
    if effect in ("fadeOut", "fadeInOut") and duration > fade_dur:
        audio = audio.filter(
            "afade", type="out",
            start_time=max(0.0, duration - fade_dur),
            duration=fade_dur,
        )

    (
        ffmpeg
        .output(audio, output_path)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def _mix_audio_strips(
    video_path: str,
    audio_strips,
    output_path: str,
    fps: float,
    existing_audio_path: Optional[str] = None,
):
    """Mix per-strip audio clips into the video, optionally alongside a soundtrack.

    Each audio strip is delayed to its ``start_frame / fps`` position and
    mixed using FFmpeg's ``amix`` filter.

    Parameters
    ----------
    video_path:
        Path to the silent (or soundtracked) video file.
    audio_strips:
        Iterable of :class:`~pavo.sequancer.seq.Strip` objects with
        ``type == 'audio'``.
    output_path:
        Destination path for the output video with mixed audio.
    fps:
        Frames per second (used to convert frame offsets to seconds).
    existing_audio_path:
        Optional path to an already-mixed audio track (e.g. the soundtrack).
        When provided, it is mixed together with the strip audio clips.
    """
    inputs = []
    filter_parts = []
    n_inputs = 0

    # Include pre-existing audio (soundtrack) as the first input if present.
    if existing_audio_path:
        inputs.append(ffmpeg.input(existing_audio_path).audio)
        filter_parts.append(f"[{n_inputs}:a]")
        n_inputs += 1

    for strip in audio_strips:
        if not strip.media_source or not os.path.exists(strip.media_source):
            continue
        delay_ms = int((strip.start_frame / fps) * 1000)
        audio = ffmpeg.input(strip.media_source).audio
        # Apply volume if specified.
        if strip.volume is not None:
            audio = audio.filter("volume", volume=strip.volume)
        # Delay the clip using the adelay filter.
        audio = audio.filter("adelay", f"{delay_ms}|{delay_ms}")
        inputs.append(audio)
        filter_parts.append(f"[{n_inputs}:a]")
        n_inputs += 1

    if not inputs:
        # Nothing to mix; just copy.
        shutil.copy(video_path, output_path)
        return

    video_in = ffmpeg.input(video_path)

    if n_inputs == 1:
        # Only one audio source; no mixing needed.
        mixed_audio = inputs[0]
    else:
        mixed_audio = ffmpeg.filter(inputs, "amix", inputs=n_inputs, duration="longest")

    (
        ffmpeg
        .output(
            video_in.video,
            mixed_audio,
            output_path,
            vcodec="copy",
            acodec="aac",
            shortest=None,
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def _detect_speech_segments(timeline, fps):
    """Detect speech intervals from video strips in the timeline.

    Iterates over all *video* strips that have an audio stream, transcribes
    each one using :class:`pavo.perception.speech.SpeechTranscriber`, and maps
    the resulting segment timestamps to the output-video timeline.

    Requires ``openai-whisper`` to be installed.  When the package is absent,
    or when no suitable video strip is found, an empty list is returned.

    Parameters
    ----------
    timeline : dict
        The parsed ``timeline`` object from the JSON specification.
    fps : float
        Frames-per-second of the output video (used to convert frame numbers
        to seconds).

    Returns
    -------
    list of (float, float)
        Speech segments as *(start_seconds, end_seconds)* pairs in the
        coordinate space of the final output video.
    """
    try:
        from pavo.perception.speech.transcriber import SpeechTranscriber
    except ImportError:
        return []

    speech_segments = []

    for track in timeline.get("tracks", []):
        for strip_data in track.get("strips", []):
            asset = strip_data.get("asset", {})
            if asset.get("type") != "video":
                continue

            src = asset.get("src", "")
            if not src or not os.path.exists(src):
                continue

            # Skip video files with no audio stream.
            try:
                probe = ffmpeg.probe(src)
                has_audio = any(
                    s["codec_type"] == "audio" for s in probe.get("streams", [])
                )
                if not has_audio:
                    continue
            except Exception:
                continue

            start_frame = strip_data.get("start", 0)
            video_start_frame = strip_data.get("video_start_frame", 0)
            length = strip_data.get("length", 0)

            strip_start_sec = start_frame / fps
            video_start_sec = video_start_frame / fps
            strip_duration_sec = length / fps

            try:
                transcriber = SpeechTranscriber()
                result = transcriber.transcribe(src)

                for seg in result.get("segments", []):
                    seg_start = seg["start"]
                    seg_end = seg["end"]

                    # Clamp to the portion of the source video used by this strip.
                    clip_start = max(seg_start, video_start_sec)
                    clip_end = min(seg_end, video_start_sec + strip_duration_sec)

                    if clip_start < clip_end:
                        out_start = strip_start_sec + (clip_start - video_start_sec)
                        out_end = strip_start_sec + (clip_end - video_start_sec)
                        speech_segments.append((out_start, out_end))
            except Exception:
                continue

    return speech_segments


def _render_gif(frames_dir: str, output: str, fps: float, width: Optional[int] = None):
    """Render JPEG frames in *frames_dir* to an animated GIF using FFmpeg.

    Uses a two-pass approach (palettegen + paletteuse) for high-quality output.

    Parameters
    ----------
    frames_dir:
        Directory containing numbered ``im-NNNNNN.jpg`` frame files.
    output:
        Destination ``.gif`` path.
    fps:
        Frames per second of the animation.
    width:
        Optional output width in pixels (height scaled proportionally).
        Defaults to 480 px when neither *width* nor source dimensions are known.
    """
    scale_w = str(width) if width else "480"
    scale_filter = f"fps={fps},scale={scale_w}:-1:flags=lanczos"

    palette_path = os.path.join(frames_dir, "palette.png")

    # Pass 1: Generate color palette.
    (
        ffmpeg
        .input(f"{frames_dir}/im-*.jpg", pattern_type="glob", framerate=fps)
        .filter_multi_output("split")[0]
        .filter("palettegen")
        .output(palette_path)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Pass 2: Encode GIF using the palette.
    frames_in = ffmpeg.input(
        f"{frames_dir}/im-*.jpg", pattern_type="glob", framerate=fps
    )
    palette_in = ffmpeg.input(palette_path)
    (
        ffmpeg
        .filter([frames_in, palette_in], "paletteuse")
        .output(output)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def render_video(
    json_path,
    output="output.mp4",
    on_progress: Optional[Callable[[float], None]] = None,
):
    """Render a JSON timeline specification to a video or GIF file.

    Parameters
    ----------
    json_path : str
        Path to the JSON timeline file.
    output : str
        Path for the output file.  The container format is determined by the
        ``output.format`` field in the JSON (default ``'mp4'``).
    on_progress : callable, optional
        A callback invoked with a ``float`` in ``[0.0, 1.0]`` after each
        rendered frame.  Useful for progress bars in web services and APIs.

    Raises
    ------
    FileNotFoundError
        If *json_path* does not exist.
    ValueError
        If the JSON content is invalid or missing required fields.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path) as fh:
            video_json = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    validate_timeline_json(video_json)

    timeline = video_json.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("JSON must contain a 'timeline' object")

    output_spec = video_json.get("output", {})
    fps = output_spec.get("fps", 25)
    width = output_spec.get("width")
    height = output_spec.get("height")
    out_format = (output_spec.get("format") or "mp4").lower()
    background = timeline.get("background", "#000000")
    soundtrack = timeline.get("soundtrack", {})
    soundtrack_src = soundtrack.get("src") if isinstance(soundtrack, dict) else None
    soundtrack_effect = soundtrack.get("effect") if isinstance(soundtrack, dict) else None
    audio_ducking = output_spec.get("audio_ducking", False)
    ducking_reduction_db = output_spec.get("ducking_reduction_db", 10.0)
    workers = int(output_spec.get("workers") or 1)

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)

    tmp_root = tempfile.mkdtemp(prefix="pavo_", dir=output_dir)
    video_temp_dir = os.path.join(tmp_root, "video_fps_temp")
    render_temp_dir = os.path.join(tmp_root, "render_temp")
    os.makedirs(video_temp_dir, exist_ok=True)
    os.makedirs(render_temp_dir, exist_ok=True)

    try:
        print(f"[pavo] Rendering timeline: {json_path}")
        list_strip = render(json_path, video_temp_dir, workers=workers)

        # Collect audio strips for later mixing.
        audio_strips = get_audio_strips_from_json(video_json)
        has_audio_strips = bool(audio_strips)

        # Determine paths for intermediate files.
        is_gif = out_format == "gif"
        has_soundtrack = bool(soundtrack_src)
        needs_audio_stage = has_soundtrack or has_audio_strips

        # Stage 1: render video frames to a silent video file.
        if is_gif:
            # For GIF output we don't need an intermediate video; frames are used directly.
            silent_video = os.path.join(tmp_root, "silent.mp4")
        else:
            silent_video = output if not needs_audio_stage else os.path.join(tmp_root, "silent.mp4")

        render_video_from_strips(
            list_strip,
            output=silent_video,
            temp_dir=render_temp_dir,
            fps=fps,
            background=background,
            width=width,
            height=height,
            on_progress=on_progress,
        )

        if is_gif:
            print(f"[pavo] Encoding GIF: {output}")
            _render_gif(render_temp_dir, output, fps=fps, width=width)
            return

        if not needs_audio_stage:
            print(f"[pavo] Done → {output}")
            return

        # Stage 2: process soundtrack (effects + ducking).
        effective_soundtrack = soundtrack_src
        if has_soundtrack:
            # Apply soundtrack fade effects.
            if soundtrack_effect:
                print(f"[pavo] Applying soundtrack effect: {soundtrack_effect}")
                effected_audio = os.path.join(tmp_root, "effected_audio.aac")
                _apply_soundtrack_effect(soundtrack_src, soundtrack_effect, effected_audio, fps)
                effective_soundtrack = effected_audio

            # Apply audio ducking if requested.
            if audio_ducking:
                print("[pavo] Detecting speech segments for audio ducking...")
                speech_segs = _detect_speech_segments(timeline, fps)
                if speech_segs:
                    ducked_audio = os.path.join(tmp_root, "ducked_audio.aac")
                    print(
                        f"[pavo] Applying audio ducking "
                        f"({len(speech_segs)} speech segment(s), "
                        f"-{ducking_reduction_db} dB)..."
                    )
                    _apply_audio_ducking(
                        effective_soundtrack, speech_segs, ducking_reduction_db, ducked_audio
                    )
                    effective_soundtrack = ducked_audio
                else:
                    print("[pavo] No speech detected; skipping audio ducking.")

        # Stage 3: mix audio strips + soundtrack into the output video.
        if has_audio_strips:
            print(f"[pavo] Mixing {len(audio_strips)} audio strip(s) into video...")
            _mix_audio_strips(
                silent_video,
                audio_strips,
                output,
                fps=fps,
                existing_audio_path=effective_soundtrack,
            )
        elif has_soundtrack:
            print(f"[pavo] Adding soundtrack: {effective_soundtrack}")
            _add_audio_to_video(silent_video, effective_soundtrack, output)

        print(f"[pavo] Done → {output}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def json_video_render(json_file, output_file):
    from pavo.preparation.preparetion import create_asset_tmp
    output_dir = os.path.dirname(os.path.abspath(output_file))
    create_asset_tmp(json_file, output_dir)
    render_video(json_file, output_file)
    return output_file


def json_render_with_s3_asset(
    json_file, output_file, s3_bucket, s3_acess_key, s3_secret_key
):
    from pavo.preparation.preparetion import create_asset_tmp_s3
    output_dir = os.path.dirname(os.path.abspath(output_file))
    json_new_path = create_asset_tmp_s3(
        json_file, output_dir, s3_bucket, s3_acess_key, s3_secret_key
    )
    render_video(json_new_path, output_file)
    return output_file
