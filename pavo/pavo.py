import json
import os
import shutil
import tempfile

import ffmpeg
from tqdm import tqdm

from pavo.sequancer.render import render


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
):
    """Render a list of FFmpeg strip objects to an MP4 video file."""
    bg_path = os.path.join(temp_dir, "bg.jpg")
    bg_created = False

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


def render_video(json_path, output="output.mp4"):
    """Render a JSON timeline specification to an MP4 video file.

    Parameters
    ----------
    json_path : str
        Path to the JSON timeline file.
    output : str
        Path for the output MP4 file.

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

    timeline = video_json.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("JSON must contain a 'timeline' object")

    output_spec = video_json.get("output", {})
    fps = output_spec.get("fps", 25)
    width = output_spec.get("width")
    height = output_spec.get("height")
    background = timeline.get("background", "#000000")
    soundtrack = timeline.get("soundtrack", {})
    soundtrack_src = soundtrack.get("src") if isinstance(soundtrack, dict) else None

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)

    tmp_root = tempfile.mkdtemp(prefix="pavo_", dir=output_dir)
    video_temp_dir = os.path.join(tmp_root, "video_fps_temp")
    render_temp_dir = os.path.join(tmp_root, "render_temp")
    os.makedirs(video_temp_dir, exist_ok=True)
    os.makedirs(render_temp_dir, exist_ok=True)

    try:
        print(f"[pavo] Rendering timeline: {json_path}")
        list_strip = render(json_path, video_temp_dir)

        video_only = output
        has_audio = bool(soundtrack_src)
        if has_audio:
            video_only = os.path.join(tmp_root, "video_only.mp4")

        render_video_from_strips(
            list_strip,
            output=video_only,
            temp_dir=render_temp_dir,
            fps=fps,
            background=background,
            width=width,
            height=height,
        )

        if has_audio:
            print(f"[pavo] Adding soundtrack: {soundtrack_src}")
            _add_audio_to_video(video_only, soundtrack_src, output)

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
