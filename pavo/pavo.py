import os
import ffmpeg
from pavo.sequancer.render import render
from pavo.preparation.preparetion import create_asset_tmp, create_asset_tmp_s3


def clear_temp(temp_dir="temp"):
    for f in os.listdir(temp_dir):
        if f.endswith(".jpg"):
            os.remove(f"{temp_dir}/{f}")


def render_video_from_strips(list_strip, output="output.mp4", temp_dir="render_temp"):
    for i, strip in enumerate(list_strip):
        if strip is not None:
            print(f"Processing frame {i}...")
            if os.path.exists(f"{temp_dir}/im-{i}.jpg"):
                os.remove(f"{temp_dir}/im-{i}.jpg")
            # print(
            strip.output(f"{temp_dir}/im-{i}.jpg").run(capture_stdout=True)
    (
        ffmpeg.input(f"{temp_dir}/*.jpg", pattern_type="glob", framerate=25)
        .output(f"{output}")
        .run()
    )

    clear_temp(temp_dir=temp_dir)


def render_video(video_json, output="output.mp4"):
    #  temp_dir="render_temp", video_temp_dir="temp_dir"
    # '/Users/admin/Desktop/pavo-engine-py/render_temp',
    # '/Users/admin/Desktop/pavo-engine-py/temp'

    # create tmp dir
    output_dir = os.path.dirname(output)
    # create video temp dir
    video_temp_dir = os.path.join(output_dir, "video_fps_temp")
    os.makedirs(video_temp_dir, exist_ok=True)

    # create render temp dir
    render_temp_dir = os.path.join(output_dir, "render_temp")
    os.makedirs(render_temp_dir, exist_ok=True)

    list_strip = render(video_json, video_temp_dir)
    render_video_from_strips(list_strip, output, render_temp_dir)
    clear_temp(temp_dir=video_temp_dir)


def json_video_render(json_file, output_file):
    output_dir = os.path.dirname(output_file)
    create_asset_tmp(json_file, output_dir)
    render_video(json_file, output_file)
    return output_file


def json_render_with_s3_asset(
    json_file, output_file, s3_bucket, s3_acess_key, s3_secret_key
):
    output_dir = os.path.dirname(output_file)
    json_new_path = create_asset_tmp_s3(json_file, output_dir, s3_bucket, s3_acess_key, s3_secret_key)
    render_video(json_new_path, output_file)
    return output_file
