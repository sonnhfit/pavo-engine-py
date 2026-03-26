import json
from pavo.sequancer.seq import Sequence, Strip


def read_json_video(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def get_strips_from_json(json_data):
    fps = float((json_data.get("output") or {}).get("fps", 25.0))
    strips = []
    for track in json_data["timeline"]["tracks"]:
        for item in track["strips"]:
            asset = item["asset"]
            asset_type = asset.get("type")

            transition = item.get("transition") or {}
            try:
                transition_duration = int(transition.get("duration", 5))
            except (TypeError, ValueError):
                transition_duration = 5
            common_kwargs = dict(
                type=asset_type,
                track_id=track["track_id"],
                start_frame=item["start"],
                length=item["length"],
                effect=item.get("effect"),
                video_start_frame=item.get("video_start_frame", 0),
                transition_in=transition.get("in"),
                transition_out=transition.get("out"),
                transition_duration=transition_duration,
            )

            if asset_type in ("text", "subtitle"):
                strip = Strip(
                    **common_kwargs,
                    media_source=None,
                    content=asset.get("content"),
                    font=asset.get("font"),
                    size=asset.get("size", 24),
                    color=asset.get("color", "white"),
                    background_color=asset.get("background_color"),
                    position=asset.get("position", {"x": 0, "y": 0}),
                    animation=asset.get("animation"),
                )
            else:
                # Resolve trim parameters: convert frame-based values to seconds.
                trim_start = asset.get("trim_start")
                trim_end = asset.get("trim_end")
                trim_start_frame = asset.get("trim_start_frame")
                trim_end_frame = asset.get("trim_end_frame")
                if trim_start is None and trim_start_frame is not None:
                    trim_start = trim_start_frame / fps
                if trim_end is None and trim_end_frame is not None:
                    trim_end = trim_end_frame / fps
                strip = Strip(
                    **common_kwargs,
                    media_source=asset.get("src"),
                    trim_start=trim_start,
                    trim_end=trim_end,
                )
            strips.append(strip)

    return strips


def init_sequence(file_path, temp_dir="temp"):
    json_data = read_json_video(file_path)
    strips = get_strips_from_json(json_data)
    seq = Sequence(
        strips=strips, n_frame=json_data["timeline"]["n_frames"], temp_dir=temp_dir
    )
    return seq


def render(input_file_path, temp_dir="temp"):
    seq = init_sequence(input_file_path, temp_dir)
    return seq.render_sequence()
