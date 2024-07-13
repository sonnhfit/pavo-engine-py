import json
from pavo.sequancer.seq import Sequence, Strip


def read_json_video(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def get_strips_from_json(json_data):
    strips = []
    for track in json_data["timeline"]["tracks"]:
        for item in track["strips"]:
            strip = Strip(
                type=item["asset"]["type"],
                media_source=item["asset"]["src"],
                track_id=track["track_id"],
                start_frame=item["start"],
                length=item["length"],
                effect=item["effect"],
                video_start_frame=item["video_start_frame"],
            )
            strips.append(strip)

    return strips


def init_sequence(file_path, temp_dir):
    json_data = read_json_video(file_path)
    strips = get_strips_from_json(json_data)
    seq = Sequence(
        strips=strips, n_frame=json_data["timeline"]["n_frames"], temp_dir=temp_dir
    )
    return seq


def render(input_file_path, temp_dir="temp"):
    seq = init_sequence(input_file_path, temp_dir)
    return seq.render_sequence()
