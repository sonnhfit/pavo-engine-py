import pytest

from sequancer.render import (
    render,
    read_json_video,
    get_strips_from_json,
    init_sequence,
)


def test_read_json_video():
    json_data = read_json_video("docs/data.json")
    assert json_data["timeline"]["n_frames"] == 30
    assert len(json_data["timeline"]["tracks"]) == 2
    assert len(json_data["timeline"]["tracks"][0]["strips"]) == 3
    assert len(json_data["timeline"]["tracks"][1]["strips"]) == 2


def test_get_strips_from_json():
    json_data = read_json_video("docs/data.json")
    strips = get_strips_from_json(json_data)

    assert len(strips) == 5
    assert strips[0].start_frame == 0
    assert strips[0].length == 5
    assert strips[0].track_id == 0
    # assert strips[0].media_source == "docs/1.mp4"
    assert strips[0].type == "image"
    # assert strips[0].video_start_frame == 0
    assert strips[1].start_frame == 3
    # assert strips[1].length == 10
    assert strips[1].track_id == 0


def test_init_sequence():
    seq = init_sequence("docs/data.json")
    assert seq.n_frame == 30
    assert len(seq.strips) == 5
