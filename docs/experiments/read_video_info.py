import ffmpeg
import os


def read_video_info(in_filename):
    probe = ffmpeg.probe(in_filename)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )



    
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    print(video_stream)

read_video_info("data/in.mp4")