#!/usr/bin/env python
from __future__ import unicode_literals
import ffmpeg
import sys


def read_frame_as_jpeg(in_filename, frame_num):
    out, err = (
        ffmpeg
        .input(in_filename)
        .filter('select', 'gte(n,{})'.format(frame_num))
        .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
        .run(capture_stdout=True)
    )
    return out


if __name__ == '__main__':
    in_filename = "data/in1.mp4"
    out_filename = "data/out_frame_123.jpg"
    frame_num = 30

    print(ffmpeg.input(out_filename).run(capture_stdout=True))

    # out = read_frame_as_jpeg(in_filename, frame_num)
    # out = ffmpeg.input(out)
    # print(type(out))
    # out.output('out22__ta.jpg').run(capture_stdout=True)


    # with open(out_filename, "wb") as binary_file:
    #     binary_file.write(out)
