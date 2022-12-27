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
import numpy as np
import cv2 as cv
import ffmpeg
import os
import sys
import time
import glob

class Effect:
    def __init__(self):
        self.type = None


# một element trong sequence
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
    ):
        self.type = type
        self.media_source = media_source

        self.track_id = track_id
        self.start_frame = start_frame
        self.length = length
        self.effect = effect

        self.video_start_frame = video_start_frame

        self.video_info = None

    def load_media_source(self):
        pass

    def read_image(self):
        img = ffmpeg.input(self.media_source)
        return img

    def read_video(self):
        pass

    def read_video_by_frame(self, frame: int):

        out, err = (
            ffmpeg
            .input(self.media_source)
            .filter('select', 'gte(n,{})'.format(frame))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True)
        )
        path_n = f"temp/temp-{frame}-{self.track_id}-{int(time.time())}.jpg"
        with open(path_n, "wb") as binary_file:
            binary_file.write(out)

        out = ffmpeg.input(path_n)
        return out

    def init_strip(self):
        pass

    def get_frame(self, frame: int):
        if self.type == "image":
            im = self.read_image()
            return im
        elif self.type == "video":
            vid = self.read_video_by_frame(frame)
            return vid
        else:
            return None

    def read_video_info(self, in_filename):
        probe = ffmpeg.probe(in_filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        self.video_info = video_stream


# một sequence
class Sequence:
    def __init__(self, strips: List[Strip] = [], n_frame: int = 0):
        self.n_frame = n_frame
        self.fps = 30  # 30 frame per second
        self.final_frame_cache = []
        self.strips: List[Strip] = strips

        self.strips_dict_by_frame = {}
        self.img_dict_by_frame = {}
        self.init_temp()

    def init_temp(self):
        files = glob.glob('temp/*')
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
            if img is None:
                img = strip.get_frame(frame)
            else:
                img = self.overlay(img, strip.get_frame(frame))
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
