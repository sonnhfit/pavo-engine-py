from typing import List, Tuple, Dict, Union, Optional, Any, Callable, TypeVar, Generic, Type
import numpy as np
import cv2 as cv

class Effect:
    def __init__(self):
        self.type = None


# một element trong sequence
class Strip:
    def __init__(self):
        self.type = None
        self.name = None
        self.media_source = None
        self.media_source_type = None
        self.media_source_path = None

        self.track_id = 0
        self.start_frame = 0
        self.length = 0
        self.effect: List[Effect] = []

        self.data: Any = None

        self.video_start_frame = 0
        self.video_length = 0


# một sequence
class Sequence:
    def __init__(self):
        self.n_frame = 0
        self.fps = 30 # 30 frame per second

        self.strips: List[Strip] = []


class RenderSequence:
    def __init__(self):
        self.sequence = Sequence()
