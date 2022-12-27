import pytest
from sequancer.seq import Strip, Sequence


strip1 = Strip(start_frame=0)
strip2 = Strip(start_frame=3)
strip3 = Strip(start_frame=2)
strip4 = Strip(start_frame=5)

strips = [strip1, strip2, strip3, strip4]

seq = Sequence(strips=strips)

def test_sort_strip_list():
    seq.sort_strips_by_start_frame()
    assert seq.strips[0].start_frame == 0 # strip1
    assert seq.strips[1].start_frame == 2 # strip2
    assert seq.strips[2].start_frame == 3 # strip3
    assert seq.strips[3].start_frame == 5 # strip4

