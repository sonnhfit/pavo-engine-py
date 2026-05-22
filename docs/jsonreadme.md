# Pavo Lang: Python DSL for Timeline JSON

Pavo Lang is a Python-based DSL for creating valid Pavo timeline JSON without manually writing raw JSON.

## 1) Goal

- Author timelines with readable Python code.
- Generate JSON that matches `pavo.schema`.
- Separate timeline authoring from rendering.

## 2) Import

```python
from pavo.pavolang import PavoVideo
```

## 3) Core API

### `PavoVideo(...)`

```python
PavoVideo(name, width, height, fps=25, background="#000000")
```

- `name`: video/project name
- `width`, `height`: output resolution
- `fps`: frames per second
- `background`: hex background color

### `addText(...)`

```python
node = video.addText(
    text="Hello",
    fontSize=72,
    fontWeight=800,
    color="#ffffff",
    x="center",
    y="center",
    trackId=0,
)
```

Returns a `PavoText` object that supports `.animate(...)`.

### `PavoText.animate(from_state, to_state, options)`

```python
node.animate(
    {"opacity": 0, "scale": 0.8},
    {"opacity": 1, "scale": 1},
    {"duration": "0.8s"},
)
```

- Each `animate` call adds one strip to the timeline.
- The playhead advances automatically by `duration`.

### `wait(duration)`

```python
video.wait("1.5s")
```

Moves the playhead forward without adding a strip.

### `setSoundtrack(...)`

```python
video.setSoundtrack(src="music.mp3", effect="fadeOut")
```

### Advanced: `addStrip(...)`

```python
video.addStrip(
    trackId=1,
    start="2s",
    length="3s",
    asset={"type": "image", "src": "assets/intro.jpg"},
)
```

Use this when you want full control over strip data.

### Export

```python
payload = video.to_dict()      # validated dict
text = video.to_json()         # JSON string
video.save_json("timeline.json")
```

## 4) Duration Rules

Supported duration formats:

- `"1.5s"` (seconds)
- `"800ms"` (milliseconds)
- `float` (seconds)
- `int` (frames)

## 5) Current Animation Mapping

`animate(from_state, to_state, ...)` maps to `asset.animation` as follows:

- `opacity: 0 -> 1` => `fadeIn`
- `opacity: 1 -> 0` => `fadeOut`
- `x:* -> 0` => `slideInLeft`

## 6) Full Example

```python
from pavo.pavolang import PavoVideo

video = PavoVideo(
    name="My Video",
    width=1920,
    height=1080,
    fps=30,
)

title = video.addText(
    text="Hello, Pavo!",
    fontSize=72,
    fontWeight=800,
    color="#ffffff",
)

title.animate(
    {"opacity": 0, "scale": 0.8},
    {"opacity": 1, "scale": 1},
    {"duration": "0.8s"},
)
video.wait("1.5s")
title.animate(
    {"opacity": 1, "scale": 1},
    {"opacity": 0, "scale": 1.2},
    {"duration": "0.8s"},
)

video.save_json("./hello.json")
```

Then render:

```python
from pavo import render_video

render_video("./hello.json", "./hello.mp4")
```
