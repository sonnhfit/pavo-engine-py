# Pavo Lang: Python DSL để tạo timeline JSON

**English summary:** Use `PavoVideo` to build timeline JSON from Python, then export with `to_dict()/to_json()/save_json()` and render with `pavo.render_video`.

`Pavo Lang` là một lớp DSL dựa trên Python, giúp người dùng tạo JSON hợp lệ cho Pavo Engine mà không cần viết JSON thủ công.

## 1) Ý tưởng

- Viết timeline bằng Python (dễ đọc, dễ tái sử dụng).
- DSL sẽ sinh ra JSON đúng schema của `pavo.schema`.
- Sau đó dùng `render_video(json_path, mp4_path)` để render.

## 2) Import

```python
from pavo.pavo_lang import PavoVideo
```

## 3) API cốt lõi

### `PavoVideo(...)`

```python
PavoVideo(name, width, height, fps=25, background="#000000")
```

- `name`: tên project/video
- `width`, `height`: độ phân giải output
- `fps`: frame rate
- `background`: màu nền hex

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

Trả về object `PavoText`, dùng để gọi `.animate(...)`.

### `PavoText.animate(from_state, to_state, options)`

```python
node.animate(
    {"opacity": 0, "scale": 0.8},
    {"opacity": 1, "scale": 1},
    {"duration": "0.8s"},
)
```

- Mỗi lần `animate` sẽ tạo 1 strip trên timeline.
- Playhead tự động tiến thêm `duration`.

### `wait(duration)`

```python
video.wait("1.5s")
```

Di chuyển playhead mà không thêm strip.

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

Dùng khi cần toàn quyền tạo strip theo schema.

### Export

```python
payload = video.to_dict()      # dict đã validate schema
text = video.to_json()         # JSON string
video.save_json("timeline.json")
```

## 4) Duration rules

DSL hỗ trợ 3 kiểu duration:

- `"1.5s"` (seconds)
- `"800ms"` (milliseconds)
- `float` (seconds)
- `int` (frames)

## 5) Mapping animation hiện tại

`animate(from_state, to_state, ...)` được map sang field `asset.animation`:

- `opacity: 0 -> 1` => `fadeIn`
- `opacity: 1 -> 0` => `fadeOut`
- `x:* -> 0` => `slideInLeft`

## 6) Ví dụ hoàn chỉnh (theo style VideoFlow)

```python
from pavo.pavo_lang import PavoVideo

video = PavoVideo(
    name="My Video",
    width=1920,
    height=1080,
    fps=30,
)

title = video.addText(
    text="Hello, VideoFlow!",
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

Sau đó render:

```python
from pavo import render_video

render_video("./hello.json", "./hello.mp4")
```
