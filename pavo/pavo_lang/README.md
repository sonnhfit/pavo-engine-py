# Pavo Lang (Python DSL)

`Pavo Lang` là DSL dựa trên Python để tạo timeline JSON hợp lệ cho Pavo Engine.

## Mục tiêu

- Viết cú pháp gần với VideoFlow.
- Tạo JSON chuẩn theo schema của `pavo.schema`.
- Tách bước **author timeline** (DSL) và **render video** (`pavo.render_video`).

## API chính

- `PavoVideo(name, width, height, fps=25, background="#000000")`
- `addText(...) -> PavoText`
- `PavoText.animate(from_state, to_state, options)`
- `wait(duration)`
- `setSoundtrack(src, effect=None)`
- `addStrip(...)` (advanced)
- `to_dict()`, `to_json()`, `save_json(path)`

## Ví dụ giống VideoFlow

```python
from pavo.pavo_lang import PavoVideo

pavo = PavoVideo(
    name="My Video",
    width=1920,
    height=1080,
    fps=30,
)

title = pavo.addText(
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

pavo.wait("1.5s")

title.animate(
    {"opacity": 1, "scale": 1},
    {"opacity": 0, "scale": 1.2},
    {"duration": "0.8s"},
)

# xuất ra JSON timeline
pavo.save_json("./hello.json")
```

## Quy ước duration

- `"1.5s"`, `"800ms"` → đổi sang frame theo `fps`.
- `float` → hiểu là giây.
- `int` → hiểu là số frame.

## Mapping animation hiện tại

- `opacity: 0 -> 1` → `fadeIn`
- `opacity: 1 -> 0` → `fadeOut`
- `x:* -> 0` → `slideInLeft`

Các state khác được giữ ở mức DSL (không ép vào schema animation nếu không có mapping trực tiếp).

## Render video từ JSON

```python
from pavo import render_video

render_video("./hello.json", "./hello.mp4")
```
