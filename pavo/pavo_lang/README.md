# Pavo Lang (Python DSL)

Pavo Lang is a Python-first DSL for authoring schema-valid Pavo timeline JSON before rendering with `pavo.render_video`.

## Goal

- Write timeline logic in Python.
- Generate JSON that matches `pavo.schema`.
- Keep authoring and rendering as separate steps.

## Core API

- `PavoVideo(name, width, height, fps=25, background="#000000")`
- `addText(...) -> PavoText`
- `PavoText.animate(from_state, to_state, options)`
- `wait(duration)`
- `setSoundtrack(src, effect=None)`
- `addStrip(...)` (advanced)
- `to_dict()`, `to_json()`, `save_json(path)`

## Example

```python
from pavo.pavo_lang import PavoVideo

pavo = PavoVideo(
    name="My Video",
    width=1920,
    height=1080,
    fps=30,
)

title = pavo.addText(
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

pavo.wait("1.5s")

title.animate(
    {"opacity": 1, "scale": 1},
    {"opacity": 0, "scale": 1.2},
    {"duration": "0.8s"},
)

# export timeline JSON
pavo.save_json("./hello.json")
```

## Duration Rules

- `"1.5s"`, `"800ms"` are converted to frames using `fps`.
- `float` values are interpreted as seconds.
- `int` values are interpreted as frames.

## Current Animation Mapping

- `opacity: 0 -> 1` => `fadeIn`
- `opacity: 1 -> 0` => `fadeOut`
- `x:* -> 0` => `slideInLeft`

## Render from JSON

```python
from pavo import render_video

render_video("./hello.json", "./hello.mp4")
```
