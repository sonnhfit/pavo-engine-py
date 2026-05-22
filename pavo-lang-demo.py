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