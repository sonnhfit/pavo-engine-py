import json

from pavo.pavolang import PavoVideo


class TestPavoLang:
    def test_text_animate_and_wait_timeline(self):
        video = PavoVideo(name="demo", width=1920, height=1080, fps=30)

        title = video.addText(text="Hello", fontSize=72, color="#ffffff")
        title.animate({"opacity": 0}, {"opacity": 1}, {"duration": "0.8s"})
        video.wait("1.5s")
        title.animate({"opacity": 1}, {"opacity": 0}, {"duration": "0.8s"})

        data = video.to_dict()
        strips = data["timeline"]["tracks"][0]["strips"]

        assert len(strips) == 2
        assert strips[0]["start"] == 0
        assert strips[0]["length"] == 24
        assert strips[0]["asset"]["animation"] == "fadeIn"

        expected_second_start = 24 + round(1.5 * 30)
        assert strips[1]["start"] == expected_second_start
        assert strips[1]["length"] == 24
        assert strips[1]["asset"]["animation"] == "fadeOut"

    def test_duration_units(self):
        video = PavoVideo(name="demo", width=1280, height=720, fps=25)

        assert video.parse_duration_to_frames("500ms", min_frames=0) == 12
        assert video.parse_duration_to_frames("1.2s", min_frames=0) == 30
        assert video.parse_duration_to_frames(10, min_frames=0) == 10
        assert video.parse_duration_to_frames(2.0, min_frames=0) == 50

    def test_add_strip_advanced(self):
        video = PavoVideo(name="demo", width=1280, height=720, fps=25)
        video.addStrip(
            trackId=1,
            start="2s",
            length="3s",
            asset={"type": "image", "src": "a.jpg"},
        )

        data = video.to_dict()
        track = data["timeline"]["tracks"][0]

        assert track["track_id"] == 1
        assert track["strips"][0]["start"] == 50
        assert track["strips"][0]["length"] == 75

    def test_set_soundtrack(self):
        video = PavoVideo(name="demo", width=1280, height=720, fps=25)
        video.setSoundtrack(src="music.mp3", effect="fadeOut")

        data = video.to_dict()
        assert data["timeline"]["soundtrack"] == {"src": "music.mp3", "effect": "fadeOut"}

    def test_save_json(self, tmp_path):
        video = PavoVideo(name="demo", width=1280, height=720, fps=25.0)
        title = video.addText(text="Hi")
        title.animate({"opacity": 0}, {"opacity": 1}, {"duration": "1s"})

        output = tmp_path / "timeline.json"
        path = video.save_json(output)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["output"]["fps"] == 25.0
        assert payload["timeline"]["tracks"][0]["strips"][0]["asset"]["content"] == "Hi"

    def test_add_subtitle_landscape_defaults(self):
        video = PavoVideo(name="demo", width=1920, height=1080, fps=30)
        subtitle = video.addSubtitle(text="Hello subtitle")

        subtitle.animate({"opacity": 0}, {"opacity": 1}, {"duration": "1s"})
        asset = video.to_dict()["timeline"]["tracks"][0]["strips"][0]["asset"]

        assert asset["type"] == "subtitle"
        assert asset["content"] == "Hello subtitle"
        assert asset["position"] == {"x": "center", "y": 952}
        assert asset["background_color"] == "black@0.55"
        assert asset["stroke_color"] == "black"
        assert asset["stroke_width"] == 2

    def test_add_subtitle_portrait_defaults(self):
        video = PavoVideo(name="demo", width=1080, height=1920, fps=30)
        subtitle = video.addSubtitle(text="Hello portrait")

        subtitle.animate({"opacity": 0}, {"opacity": 1}, {"duration": "1s"})
        asset = video.to_dict()["timeline"]["tracks"][0]["strips"][0]["asset"]

        assert asset["position"] == {"x": "center", "y": 1056}

    def test_add_subtitle_style_and_position_overrides(self):
        video = PavoVideo(name="demo", width=1080, height=1920, fps=30)
        subtitle = video.addSubtitle(
            text="Styled subtitle",
            fontSize=36,
            color="#ffeeaa",
            font="assets/fonts/Inter-Bold.ttf",
            backgroundColor="#000000AA",
            strokeColor="#111111",
            strokeWidth=3,
            lineSpacing=6,
            x=120,
            y=900,
            trackId=3,
        )

        subtitle.animate({"opacity": 0}, {"opacity": 1}, {"duration": "1s"})
        track = video.to_dict()["timeline"]["tracks"][0]
        asset = track["strips"][0]["asset"]

        assert track["track_id"] == 3
        assert asset["size"] == 36
        assert asset["color"] == "#ffeeaa"
        assert asset["font"] == "assets/fonts/Inter-Bold.ttf"
        assert asset["background_color"] == "#000000AA"
        assert asset["stroke_color"] == "#111111"
        assert asset["stroke_width"] == 3
        assert asset["line_spacing"] == 6
        assert asset["position"] == {"x": 120, "y": 900}
