"""Tests for remote HTTP/HTTPS asset resolution (pavo/preparation/preparetion.py)."""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

from pavo.preparation.preparetion import (
    is_remote_url,
    download_remote_asset,
    resolve_remote_assets,
    _DEFAULT_CACHE_DIR,
)


# ---------------------------------------------------------------------------
# is_remote_url
# ---------------------------------------------------------------------------

class TestIsRemoteUrl:
    def test_http_url(self):
        assert is_remote_url("http://example.com/video.mp4") is True

    def test_https_url(self):
        assert is_remote_url("https://example.com/image.jpg") is True

    def test_local_path(self):
        assert is_remote_url("/path/to/local/file.mp4") is False

    def test_relative_path(self):
        assert is_remote_url("assets/video.mp4") is False

    def test_s3_url(self):
        assert is_remote_url("s3://bucket/key.mp4") is False

    def test_empty_string(self):
        assert is_remote_url("") is False

    def test_none_like_non_string(self):
        # is_remote_url should not crash on non-string input
        assert is_remote_url(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# download_remote_asset
# ---------------------------------------------------------------------------

class TestDownloadRemoteAsset:
    def test_raises_for_non_http_url(self, tmp_path):
        with pytest.raises(ValueError, match="Not a valid HTTP/HTTPS URL"):
            download_remote_asset("/local/path.mp4", cache_dir=str(tmp_path))

    def test_returns_cached_path_on_hit(self, tmp_path):
        """If the cached file already exists, no network request should be made."""
        import hashlib
        url = "https://example.com/clip.mp4"
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cached = tmp_path / (url_hash + ".mp4")
        cached.write_bytes(b"fake video data")

        with patch("pavo.preparation.preparetion.requests.head") as mock_head:
            result = download_remote_asset(url, cache_dir=str(tmp_path))

        mock_head.assert_not_called()
        assert result == str(cached)

    def test_raises_for_unreachable_url(self, tmp_path):
        with patch("pavo.preparation.preparetion.requests.head") as mock_head:
            mock_head.side_effect = requests.RequestException("connection refused")
            with pytest.raises(ValueError, match="unreachable"):
                download_remote_asset(
                    "https://unreachable.example.com/video.mp4",
                    cache_dir=str(tmp_path),
                )

    def test_raises_for_bad_status(self, tmp_path):
        with patch("pavo.preparation.preparetion.requests.head") as mock_head:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
            mock_head.return_value = mock_resp
            with pytest.raises(ValueError, match="unreachable"):
                download_remote_asset(
                    "https://example.com/missing.mp4",
                    cache_dir=str(tmp_path),
                )

    def test_downloads_and_caches_file(self, tmp_path):
        """A successful download should save the file and return its path."""
        url = "https://example.com/video.mp4"
        fake_content = b"video bytes"

        mock_head_resp = MagicMock()
        mock_head_resp.raise_for_status.return_value = None

        with patch("pavo.preparation.preparetion.requests.head", return_value=mock_head_resp):
            with patch("pavo.preparation.preparetion.download_video") as mock_dl:
                # Simulate download_video writing the file.
                def _fake_download(dl_url, dest, **kwargs):
                    with open(dest, "wb") as f:
                        f.write(fake_content)

                mock_dl.side_effect = _fake_download
                result = download_remote_asset(url, cache_dir=str(tmp_path))

        assert os.path.exists(result)
        assert result.startswith(str(tmp_path))
        mock_dl.assert_called_once()

    def test_extension_preserved(self, tmp_path):
        """The cached filename should preserve the original file extension."""
        url = "https://example.com/intro.jpg"

        mock_head_resp = MagicMock()
        mock_head_resp.raise_for_status.return_value = None

        with patch("pavo.preparation.preparetion.requests.head", return_value=mock_head_resp):
            with patch("pavo.preparation.preparetion.download_video") as mock_dl:
                mock_dl.side_effect = lambda u, dest, **kw: open(dest, "wb").write(b"img")
                result = download_remote_asset(url, cache_dir=str(tmp_path))

        assert result.endswith(".jpg")

    def test_creates_cache_dir_if_missing(self, tmp_path):
        url = "https://example.com/video.mp4"
        new_cache = str(tmp_path / "new" / "cache")

        mock_head_resp = MagicMock()
        mock_head_resp.raise_for_status.return_value = None

        with patch("pavo.preparation.preparetion.requests.head", return_value=mock_head_resp):
            with patch("pavo.preparation.preparetion.download_video") as mock_dl:
                mock_dl.side_effect = lambda u, dest, **kw: open(dest, "wb").write(b"v")
                download_remote_asset(url, cache_dir=new_cache)

        assert os.path.isdir(new_cache)


# ---------------------------------------------------------------------------
# resolve_remote_assets
# ---------------------------------------------------------------------------

class TestResolveRemoteAssets:
    def _make_timeline(self, src_value):
        return {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "video", "src": src_value},
                                "start": 0,
                                "length": 10,
                            }
                        ],
                    }
                ],
            }
        }

    def test_local_src_unchanged(self, tmp_path):
        data = self._make_timeline("/local/video.mp4")
        result = resolve_remote_assets(data, cache_dir=str(tmp_path))
        src = result["timeline"]["tracks"][0]["strips"][0]["asset"]["src"]
        assert src == "/local/video.mp4"

    def test_remote_src_replaced(self, tmp_path):
        url = "https://cdn.example.com/clip.mp4"
        data = self._make_timeline(url)

        with patch("pavo.preparation.preparetion.download_remote_asset") as mock_dl:
            mock_dl.return_value = "/cached/clip.mp4"
            result = resolve_remote_assets(data, cache_dir=str(tmp_path))

        src = result["timeline"]["tracks"][0]["strips"][0]["asset"]["src"]
        assert src == "/cached/clip.mp4"
        mock_dl.assert_called_once_with(url, str(tmp_path))

    def test_original_data_not_mutated(self, tmp_path):
        url = "https://cdn.example.com/clip.mp4"
        data = self._make_timeline(url)
        original_src = data["timeline"]["tracks"][0]["strips"][0]["asset"]["src"]

        with patch("pavo.preparation.preparetion.download_remote_asset") as mock_dl:
            mock_dl.return_value = "/cached/clip.mp4"
            resolve_remote_assets(data, cache_dir=str(tmp_path))

        # Original dict must be unchanged (deep-copy semantics).
        assert data["timeline"]["tracks"][0]["strips"][0]["asset"]["src"] == original_src

    def test_remote_soundtrack_resolved(self, tmp_path):
        url = "https://cdn.example.com/music.mp3"
        data = {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "tracks": [],
                "soundtrack": {"src": url, "effect": "fadeOut"},
            }
        }

        with patch("pavo.preparation.preparetion.download_remote_asset") as mock_dl:
            mock_dl.return_value = "/cached/music.mp3"
            result = resolve_remote_assets(data, cache_dir=str(tmp_path))

        assert result["timeline"]["soundtrack"]["src"] == "/cached/music.mp3"

    def test_local_soundtrack_unchanged(self, tmp_path):
        data = {
            "timeline": {
                "n_frames": 10,
                "background": "#000000",
                "tracks": [],
                "soundtrack": {"src": "/local/music.mp3"},
            }
        }
        result = resolve_remote_assets(data, cache_dir=str(tmp_path))
        assert result["timeline"]["soundtrack"]["src"] == "/local/music.mp3"

    def test_mixed_local_and_remote_strips(self, tmp_path):
        data = {
            "timeline": {
                "n_frames": 20,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "image", "src": "local.jpg"},
                                "start": 0,
                                "length": 10,
                            },
                            {
                                "asset": {"type": "video", "src": "https://cdn.example.com/v.mp4"},
                                "start": 10,
                                "length": 10,
                            },
                        ],
                    }
                ],
            }
        }

        with patch("pavo.preparation.preparetion.download_remote_asset") as mock_dl:
            mock_dl.return_value = "/cached/v.mp4"
            result = resolve_remote_assets(data, cache_dir=str(tmp_path))

        strips = result["timeline"]["tracks"][0]["strips"]
        assert strips[0]["asset"]["src"] == "local.jpg"
        assert strips[1]["asset"]["src"] == "/cached/v.mp4"
        mock_dl.assert_called_once()

    def test_text_strip_has_no_src(self, tmp_path):
        data = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "text", "content": "Hello"},
                                "start": 0,
                                "length": 5,
                            }
                        ],
                    }
                ],
            }
        }
        with patch("pavo.preparation.preparetion.download_remote_asset") as mock_dl:
            resolve_remote_assets(data, cache_dir=str(tmp_path))
        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# render_video integration – remote URL path
# ---------------------------------------------------------------------------

class TestRenderVideoRemoteUrl:
    """Verify render_video resolves remote URLs before calling the renderer."""

    def _write_timeline(self, path, src_value, is_remote=False):
        data = {
            "timeline": {
                "n_frames": 5,
                "background": "#000000",
                "tracks": [
                    {
                        "track_id": 0,
                        "strips": [
                            {
                                "asset": {"type": "image", "src": src_value},
                                "start": 0,
                                "video_start_frame": 0,
                                "length": 5,
                            }
                        ],
                    }
                ],
            },
            "output": {"format": "mp4", "fps": 25, "width": 320, "height": 240},
        }
        with open(path, "w") as fh:
            json.dump(data, fh)
        return data

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_render_video_with_remote_src(self, mock_rvfs, mock_render, tmp_path):
        """render_video with a remote src downloads and passes local path to renderer."""
        json_path = str(tmp_path / "timeline.json")
        url = "https://cdn.example.com/intro.jpg"
        local_cached = str(tmp_path / "cached_intro.jpg")
        # Create a fake cached file so the renderer doesn't crash.
        open(local_cached, "w").close()

        self._write_timeline(json_path, url)
        mock_render.return_value = [None] * 5

        # Capture the resolved JSON content while the temp dir still exists.
        captured = {}

        def _capture_render(json_file_path, temp_dir):
            with open(json_file_path) as fh:
                captured["data"] = json.load(fh)
            return [None] * 5

        mock_render.side_effect = _capture_render

        with patch(
            "pavo.preparation.preparetion.download_remote_asset",
            return_value=local_cached,
        ):
            from pavo.pavo import render_video

            render_video(json_path, str(tmp_path / "out.mp4"), cache_dir=str(tmp_path))

        # The renderer must have been called with a JSON that has the local path.
        assert "data" in captured, "render() side-effect was not invoked"
        src_in_rendered = (
            captured["data"]["timeline"]["tracks"][0]["strips"][0]["asset"]["src"]
        )
        assert src_in_rendered == local_cached

    @patch("pavo.pavo.render")
    @patch("pavo.pavo.render_video_from_strips")
    def test_render_video_with_local_src_unchanged(self, mock_rvfs, mock_render, tmp_path):
        """render_video with a local src does not call download_remote_asset."""
        json_path = str(tmp_path / "timeline.json")
        self._write_timeline(json_path, "local.jpg")
        mock_render.return_value = [None] * 5

        with patch(
            "pavo.preparation.preparetion.download_remote_asset"
        ) as mock_dl:
            from pavo.pavo import render_video

            render_video(json_path, str(tmp_path / "out.mp4"), cache_dir=str(tmp_path))

        mock_dl.assert_not_called()
