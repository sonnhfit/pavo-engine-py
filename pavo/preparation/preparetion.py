import copy
import hashlib
import json
import logging
import os
import time
from urllib.parse import urlparse

import requests
from PIL import Image
from retrying import retry
from tqdm import tqdm


# Default directory used to cache remotely downloaded assets.
_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pavo", "cache")


def download_video(url, destination, retries=5, chunk_size=1024):
    """
    Downloads a file from a URL with retry logic.

    :param url: The URL of the file to download.
    :param destination: The file path where the download will be saved.
    :param retries: The number of times to retry the download on failure.
    :param chunk_size: The size of each chunk to download at a time (in bytes).
    """
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes

            total_size = int(response.headers.get("content-length", 0))
            with open(destination, "wb") as file, tqdm(
                desc=destination,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    bar.update(len(chunk))
            print(f"Download completed successfully: {destination}")
            break
        except requests.RequestException as e:
            attempt += 1
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < retries:
                wait_time = attempt * 2  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {retries} attempts.")


# Retry settings: retry up to 5 times with a delay of 2 seconds between retries
@retry(stop_max_attempt_number=5, wait_fixed=2000)
def download_image(url, path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        with open(path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded image to {path}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        raise


def is_remote_url(src: str) -> bool:
    """Return ``True`` if *src* is an HTTP or HTTPS URL, ``False`` otherwise."""
    try:
        parsed = urlparse(src)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def download_remote_asset(url: str, cache_dir: str = _DEFAULT_CACHE_DIR) -> str:
    """Download a remote HTTP/HTTPS asset and return the local cached path.

    The file is stored in *cache_dir* using an MD5 hash of the URL as the
    filename so subsequent calls with the same URL reuse the cached copy.

    Parameters
    ----------
    url:
        Remote HTTP or HTTPS URL to download.
    cache_dir:
        Directory where downloaded files are cached.
        Defaults to ``~/.pavo/cache/``.

    Returns
    -------
    str
        Absolute path to the locally cached file.

    Raises
    ------
    ValueError
        If *url* is not a valid HTTP/HTTPS URL or the server is unreachable.
    """
    if not is_remote_url(url):
        raise ValueError(f"Not a valid HTTP/HTTPS URL: {url!r}")

    os.makedirs(cache_dir, exist_ok=True)

    # Build a stable local filename: MD5 of the URL + original extension.
    url_hash = hashlib.md5(url.encode()).hexdigest()
    parsed_path = urlparse(url).path
    _, ext = os.path.splitext(parsed_path)
    cached_filename = url_hash + (ext if ext else "")
    cached_path = os.path.join(cache_dir, cached_filename)

    if os.path.exists(cached_path):
        print(f"[pavo] Using cached asset: {cached_path}")
        return cached_path

    # Verify the URL is reachable before attempting a full download.
    try:
        head = requests.head(url, timeout=10, allow_redirects=True)
        head.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(
            f"Remote asset URL is unreachable: {url!r} — {exc}"
        ) from exc

    download_video(url, cached_path)
    return cached_path


def resolve_remote_assets(data: dict, cache_dir: str = _DEFAULT_CACHE_DIR) -> dict:
    """Replace remote URLs in *data*'s ``asset.src`` fields with local paths.

    Walks the timeline JSON, downloads any HTTP/HTTPS ``src`` values (using
    the local cache to avoid repeated downloads), and returns a deep copy of
    *data* with all remote URLs replaced by absolute local file paths.

    The ``soundtrack.src`` field is also resolved if it is a remote URL.

    Parameters
    ----------
    data:
        Parsed timeline JSON dict (the top-level object).
    cache_dir:
        Directory used for the local asset cache.

    Returns
    -------
    dict
        Deep copy of *data* with remote ``src`` values replaced by local paths.
    """
    data = copy.deepcopy(data)
    timeline = data.get("timeline", {})

    # Resolve soundtrack src if it is a remote URL.
    soundtrack = timeline.get("soundtrack")
    if isinstance(soundtrack, dict):
        src = soundtrack.get("src")
        if src and is_remote_url(src):
            print(f"[pavo] Downloading remote soundtrack: {src}")
            soundtrack["src"] = download_remote_asset(src, cache_dir)

    # Resolve each strip asset src.
    for track in timeline.get("tracks", []):
        for strip in track.get("strips", []):
            asset = strip.get("asset", {})
            src = asset.get("src")
            if src and is_remote_url(src):
                print(f"[pavo] Downloading remote asset: {src}")
                asset["src"] = download_remote_asset(src, cache_dir)

    return data


def download_file_from_s3(
    file_name,
    bucket,
    output_name=None,
    s3_acess_key=None,
    s3_secret_key=None,
    s3_prefix=None,
):
    folder_path = os.path.dirname(output_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    try:
        import boto3
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=s3_acess_key,
            aws_secret_access_key=s3_secret_key,
        )

        s3_client.download_file(bucket, file_name, output_name)
    except Exception as e:
        logging.error(e)
        return False
    return True


def create_directory_structure(json_file, output_folder):
    # Đọc dữ liệu từ file JSON
    with open(json_file) as file:
        data = json.load(file)

    # Tạo thư mục gốc
    root_dir = output_folder
    os.makedirs(root_dir, exist_ok=True)

    # Tạo thư mục cho timeline
    timeline_dir = os.path.join(root_dir, "timeline")
    os.makedirs(timeline_dir, exist_ok=True)

    # Tạo thư mục cho từng track
    tracks = data["timeline"]["tracks"]
    for i, track in enumerate(tracks):
        track_dir = os.path.join(timeline_dir, f"track_{i+1}")
        os.makedirs(track_dir, exist_ok=True)

        # Tạo thư mục cho video, ảnh và text trong mỗi track
        video_dir = os.path.join(track_dir, "video")
        image_dir = os.path.join(track_dir, "image")
        text_dir = os.path.join(track_dir, "text")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        # Tải và lưu các tệp tin vào thư mục tương ứng
        strips = track["strips"]
        for j, strip in enumerate(strips):
            asset = strip["asset"]
            asset_type = asset["type"]
            asset_src = asset["src"]

            if asset_type == "video":
                # Tải và lưu tệp tin video vào thư mục video
                output_path = os.path.join(video_dir, f"video_{j+1}.mp4")
                download_video(asset_src, output_path)
            elif asset_type == "image":
                # Tải và lưu tệp tin ảnh vào thư mục image
                output_path = os.path.join(image_dir, f"image_{j+1}.jpg")
                download_image(asset_src, output_path)


# Kiểm tra và chuyển đổi kích thước hình ảnh
def check_and_convert_image(image_path):
    image = Image.open(image_path)
    width, height = image.size

    # Kiểm tra nếu chiều cao không chia hết cho 2
    if height % 2 != 0:
        # Chuyển đổi kích thước hình ảnh để chiều cao chia hết cho 2
        new_height = height - 1
        new_image = image.resize((width, new_height))
        
        # Lưu hình ảnh mới
        new_image_path = image_path.replace(".jpg", "_converted.jpg")
        new_image.save(new_image_path)
        
        return new_image_path
    else:
        return image_path


def create_directory_structure_s3(
    json_file,
    output_folder,
    bucket=None,
    s3_acess_key=None,
    s3_secret_key=None
):
    # Đọc dữ liệu từ file JSON
    with open(json_file) as file:
        data = json.load(file)

    # Tạo thư mục gốc
    root_dir = output_folder
    os.makedirs(root_dir, exist_ok=True)

    # Tạo thư mục cho timeline
    timeline_dir = os.path.join(root_dir, "timeline")
    os.makedirs(timeline_dir, exist_ok=True)

    # Tạo thư mục cho từng track
    tracks = data["timeline"]["tracks"]
    for i, track in enumerate(tracks):
        track_dir = os.path.join(timeline_dir, f"track_{i+1}")
        os.makedirs(track_dir, exist_ok=True)

        # Tạo thư mục cho video, ảnh và text trong mỗi track
        video_dir = os.path.join(track_dir, "video")
        image_dir = os.path.join(track_dir, "image")
        text_dir = os.path.join(track_dir, "text")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        # Tải và lưu các tệp tin vào thư mục tương ứng
        strips = track["strips"]
        for j, strip in enumerate(strips):
            asset = strip["asset"]
            asset_type = asset["type"]
            asset_src = asset["src"]

            if asset_type == "video":
                # Tải và lưu tệp tin video vào thư mục video
                output_path = os.path.join(video_dir, f"video_{j+1}.mp4")
                print(f"Downloading video {asset_src} to {output_path}")
                
                download_file_from_s3(
                    asset_src, bucket, output_path, s3_acess_key, s3_secret_key
                )
            elif asset_type == "image":
                # Tải và lưu tệp tin ảnh vào thư mục image
                output_path = os.path.join(image_dir, f"image_{j+1}.jpg")
                print(f"Downloading image {asset_src} to {output_path}")
                download_file_from_s3(
                    asset_src, bucket, output_path, s3_acess_key, s3_secret_key
                )
                
                # Kiểm tra và chuyển đổi kích thước hình ảnh
                converted_image_path = check_and_convert_image(output_path)
                if converted_image_path != output_path:
                    print(f"Image size converted: {output_path} -> {converted_image_path}")
                    os.remove(output_path)  # Xóa hình ảnh gốc
                    os.rename(converted_image_path, output_path)  # Ghi đè ảnh đã convert vào ảnh gốc


def create_new_json(json_file, output_folder):
    # Đọc dữ liệu từ file JSON
    with open(json_file) as file:
        data = json.load(file)

    # Cập nhật đường dẫn tệp tin trong JSON
    tracks = data["timeline"]["tracks"]
    for i, track in enumerate(tracks):
        track_dir = os.path.join(output_folder, "timeline", f"track_{i+1}")
        strips = track["strips"]
        for j, strip in enumerate(strips):
            asset = strip["asset"]
            asset_type = asset["type"]

            if asset_type == "video":
                asset["src"] = os.path.join(track_dir, "video", f"video_{j+1}.mp4")
            elif asset_type == "image":
                asset["src"] = os.path.join(track_dir, "image", f"image_{j+1}.jpg")

    # Lưu dữ liệu JSON mới vào file
    new_json_file = os.path.join(output_folder, "new_data.json")
    with open(new_json_file, "w") as file:
        json.dump(data, file, indent=4)

    return new_json_file


def create_asset_tmp(json_file, output_folder):
    create_directory_structure(json_file, output_folder)
    create_new_json(json_file, output_folder)


def create_asset_tmp_s3(
    json_file,
    output_folder,
    bucket=None,
    s3_acess_key=None,
    s3_secret_key=None
):
    create_directory_structure_s3(
        json_file, output_folder, bucket, s3_acess_key, s3_secret_key
    )
    json_new_path = create_new_json(json_file, output_folder)
    return json_new_path
