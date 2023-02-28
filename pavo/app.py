import os
import ffmpeg
import requests
from sequancer.render import render
from utils import create_temp_folder


def clear_temp():
    for f in os.listdir('temp'):
        if f.endswith('.jpg'):
            os.remove(f'temp/{f}')


def download_file(url):
    local_filename = url.split('/')[-1]

    response = requests.get(url)
    with open(local_filename, "wb") as file:
        file.write(response.content)
    return local_filename


def render_video_from_strips(list_strip, video_id=None):
    fpath = create_temp_folder(video_id)

    for i, strip in enumerate(list_strip):
        if strip is not None:
            print(f'Processing frame {i}...')
            if os.path.exists(f'{fpath}/im-{i}.jpg'):
                os.remove(f'{fpath}/im-{i}.jpg')
            strip.output(f'{fpath}/im-{i}.jpg').run(capture_stdout=True)
    (
        ffmpeg
        .input(f'{fpath}/*.jpg', pattern_type='glob', framerate=25)
        .output(f'output/{video_id}.mp4')
        .run()
    )

    clear_temp()
    return f'output/{video_id}.mp4'


def main():
    list_strip = render('docs/data.json')
    render_video_from_strips(list_strip)


if __name__ == '__main__':
    download_file(ur ="https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
)
