import os
import ffmpeg
from sequancer.render import render


def clear_temp():
    for f in os.listdir('temp'):
        if f.endswith('.jpg'):
            os.remove(f'temp/{f}')


def render_video_from_strips(list_strip):
    for i, strip in enumerate(list_strip):
        if strip is not None:
            print(f'Processing frame {i}...')
            if os.path.exists(f'render_temp/im-{i}.jpg'):
                os.remove(f'render_temp/im-{i}.jpg')
            
            strip.output(f'render_temp/im-{i}.jpg').run(capture_stdout=True)

    (
        ffmpeg
        .input('render_temp/*.jpg', pattern_type='glob', framerate=25)
        .output('output/movie.mp4')
        .run()
    )

    clear_temp()



def main():
    list_strip = render('docs/data.json')
    render_video_from_strips(list_strip)


if __name__ == '__main__':
    main()
