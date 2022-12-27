import ffmpeg


overlay = ffmpeg.input('data/overlay.png')
overlay2 = ffmpeg.input('data/out.jpg')
print(overlay)

stream = overlay2.overlay(overlay)

stream.output('out22.jpg').run(capture_stdout=True)
