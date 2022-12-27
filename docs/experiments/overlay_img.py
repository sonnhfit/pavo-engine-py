import ffmpeg


overlay = ffmpeg.input('data/out_frame_123.jpg')
overlay2 = ffmpeg.input('data/out.jpg')
print(overlay)

stream = overlay2.overlay(overlay)

stream.output('out2233.jpg').run(capture_stdout=True)

