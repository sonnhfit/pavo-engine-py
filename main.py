from pavo import render_video, json_video_render


def main():
    # render_video(
    #     '/Users/admin/Desktop/pavo-engine-py/notebook/output2/new_data.json',
    #     '/Users/admin/Desktop/pavo-engine-py/output3/output2.mp4'
    # )
    json_video_render(
        '/Users/admin/Desktop/pavo-engine-py/docs/data2.json',
        '/Users/admin/Desktop/pavo-engine-py/vinhdemo/vinh.mp4'
    )


if __name__ == '__main__':
    main()
