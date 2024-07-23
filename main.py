from pavo import render_video, json_video_render, json_render_with_s3_asset


def main():
    # render_video(
    #     '/Users/admin/Desktop/pavo-engine-py/notebook/output2/new_data.json',
    #     '/Users/admin/Desktop/pavo-engine-py/output3/output2.mp4'
    # )
    # json_video_render(
    #     '/Users/admin/Desktop/pavo-engine-py/docs/data2.json',
    #     '/Users/admin/Desktop/pavo-engine-py/vinhdemo/vinh.mp4'
    # )

    S3_BUCKET_RESOURCE = 'video-database'
    S3_BUCKET_RESULT = 'cotifyai'
    S3_ACESS_KEY = 'AKIA3W7YJLAERCMHQLVW'
    S3_SECRET_KEY = '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'


    json_render_with_s3_asset(
        '/Users/admin/Desktop/pavo-engine-py/docs/data3.json',
        '/Users/admin/Desktop/pavo-engine-py/vinhdemo/vinh.mp4',
        S3_BUCKET_RESOURCE, S3_ACESS_KEY, S3_SECRET_KEY
    )


if __name__ == '__main__':
    main()
