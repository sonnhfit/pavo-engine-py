import requests

url = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
response = requests.get(url)

with open("big_buck_bunny.mp4", "wb") as file:
    file.write(response.content)