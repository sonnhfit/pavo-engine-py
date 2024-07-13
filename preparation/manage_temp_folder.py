import os
import json
import requests
from urllib.parse import urlparse


def download_asset(url, output_path, retries=3):
    while retries > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, 'wb') as file:
                file.write(response.content)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            retries -= 1
    return False


def process_timeline(timeline):
    temp_folder = 'temp'
    os.makedirs(temp_folder, exist_ok=True)

    for track in timeline['tracks']:
        track_folder = os.path.join(temp_folder, f"track_{track['track_id']}")
        os.makedirs(track_folder, exist_ok=True)

        strips = sorted(track['strips'], key=lambda x: x['start'])

        for i, strip in enumerate(strips):
            asset = strip['asset']
            asset_type = asset['type']
            asset_url = asset['src']

            file_name = os.path.basename(urlparse(asset_url).path)
            output_path = os.path.join(track_folder, f"{i+1}_{file_name}")

            print(f"Downloading {asset_type}: {asset_url}")
            if asset_type == 'video':
                success = download_asset(asset_url, output_path, retries=5)
            else:
                success = download_asset(asset_url, output_path)

            if success:
                print(f"Downloaded {asset_type} successfully: {output_path}")
            else:
                print(f"Failed to download {asset_type}: {asset_url}")


# Read the JSON data from a file or a string
with open('timeline.json', 'r') as file:
    data = json.load(file)

# Process the timeline
process_timeline(data['timeline'])
