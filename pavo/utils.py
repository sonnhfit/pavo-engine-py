import os
import time


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def create_temp_folder(folder_path):
    create_folder(f'temp/{folder_path}')
    return f'temp/{folder_path}'

def remove_temp_folder(folder_path):
    os.removedirs(f'temp/{folder_path}')


def dowload_resource(url, file_path):
    import requests
    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)

# if __name__ == '__main__':
#     create_temp_folder('testabc')
#     create_temp_folder('testabc2')
#     time.sleep(5)
#     remove_temp_folder('testabc')
