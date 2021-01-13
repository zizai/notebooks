import os
import requests
from pprint import pprint
from zipfile import ZipFile

'''
https://github.com/erikwijmans/Pointnet2_PyTorch
https://github.com/WangYueFt/dgcnn
https://github.com/FrankCAN/GAPNet

'''



API_TOKEN = '0ac247d28da349fa801934042b4c655c'


def main():
    url = 'https://api.sketchfab.com/v3/models/d36762d2e111470b988fdfdd2d333955/download'
    headers = {
        'Authorization': 'Token {}'.format(API_TOKEN)
    }

    res = requests.get(url, headers=headers)
    pprint(res.status_code)

    if res.status_code == requests.codes.ok:
        pprint(res.json())
        url = res.json()['gltf']['url']

        res = requests.get(url)
        filepath = os.path.expanduser('~/Downloads/sample_model.zip')
        with open(filepath, 'wb') as fd:
            for chunk in res.iter_content(chunk_size=128):
                fd.write(chunk)
        with ZipFile(filepath) as zip_obj:
            new_filepath = os.path.expanduser('~/Downloads/sample_model')
            zip_obj.extractall(new_filepath)


if __name__ == '__main__':
    main()
