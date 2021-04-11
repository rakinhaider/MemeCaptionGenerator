import wget
from wget import bar_adaptive
import os
import gzip
import shutil
import argparse
import zipfile
import requests


def download_word_embedding(args):
    base_url = 'https://s3.amazonaws.com/dl4j-distribution/'
    name = 'GoogleNews-vectors-negative300.bin'
    dir_path = args.data
    os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(dir_path + name + '.gz'):
        filename = wget.download(base_url + name + '.gz',
                                 out=dir_path + name + '.gz',
                                 bar=bar_adaptive)
    elif not os.path.exists(dir_path + name):
        with gzip.open(dir_path + name + '.gz', 'rb') as f_in:
            with open(dir_path + name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def download_memes(args):
    url = 'https://minhaskamal.github.io/DownGit/#/home?' \
          'url=https://github.com/alpv95/MemeProject/tree/master/im2txt/ops&file_name=ops.zip'
    r = requests.get(url, allow_redirects=True)
    print(r)
    file_name = os.path.join(args.data, 'ops.zip')
    f = open(file_name, 'wb')
    f.write(r.content)
    f.close()

    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(args.data)


def download_captions(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='data/',
                        help='Data Directory')
    parser.add_argument('--all', action='store_true',
                        help='Download memes, captions, word embedding.')
    parser.add_argument('--what', choices=['meme', 'caption', 'embedding'],
                        help='Specifies what to download.)')
    args = parser.parse_args()

    if args.all:
        download_word_embedding(args)
        download_memes(args)
    elif args.what == 'meme':
        download_memes(args)
    elif args.what == 'captions':
        download_captions(args)
    elif args.what == 'embedding':
        download_word_embedding(args)
