import wget
from wget import bar_adaptive
import os
import gzip
import shutil
import argparse
import zipfile
import git


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
    # The Git folder need to be cloned and cleaned.
    url = 'https://github.com/alpv95/MemeProject.git'
    repo = git.Repo.clone_from(url, os.path.join('data/repo/', 'repo'), branch='master')

    heads = repo.heads
    master = heads.master  # lists can be accessed by name for convenience
    master.commit  # the commit pointed to by head called master
    master.rename('new_name')  # rename heads
    master.rename('master')


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
