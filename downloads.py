import wget
from wget import bar_adaptive
import os
import gzip
import shutil
import argparse
import zipfile
import git


def download_file(base_url, file_name, out_dir_name):
    os.makedirs(out_dir_name, exist_ok=True)
    if not os.path.exists(out_dir_name + file_name):
        filename = wget.download(base_url + file_name,
                                 out=out_dir_name + file_name,
                                 bar=bar_adaptive)
    else:
        pass


def download_word_embedding(args, embedding_name):
    if embedding_name == 'google':
        base_url = 'https://s3.amazonaws.com/dl4j-distribution/'
        name = 'GoogleNews-vectors-negative300.bin'
        dir_name = args.data
        download_file(base_url, name + '.gz', dir_name)
        if not os.path.exists(dir_name + name):
            with gzip.open(dir_name + name + '.gz', 'rb') as f_in:
                with open(dir_name + name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    elif embedding_name == 'glove':
        base_url = 'http://nlp.stanford.edu/data/'
        name = 'glove.6B'
        dir_name = args.data
        download_file(base_url, name + '.zip', dir_name)
        with zipfile.ZipFile(dir_name + '/' + name + '.zip', "r") as zip_ref:
            zip_ref.extractall(dir_name + '/glove/raw')


def download_memes(args):
    # The Git folder need to be cloned and cleaned.
    # This code doesn't work. Wanted to clone the previous
    # repo rather than pushing all memes in remote.
    exit()
    url = 'https://github.com/alpv95/MemeProject.git'
    repo = git.Repo.clone_from(url, os.path.join('data/repo/', 'repo'), branch='master')

    heads = repo.heads
    master = heads.master  # lists can be accessed by name for convenience
    master.commit  # the commit pointed to by head called master
    master.rename('new_name')  # rename heads
    master.rename('master')


def download_captions(args):
    # Should download captions here.
    base_url = 'https://raw.githubusercontent.com/alpv95/' \
               'MemeProject/master/im2txt/'
    name = 'Captions.txt'
    dir_name = args.data
    download_file(base_url, name, dir_name)
    name = 'CaptionsClean.txt'
    download_file(base_url, name, dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='data/',
                        help='Data Directory')
    parser.add_argument('--all', action='store_true',
                        help='Download memes, captions, word embedding.')
    parser.add_argument('--what', choices=['meme', 'caption', 'embedding'],
                        help='Specifies what to download.)')
    parser.add_argument('--emb', choices=['google', 'glove'],
                        help='Specifies which embedding to download.)')
    args = parser.parse_args()

    if args.all:
        download_word_embedding(args, 'google')
        download_word_embedding(args, 'glove')
        download_memes(args)
    elif args.what == 'meme':
        download_memes(args)
    elif args.what == 'captions':
        download_captions(args)
    elif args.what == 'embedding':
        if args.emb == 'google':
            download_word_embedding(args, 'google')
        elif args.emb == 'glove':
            download_word_embedding(args, 'glove')

