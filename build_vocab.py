# Adapted from Pytorch Image Captioning Tutorial.
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py

import nltk
from collections import Counter
import argparse
import os
import pickle
from meme_cap_generator.vocabulary import Vocabulary
from pycontractions import Contractions
import string


def get_contractor(data_dir, embedding='g'):
    if embedding == 'g':
        # Load your favorite semantic vector model in gensim keyedvectors format from disk
        w2vec_path = os.path.join(data_dir, 'GoogleNews-vectors-negative300.bin')
        cont = Contractions(w2vec_path)

    if embedding == 't':
        # or specify any model from the gensim.downloader api
        cont = Contractions(api_key="glove-twitter-100")

    # optional, prevents loading on first expand_texts call
    cont.load_models()
    print('Contraction Loaded.')
    return cont


def read_captions(args):
    caption_file = os.path.join(args.data, args.file)
    ids = []
    captions = []
    with open(caption_file, 'r') as f:
        for line in f:
            if len(ids) == args.n_caps:
                break
            elif len(ids) % 1000 == 0:
                print('Captions Read', len(ids))
            splits = line.split('-')
            if len(splits) != 2:
                continue
            ids.append(splits[0].strip().lower())
            captions.append(splits[1].strip().lower())

    return ids, captions


def build_vocab(data_dir, caption_file, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    caption_path = os.path.join(data_dir, caption_file)
    with open(caption_path, 'r') as f:
        line_num = 0
        for line in f:
            splits = line.split('-')
            img_id, caption = splits[0], splits[1].strip()
            tokens = nltk.word_tokenize(caption)
            counter.update(tokens)
            line_num += 1
            if (line_num) % 100 == 0:
                print("[{}] Tokenized the captions.".format(line_num))

    print()
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def contract_captions(args):
    ids, captions = read_captions(args)
    print('Captions Read', len(ids))
    print('Loading Contractor')
    cont = get_contractor(args.data, args.embedding)
    contracted_generator = cont.expand_texts(captions)
    name, extension = os.path.splitext(args.file)
    name = '_'.join([name, 'contracted',
                     str(args.n_caps), args.embedding])
    file_name = os.path.join(args.data, name + extension)
    with open(file_name, 'w') as f:
        i = 0
        total = len(ids)
        # for id, caption in zip(ids, captions):
        for i, id in enumerate(ids):
            line = ' - '.join([id, next(contracted_generator)]) + '\n'
            f.write(line)
            if i % 1000 == 0:
                print('Contracted {}/{}'.format(i, total))
                f.flush()
            i += 1
        print('Contracted {}/{}'.format(i, total))


def remove_punctuations(args):
    punc_table = dict((ord(char), None) for char in string.punctuation)
    ids, captions = read_captions(args)
    name, extension = os.path.splitext(args.file)
    splits = name.split('_')
    splits[1] = 'nopunc'
    name = '_'.join(splits)
    file_name = os.path.join(args.data, name + extension)
    with open(file_name, 'w') as f:
        i = 0
        total = len(ids)
        for i, id in enumerate(ids):
            line = ' - '.join([id, captions[i].translate(punc_table)]) + '\n'
            f.write(line)
            if i % 1000 == 0:
                print('Processed {}/{}'.format(i, total))
                f.flush()
            i += 1
        print('Processed {}/{}'.format(i, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                      type=str, required=False,
                      help="Path to the data directory.",
                      default="data/")
    parser.add_argument('-f', '--file',
                      type=str, required=True,
                      help="Caption file name."
                      )
    parser.add_argument('-t', '--thresh',
                      type=int, required=False,
                      help="Frequency threshold to discard word.",
                      default=2
                      )
    parser.add_argument('-n', '--n-caps',
                      type=int, required=False,
                      help="Number of captions.",
                      default=-1
                      )
    parser.add_argument('-e', '--embedding',
                      choices=['g', 't'],
                      help="Google or Twitter Glove Embedding.",
                      default='g'
                      )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--contract', action='store_true',
                       help='Remove Contractions')
    group.add_argument('-p', '--punc', action='store_true',
                       help='Remove Punctuations')
    group.add_argument('-v', '--vocab', action='store_true',
                       help='Build Vocabulary')

    args = parser.parse_args()

    if args.contract:
        print(args.data, args.file, args.thresh, args.n_caps, args.embedding)
        contract_captions(args)
    elif args.punc:
        remove_punctuations(args)
    elif args.vocab:
        print(args.data, args.file, args.thresh, args.n_caps)
        vocab = build_vocab(args.data, args.file, args.thresh)
        name, extension = os.path.splitext(args.file)
        splits = name.split('_')
        vocab_file = '_'.join(['vocab'] + [str(args.thresh)] +
                              splits[:2] + splits[3:]) + '.pkl'
        print(len(vocab))
        vocab.save_vocab(file_name=vocab_file, data_dir=args.data)