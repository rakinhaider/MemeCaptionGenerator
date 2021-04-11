# Adapted from Pytorch Image Captioning Tutorial.
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py

import nltk
from collections import Counter
import argparse
import os
import pickle
from pycontractions import Contractions
import string


def get_contractor(data_dir):
    # Load your favorite semantic vector model in gensim keyedvectors format from disk
    w2vec_path = os.path.join(data_dir, 'GoogleNews-vectors-negative300.bin')
    cont = Contractions(w2vec_path)

    # or specify any model from the gensim.downloader api
    # cont = Contractions(api_key="glove-twitter-100")

    # optional, prevents loading on first expand_texts call
    cont.load_models()
    print('Contraction Loaded.')
    return cont


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.length = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.length
            self.idx2word[self.length] = word
            self.length += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def load_vocab(self, file_name='vocab.pkl', data_dir=None):
        file_name = os.path.join(data_dir, file_name)
        f = open(file_name, 'rb')
        vocab = pickle.load(f)
        f.close()
        self.word2idx = vocab.word2idx
        self.idx2word = vocab.idx2word
        self.length = vocab.length
        return self

    def save_vocab(self, file_name='vocab.pkl', data_dir=None):
        file_name = os.path.join(data_dir, file_name)
        f = open(file_name, 'wb')
        pickle.dump(self, f)
        f.close()


def build_vocab(data_dir, caption_file, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    cont = get_contractor(data_dir)
    punc_table = dict((ord(char), None) for char in string.punctuation)
    caption_path = os.path.join(data_dir, caption_file)
    with open(caption_path, 'r') as f:
        line_num = 0
        for line in f:
            splits = line.split('-')
            img_id, caption = splits[0], splits[1].strip().lower()
            caption = list(cont.expand_texts([caption], precise=True))[0]
            caption = caption.translate(punc_table)
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
    args = parser.parse_args()
    print(args.data, args.file, args.thresh)
    vocab = build_vocab(args.data, args.file, args.thresh)
    print(len(vocab), vocab.length)
    vocab_file = os.path.join(args.data, 'vocab.pkl')
    pickle.dump(vocab, vocab_file)

    vocab.save_vocab(file_name='vocab.pkl', data_dir=args.data)