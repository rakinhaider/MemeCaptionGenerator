import numpy as np
import argparse
import torch
import pickle
import os
from utils import MyProgressBar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--dir', type=str, default='data/glove/')
    args = parser.parse_args()

    idx = 0
    word2idx = {}
    idx2word = {}
    vectors = []

    file_name = '{:s}/raw/glove.6B.{:d}d.txt'.format(args.dir, args.dim)
    bar = MyProgressBar(os.path.getsize(file_name))
    with open(file_name, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
            vect = [np.float(f) for f in line[1:]]
            vectors.append(vect)
            bar.update(len(l))
            # break
    bar.finish()
    vectors = torch.tensor(vectors)
    print(vectors.shape)
    os.makedirs(os.path.join(args.dir, 'processed'), exist_ok=True)
    output_path = os.path.join(args.dir, 'processed',
                               'glove.6B.{:d}'.format(args.dim))
    pickle.dump(vectors, open(output_path + '_dat.pkl', 'wb'))
    pickle.dump(word2idx, open(output_path + '_w2i.pkl', 'wb'))
    pickle.dump(idx2word, open(output_path + '_i2w.pkl', 'wb'))

    vectors = pickle.load(open(output_path + '_dat.pkl', 'rb'))
    word2idx = pickle.load(open(output_path + '_w2i.pkl', 'rb'))
    idx2word = pickle.load(open(output_path + '_i2w.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in word2idx}
    print(glove['the'])