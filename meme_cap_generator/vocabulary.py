import os
import pickle


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

    def load_vocab(self, file_name, data_dir=None):
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