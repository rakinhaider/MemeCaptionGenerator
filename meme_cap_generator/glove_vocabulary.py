from .vocabulary import Vocabulary
import pickle
import os


class GloveVocabulary(Vocabulary):
    def __init__(self, dim):
        super(GloveVocabulary, self).__init__()
        self.dim = dim

    def load_vocab(self, file_name, data_dir=None):
        file_name = os.path.join(data_dir,
                                  'glove', 'processed',
                                  file_name + '.' + str(self.dim))

        with open(file_name+'_w2i.pkl', 'rb') as f:
            self.word2idx = pickle.load(f)
        with open(file_name+'_i2w.pkl', 'rb') as f:
            self.idx2word = pickle.load(f)
        self.length = len(self.word2idx)
        self.add_word('<unk>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<pad>')

    def save_vocab(self, file_name='vocab.pkl', data_dir=None):
        pass