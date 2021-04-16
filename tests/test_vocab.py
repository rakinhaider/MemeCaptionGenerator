import unittest
from build_vocab import build_vocab
from meme_cap_generator.vocabulary import Vocabulary


class TestVocab(unittest.TestCase):
    w2i = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3,
           'meme': 4, 'y': 5, 'u': 6, 'no': 7, 'give': 8,
           'more': 9, '?': 10, 'i': 11, 'do': 12}
    i2w = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>',
           4: 'meme', 5: 'y', 6: 'u', 7: 'no', 8: 'give',
           9: 'more', 10: '?', 11: 'i', 12: 'do'}
    length = 13

    def test_build_vocab(self):
        data_dir = '../data/'
        caption_file = 'test_captions.txt'
        threshold = 2
        vocab = build_vocab(data_dir, caption_file, threshold)
        assert vocab.word2idx == self.w2i
        assert vocab.idx2word == self.i2w
        assert vocab.length == self.length

    def test_save_vocab(self):
        vocab_load = Vocabulary()
        vocab_load.load_vocab(file_name='test_vocab.pkl',
                              data_dir='../data')
        assert vocab_load.length == self.length
        assert vocab_load.word2idx == self.w2i
        assert vocab_load.idx2word == self.i2w
