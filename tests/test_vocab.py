import unittest
from vocab import Vocabulary, build_vocab


class TestVocab(unittest.TestCase):
    def test_build_vocab(self):
        data_dir = '../data/'
        caption_file = 'test_captions.txt'
        threshold = 2
        vocab = build_vocab(data_dir, caption_file, threshold)
        print(vocab.word2idx)
        print(vocab.idx2word)
        print(len(vocab))
        vocab.save_vocab(file_name='test_vocab.pkl',
                         data_dir='../data/')

        vocab_load = Vocabulary().load_vocab(file_name='test_vocab.pkl',
                                             data_dir='../data/')
        print(vocab.word2idx)
        print(vocab.idx2word)
        print(vocab.length)
        assert vocab_load.length == vocab.length
        assert vocab_load.word2idx == vocab.word2idx
        assert vocab_load.idx2word == vocab.idx2word