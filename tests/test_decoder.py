import unittest
import torch
from meme_cap_generator.decoder import Decoder
from meme_cap_generator.vocabulary import Vocabulary


class TestDecoder(unittest.TestCase):
    def test_init(self):
        vocab = Vocabulary().load_vocab(file_name='test_vocab.pkl',
                                        data_dir='../data/')
        d = Decoder(5, 5, vocab, 1)
        # TODO: convert to assertions
        print(d)

    def test_forward(self):
        torch.manual_seed(23)
        vocab = Vocabulary().load_vocab(file_name='test_vocab.pkl',
                                        data_dir='../data/')
        d = Decoder(5, 5, vocab, 1)
        captions = torch.tensor([[vocab(w) for w in ['i', 'upvote', 'no']],
                                 [vocab(w) for w in ['more', 'meme', '<pad>']]])

        lengths = [3, 2]

        output = d.forward(torch.ones((2, 5)), captions, lengths)
        assert output.shape == torch.Size([2, 3, 13])