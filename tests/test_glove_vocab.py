import unittest
from meme_cap_generator import GloveVocabulary

class TestGloveVocabulary(unittest.TestCase):
    def test_load_glove(self):
        vocab = GloveVocabulary(50)
        vocab.load_vocab('glove.6B', '../data/')
        assert vocab.length == 400004