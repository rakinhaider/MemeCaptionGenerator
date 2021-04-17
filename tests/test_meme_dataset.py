import unittest
from meme_cap_generator import MemeDataset, Vocabulary
from torchvision import transforms
from PIL import Image
import numpy as np


class TestMemeDataset(unittest.TestCase):
    def test_get_item(self):
        vocab = Vocabulary()
        vocab.load_vocab('test_vocab.pkl', '../data/')
        md = MemeDataset('../data/', 'test_captions.txt', vocab)
        md.load_dataset(-1)
        image, caption = md.__getitem__(3)
        print(image)
        print(caption)
        """
        # print(transformed)
        img = transforms.ToPILImage()(transformed)
        # print(np.asarray(img).transpose((2, 0, 1))/255)
        npimg = np.asarray(image)
        data = transformed.numpy().transpose(1, 2, 0)*255
        data = data.astype(npimg.dtype)
        print(npimg)
        print(data)
        assert np.all(npimg == data)
        Image.fromarray(npimg, mode='RGB').show()
        Image.fromarray(data, mode='RGB').show()
        """
        for idx in caption.data:
            print(vocab.idx2word[idx.item()], end=' ')

    def test_load_dataset(self):
        vocab = Vocabulary()
        vocab.load_vocab('test_vocab.pkl', '../data/')
        md = MemeDataset('../data/', 'test_captions.txt', vocab)
        md.load_dataset(-1)
        assert len(md) == 4
        assert md.ids == [
            'y-u-no', 'dont-you-squidward',
            'why-cant-i-hold-all-these', 'i-dont-know-who-you-are'
        ]
        assert {i: md.id2index[i] for i in md.ids} == {
            'y-u-no': 2465, 'dont-you-squidward': 608,
            'why-cant-i-hold-all-these': 2427, 'i-dont-know-who-you-are': 987
        }

    def test_load_dataset_full(self):
        vocab = Vocabulary()
        vocab.load_vocab('vocab_2_CaptionsClean_nopunc_t.pkl', '../data/')
        md = MemeDataset('../data/', 'CaptionsClean_nopunc_-1_t.txt', vocab)
        md.load_dataset(-1)
        print(len(md))
        assert len(md) == 394549
