import unittest
from meme_cap_generator import MemeDataset, Vocabulary
from torchvision import transforms
from PIL import Image
import numpy as np

class TestMemeDataset(unittest.TestCase):
    def test_get_item(self):
        vocab = Vocabulary()
        vocab.load_vocab('vocab_2_-1_g.pkl', '../data/')
        md = MemeDataset('../data/', 'CaptionsClean_nopunc_-1.txt', vocab)
        image, caption = md.__getitem__(2)
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
