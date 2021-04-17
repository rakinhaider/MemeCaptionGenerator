import unittest
from meme_cap_generator.encoder import Encoder
from PIL import Image
from torchvision import transforms
import torch
import torch.optim as optim
from .utils import load_test_image


class TestEncoder(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(23)
        e = Encoder(50, 'inc', 'data/')
        img = torch.tensor([10])
        features = e(img)
        assert features.shape == torch.Size([1, 50])
        prev_param = []
        for i, param in enumerate(e.embedding.parameters()):
            prev_param.append(param.clone().detach())
        torch.sum(features).backward()
        optimizer = optim.SGD(e.parameters(), lr=0.001, momentum=0.9)
        optimizer.step()
        for i, param in enumerate(e.embedding.parameters()):
            assert torch.equal(param, prev_param[i])