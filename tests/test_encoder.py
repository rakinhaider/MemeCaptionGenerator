import unittest
from meme_cap_generator.encoder import Encoder
from PIL import Image
from torchvision import transforms
import torch
import torch.optim as optim


class TestEncoder(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(23)
        e = Encoder(50)
        e.eval()
        img = Image.open('../data/memes/y-u-no.jpg').convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        print(img.shape)
        features = e(img)
        print(features.shape)
        prev_param = []
        for i, param in enumerate(e.conv_net.parameters()):
            prev_param.append(param.clone().detach())
        torch.sum(features).backward()
        optimizer = optim.SGD(e.parameters(), lr=0.001, momentum=0.9)
        optimizer.step()
        for i, param in enumerate(e.conv_net.parameters()):
            assert torch.equal(param, prev_param[i])