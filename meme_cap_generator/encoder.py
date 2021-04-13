import torch.nn as nn
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.conv_net = models.inception_v3()
        in_size = self.conv_net.fc.in_features
        self.conv_net.fc = nn.Identity(in_size, in_size)
        for param in self.conv_net.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(in_size, embed_size)

    def forward(self, image):
        with torch.no_grad():
            features = self.conv_net(image)
        return self.linear(features)


if __name__ == "__main__":
    e = Encoder(50)
