import torch.nn as nn
import torchvision.models as models
import torch


class Encoder(nn.Module):
    def __init__(self, embed_size, encoder_type):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        if encoder_type == 'inc':
            self.conv_net = models.inception_v3(pretrained=True)
            in_size = self.conv_net.fc.in_features
            self.conv_net.fc = nn.Identity(in_size, in_size)
        elif encoder_type == 'res':
            self.conv_net = models.resnet34(pretrained=True)
            # TODO: fix the last layer.
        for param in self.conv_net.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(in_size, embed_size)

    def forward(self, image):
        with torch.no_grad():
            self.conv_net.eval()
            features = self.conv_net(image)
        return self.linear(features)


if __name__ == "__main__":
    e = Encoder(50)
