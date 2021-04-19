import os
import torch.nn as nn
import torchvision.models as models
import torch


class Encoder(nn.Module):
    def __init__(self, embed_size, encoder_type, data_dir):
        super(Encoder, self).__init__()
        self.embed_size = embed_size

        embedding_file = os.path.join(data_dir, encoder_type + '.pkl')
        embedding = torch.load(embedding_file)
        self.embedding = nn.Embedding(embedding.shape[0],
                                      embedding.shape[1])
        self.embedding.load_state_dict({'weight': embedding})
        self.embedding.eval()

        in_size = embedding.shape[1]
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(in_size, embed_size)

    def forward(self, image):
        with torch.no_grad():
            features = self.embedding(image)
        return self.linear(features)


if __name__ == "__main__":
    e = Encoder(50)
