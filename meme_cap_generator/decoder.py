import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, layers,
                 pre_trained_embed=None,
                 pre_trained_embed_file=None
                 ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(embed_size, len(vocab))
        self.lstm = nn.LSTM(embed_size, hidden_size, layers)

        if pre_trained_embed is not None:
            self.load_pretrained_embedding(pre_trained_embed_file, vocab)

    def forward(self, img_features, captions):
        pass