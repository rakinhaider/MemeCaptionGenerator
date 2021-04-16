import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, layers,
                 pre_trained_embed=None,
                 pre_trained_embed_file=None
                 ):
        super(Decoder, self).__init__()
        self.vocab = vocab
        if pre_trained_embed:
            self.load_pretrained_embedding(pre_trained_embed_file, vocab)
        else:
            self.embed_size = embed_size
            self.embedding = nn.Embedding(vocab.length, embed_size,
                                          padding_idx=vocab('<pad>'))
        self.lstm = nn.LSTM(embed_size, hidden_size, layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, img_features, captions, lengths):
        embeddings = self.embedding(captions)
        packed_embedding = pack_padded_sequence(embeddings,
                                                lengths, batch_first=True,
                                                enforce_sorted=False)
        img_features = img_features.unsqueeze(1)
        img_output, (h1, c1) = self.lstm(img_features)
        cap_output, (_, _) = self.lstm(packed_embedding, (h1, c1))
        # The softmax is not added. Because we will use CrossEntropyLoss
        # CrossEntropyLoss already includes LogSoftMax and NLLLoss
        output = self.linear(cap_output.data)
        return output

    def sample(self, feature, max_len):
        caption = []
        for i in range(max_len):
            hidden, state = self.lstm(feature, state)
            output = self.linear(feature)
            output = torch.softmax(output)
            prediction = torch.argmax(output)
            print(self.vocab[prediction.item()])
            caption.append(self.vocab[prediction.item()])
            if caption[-1] == '<end>':
                break
            feature = self.embedding(prediction)
        return caption
