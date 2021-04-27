import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import logging
import pickle
import os

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, layers,
                 pre_trained_embed=None,
                 data_dir=None
                 ):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab.length, embed_size,
                                      padding_idx=vocab('<pad>'))
        if pre_trained_embed:
            self.load_pretrained_embedding(pre_trained_embed, data_dir)
            self.embedding.requires_grad_(False)
        self.lstm = nn.LSTM(embed_size, hidden_size, layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

    def load_pretrained_embedding(self, embed, data_dir):
        if embed == 'g':
            file_name = os.path.join(
                data_dir, 'glove', 'processed',
                'glove.6B.' + str(self.embed_size) + '_dat.pkl'
            )
            with open(file_name, 'rb') as f:
                glove = pickle.load(f)
            file_name = os.path.join(
                data_dir, 'glove', 'processed',
                'glove.6B.' + str(self.embed_size) + '_w2i.pkl'
            )
            with open(file_name, 'rb') as f:
                glove_w2i = pickle.load(f)
            embedding = torch.zeros(len(self.vocab), self.embed_size)
            for i in range(len(self.vocab)):
                w = self.vocab.idx2word[i]
                if w in glove_w2i:
                    glove_index = glove_w2i[w]
                    embedding[i][:] = glove[glove_index][:]
                else:
                    embedding[i][:] = torch.rand(1, self.embed_size)[0]
                if w == 'the':
                    print(embedding[i])
            assert embedding.shape == torch.Size([self.vocab.length,
                                                  self.embed_size])
        elif embed == 't':
            pass
        else:
            raise ValueError('Pretrained embedding can only be g or t.')
        self.embedding.load_state_dict({'weight': embedding})

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
        state = None
        logging.debug(feature.shape)
        hidden, state = self.lstm(feature, state)
        feature = self.embedding(torch.tensor([[self.vocab('<start>')]]))

        for i in range(max_len):
            hidden, state = self.lstm(feature, state)
            output = self.linear(hidden)
            output = torch.softmax(output, dim=2)
            prediction = output.argmax()
            caption.append(prediction.item())
            if caption[-1] == self.vocab('<end>'):
                break
            feature = self.embedding(prediction).unsqueeze(0).unsqueeze(0)
        return caption
