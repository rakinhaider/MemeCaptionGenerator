import unittest
import torch
from main import Main
import os


class TestTrainedModels(unittest.TestCase):
    def test_trained_decoder(self):
        torch.manual_seed(23)
        m = Main()
        print(m.data_dir)
        m.data_dir = '../data/'
        m.vocab_file = 'vocab_2_CaptionsClean_nopunc_t.pkl'
        m.cap_file = 'CaptionsClean_nopunc_-1_t_s.txt'
        m.pretrained_embed = 'g'
        m.num_samples = 1000

        m.title = m.generate_title()
        print(m.title)
        m.transforms = m.get_transfroms()

        m.vocab = m.get_vocabulary()
        m.prepare_models()

        m.dataset = m.get_dataset()
        m.dataset.load_dataset(m.num_samples)
        m.loader = m.get_loader()
        m.prepare_models()
        path = os.path.join('../logs', m.title, 'encoder_best.pt')
        m.encoder.load_state_dict(torch.load(path))
        path = os.path.join('../logs', m.title, 'decoder_best.pt')
        m.decoder.load_state_dict(torch.load(path))
        prefix = [m.vocab(w) for w in m.vocab.word2idx]
        index = m.vocab('<say>')

        state = (torch.rand((3, 1, 50)), torch.rand((3, 1, 50)))
        m.prepare_trainer()
        print(m.train_minibatch())
        embedding = m.decoder.embedding(torch.tensor([index]))
        # print(embedding)
        hidden, state2 = m.decoder.lstm(embedding.unsqueeze(0), state)
        # print(hidden.shape)
        linear = m.decoder.linear(hidden)
        # print(linear.shape)
        output = torch.softmax(linear, dim=2)
        topk = torch.topk(output, k=10, dim=2)
        print(topk.indices)
        print([m.vocab.idx2word[index.item()] for index in topk.indices[0][0]])
        prediction = output.argmax()
        print(prediction, m.vocab.idx2word[prediction.item()])

        embedding = m.decoder.embedding(topk.indices)
        # print(embedding)
        hidden, state2 = m.decoder.lstm(embedding.unsqueeze(0), state)
        # print(hidden.shape)
        linear = m.decoder.linear(hidden)
        # print(linear.shape)
        output = torch.softmax(linear, dim=2)
        topk = torch.topk(output, k=10, dim=2)
        print(topk.indices)
        print([m.vocab.idx2word[index.item()] for index in topk.indices[0][0]])
        prediction = output.argmax()
        print(prediction, m.vocab.idx2word[prediction.item()])
