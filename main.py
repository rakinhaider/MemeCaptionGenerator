import os
import logging
logging.basicConfig(level=logging.DEBUG)
from meme_cap_generator import MemeDataset, Vocabulary
from meme_cap_generator.encoder import Encoder
from meme_cap_generator.decoder import Decoder
from meme_cap_generator.meme_dataset import collate_memes
import argparse
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy
from PIL import Image

class Main(object):
    def __init__(self):
        self.console = argparse.ArgumentParser()
        self.add_arguments()
        self.args = self.console.parse_args()
        self.train = self.args.train
        self.sample = self.args.sample
        self.num_workers = self.args.num_workers
        self.data_dir = self.args.data_dir
        self.cap_file = self.args.cap_file
        self.vocab_file = self.args.vocab_file
        self.embed_file = self.args.embed_file

        self.sbatch = self.args.sbatch
        self.num_epochs = self.args.num_epochs
        self.random_seed = self.args.random_seed
        self.num_samples = self.args.num_samples
        self.learning_rate = self.args.learning_rate

        self.sample_images = self.args.sample_images
        self.max_len = self.args.max_len

        self.embed_size = self.args.embed_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.lstm_layers = self.args.lstm_layers
        self.pretrain_embed = self.args.pretrained_embed

        self.device = self.args.device
        self.debug = self.args.debug

        self.rng_cpu = torch.Generator("cpu")
        self.rng_gpu = torch.Generator(self.device)
        self.rng_cpu.manual_seed(self.random_seed)
        self.rng_gpu.manual_seed(self.random_seed)

        self.add_logging()

        self.title = self.generate_title()
        os.makedirs(
            os.path.join("logs", self.title),
            exist_ok=True,
        )

        if self.sbatch:
            self.sbatch_submit()
            exit()

        self.vocab = self.get_vocabulary()
        self.transforms = self.get_transfroms()
        self.prepare_models()
        if self.train:
            self.dataset = self.get_dataset()
            self.loader = self.get_loader()

            self.fit()
        elif self.sample:
            self.sample_caption()

    def add_logging(self):
        # Wanted to do something else.
        # But logger doesn't work in submodules when logger is configured
        # before import other modules.
        self.logger = logging

    def add_arguments(self):
        parser = self.console
        group = parser.add_mutually_exclusive_group()
        group.add_argument('-t', '--train', action='store_true',
                           help='Train the model')
        group.add_argument('-s', '--sample', action='store_true',
                           help='Sample captions the model')

        parser.add_argument('--num-workers', type=int,
                            help='Number of workers in GPU')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Debug level')
        # Dataset Arguments
        parser.add_argument('--data-dir', help='Data Directory',
                            default='data/')
        parser.add_argument('-cf', '--cap-file', help='Caption file')

        parser.add_argument('--vocab-file', help='Vocabulary File')
        parser.add_argument('--embed-file', help='Use pretrained embedding')

        # Train Paramters
        parser.add_argument('--sbatch', help='Run using sbatch',
                            action='store_true')
        parser.add_argument('-e', '--num-epochs', help='Number of epochs',
                            type=int)
        parser.add_argument('--device', help='Device for training',
                            choices=['cuda', 'cpu'])
        parser.add_argument('--random-seed', help='Random Seed',
                            type=int, default=23)
        parser.add_argument('--num-samples', help='Number of training samples.',
                            type=int, default=-1)
        parser.add_argument('-lr', '--learning-rate', help='Learning Rate',
                            type=float, default=10**(-4))

        # Sample Parameters
        parser.add_argument('--sample-images', nargs='+', default=[],
                            help='Images to sample')
        parser.add_argument('--max-len', type=int, default=20,
                            help='Maximum length of generated caption')
        # Model Parameters
        parser.add_argument('--embed-size', help='Embedding Size', type=int)
        parser.add_argument('--batch-size', help='Batch size', type=int)
        parser.add_argument('--hidden-size', help='Hidden layer size', type=int)
        parser.add_argument('--lstm-layers', help='LSTM layers', type=int)

        parser.add_argument('--pretrained-embed', choices=['g', 't'],
                            default=None,
                            help='Use pretrained embedding')

    def sbatch_submit(self) -> None:
        r"""
        Submit by `sbatch`.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        # Initialize sbatch submission file.
        sbatch_lines = ["#!/bin/bash"]

        # Hardware resources.
        if (self.device == "cuda"):
            sbatch_lines.append("#SBATCH -A gpu")
            sbatch_lines.append("#SBATCH --gres=gpu:1")
        elif (self.device == "cpu"):
            sbatch_lines.append("#SBATCH -A scholar")
        else:
            print(
                "[\033[91mError\033[0m]: Unknown device \"{:s}\".".format(
                    self.device
                ),
            )
            raise RuntimeError
        sbatch_lines.append(
            "#SBATCH --cpus-per-task={:d}".format(self.num_workers + 1),
        )
        sbatch_lines.append("#SBATCH --nodes=1")

        # Time limit.
        sbatch_lines.append("#SBATCH --job-name {:s}".format(self.title))
        sbatch_lines.append("#SBATCH --time=60:00")

        # IO redirection.
        sbatch_lines.append(
            "#SBATCH --output {:s}".format(
                os.path.join("logs", self.title, "output"),
            ),
        )
        sbatch_lines.append(
            "#SBATCH --error {:s}".format(
                os.path.join("logs", self.title, "error"),
            ),
        )

        # Python script.
        sbatch_lines.append(
            "python main.py \\",
        )
        if self.train:
            sbatch_lines.append(
                "   -t\\"
            )
        elif self.sample:
            sbatch_lines.append(
                "   -s\\",
            )
        sbatch_lines.append(
            "   --num-workers {:d}\\".format(self.num_workers)
        )
        sbatch_lines.append(
            "   --data-dir {:s} -cf {:s} \\".format(
                self.data_dir, self.cap_file
            )
        )
        sbatch_lines.append(
            "   --vocab-file vocab_2_CaptionsClean_nopunc_t.pkl \\"
        )
        sbatch_lines.append(
            "   -e {:d} --device {:s} --random-seed {:d}\\".format(
                self.num_epochs, self.device, self.random_seed
            )
        )
        sbatch_lines.append(
            "   "\
            "--embed-size {:d} --batch-size {:d} --lstm-layers 3 \\".format(
                self.embed_size, self.batch_size, self.lstm_layers
            )
        )
        sbatch_lines.append(
            "   "\
            "--num-samples {:d} {:s} --hidden-size {:d}\\".format(
                self.num_samples,
                '--debug' if self.debug else '',
                self.hidden_size
            )
        )

        sbatch_lines.append(
            "   --learning-rate {:f}\\".format(self.learning_rate)
        )


        # Save to file.
        path = os.path.join("logs", self.title, "submit.sb")
        with open(path, "w") as file:
            file.write("\n".join(sbatch_lines) + "\n")

        # Run the command.
        print("[\033[31msbatch\033[0m] {:s}".format(path))
        os.system("sbatch {:s}".format(path))


    def get_transfroms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([transforms.Resize(229),
                                        transforms.CenterCrop(229),
                                        transforms.ToTensor(), normalize
                                        ])
        return {'train': transform, 'test': transform}

    def get_dataset(self):
        self.logger.info('Dataset Read')
        dataset = MemeDataset(self.data_dir, self.cap_file,
                              self.vocab, self.transforms['train'])
        dataset.load_dataset(self.num_samples)
        return dataset

    def get_loader(self):
        sampler = torch.utils.data.RandomSampler(
            self.dataset,
            replacement=False, generator=self.rng_cpu,
        )
        loader = DataLoader(self.dataset, batch_size=5, sampler=sampler,
                            num_workers=1, drop_last=True,
                            collate_fn=collate_memes)

        return loader

    def get_vocabulary(self):
        vocab = Vocabulary()
        vocab.load_vocab(self.vocab_file, self.data_dir)
        self.logger.info('Vocabulary contains {} words'.format(vocab.length))
        return vocab

    def prepare_models(self):
        self.encoder = Encoder(self.embed_size)
        self.decoder = Decoder(self.embed_size, self.hidden_size,
                               self.vocab, self.lstm_layers,
                               pre_trained_embed=self.pretrain_embed,
                               pre_trained_embed_file=self.embed_file
                               )

    def generate_title(self):
        # TODO: add v_thresh as commandline argument.
        self.v_thresh = 2
        self.encoder_type = 'inc'
        title = 'MCG_{:s}_{:d}_{:d}_{:d}_{:d}_{:s}'.format(
            self.encoder_type, self.embed_size,
            self.hidden_size, self.lstm_layers,
            self.v_thresh,
            self.pretrain_embed if self.pretrain_embed else '0')
        self.logger.info(title)
        return title

    def train_minibatch(self):
        batch_loss = 0
        for i, (images, captions, lengths) in enumerate(self.loader):
            images.to(self.device)
            captions.to(self.device)
            lengths.to(self.device)
            features = self.encoder(images)
            prediction = self.decoder(features, captions, lengths)
            targets = pack_padded_sequence(captions,
                                           lengths,
                                           batch_first=True,
                                           enforce_sorted=False)
            loss = self.criterion(prediction, targets.data)
            batch_loss += loss
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            loss.backward()
            self.optimizer.step()

        return batch_loss/len(self.loader)

    def fit(self):

        self.criterion = torch.nn.CrossEntropyLoss()
        params = list(self.decoder.parameters())
        params += list(self.encoder.linear.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        loss = self.train_minibatch()
        best_loss = loss
        best_model_dict = {'encoder': deepcopy(self.encoder.state_dict()),
                           'decoder': deepcopy(self.decoder.state_dict())}

        for epoch in range(self.num_epochs):
            loss = self.train_minibatch()
            logging.info('Epoch {}/{} Loss {}'.format(
                epoch, self.num_epochs, loss
            ))
            if loss < best_loss:
                best_loss = loss
                best_model_dict = {
                    'encoder': deepcopy(self.encoder.state_dict()),
                    'decoder': deepcopy(self.decoder.state_dict())
                }
            # break

        for model in ['encoder', 'decoder']:
            torch.save(
                {key: val.cpu() for key, val in best_model_dict[model].items()},
                os.path.join('logs', self.title, model + '_best.pt')
            )

    def sample_caption(self):
        path = os.path.join('logs', self.title, 'encoder_best.pt')
        self.encoder.load_state_dict(torch.load(path))
        path = os.path.join('logs', self.title, 'decoder_best.pt')
        self.decoder.load_state_dict(torch.load(path))

        for img_file in self.sample_images:
            self.logger.info(img_file)
            img_file = os.path.join(self.data_dir, 'memes/', img_file)
            img = Image.open(img_file).convert('RGB')
            img.show()
            img = self.transforms['test'](img)
            img = img.unsqueeze(0)
            feature = self.encoder(img)
            feature = feature.unsqueeze(1)
            caption = self.decoder.sample(feature, self.max_len)
            caption = [self.vocab.idx2word[i] for i in caption]
            logging.info(caption)


if __name__ == "__main__":
    Main()
