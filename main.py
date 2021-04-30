import os
from meme_cap_generator import MemeDataset, Vocabulary, GloveVocabulary
from meme_cap_generator.encoder import Encoder
from meme_cap_generator.decoder import Decoder
from meme_cap_generator.meme_dataset import collate_memes
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy
from PIL import Image


class Main(object):
    def __init__(self):
        print('1')
        self.console = argparse.ArgumentParser()
        self.add_arguments()
        self.args = self.console.parse_args()
        self.train = self.args.train
        self.sample = self.args.sample
        self.gen = self.args.gen

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

        self.encoder_type = self.args.encoder_type
        self.embed_size = self.args.embed_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.lstm_layers = self.args.lstm_layers
        self.pretrained_embed = self.args.pretrained_embed

        self.vthresh = int(self.vocab_file.split('_')[1])

        self.device = self.args.device
        self.debug = self.args.debug

        self.rng_cpu = torch.Generator("cpu")
        self.rng_gpu = torch.Generator(self.device)
        self.rng_cpu.manual_seed(self.random_seed)
        self.rng_gpu.manual_seed(self.random_seed)

    def main(self):
        print(2)
        self.title = self.generate_title()
        os.makedirs(
            os.path.join("logs", self.title),
            exist_ok=True,
        )

        if self.sbatch:
            self.sbatch_submit()
            exit()

        print('3')
        self.transforms = self.get_transfroms()
        if self.gen:
            self.generate_image_embeddings()
            return

        print('4')
        self.vocab = self.get_vocabulary()
        print('5')
        self.prepare_models()
        print('6')
        self.dataset = self.get_dataset()
        if self.train:
            print('num_samples', self.num_samples)
            self.dataset.load_dataset(self.num_samples)
            print('Dataset contains {:d} items.'.format(
                len(self.dataset)), flush=True)

            print('7')
            self.loader = self.get_loader()

            self.fit()
        elif self.sample:
            self.dataset.load_image_ids()
            self.sample_caption()

    def add_arguments(self):
        parser = self.console
        group = parser.add_mutually_exclusive_group()
        group.add_argument('-t', '--train', action='store_true',
                           help='Train the model')
        group.add_argument('-s', '--sample', action='store_true',
                           help='Sample captions the model')
        group.add_argument('-g', '--gen', action='store_true',
                           help='Generate image embeddings')

        parser.add_argument('--num-workers', type=int,
                            help='Number of workers in GPU')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Debug level')
        # Dataset Arguments
        parser.add_argument('--data-dir', help='Data Directory',
                            default='data/')
        parser.add_argument('-cf', '--cap-file', help='Caption file')

        # Vocabulary arguments
        parser.add_argument('--vocab-file', help='Vocabulary File')
        parser.add_argument('--vthresh', default=2,
                            help='Vocabulary word discard threshold')
        parser.add_argument('--pretrained-embed', choices=['g', 't'],
                           help='Use pretrained embedding', default=None)

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
        parser.add_argument('--embed-file', help='Use pretrained embedding')

        # Sample Parameters
        parser.add_argument('--sample-images', nargs='+', default=[],
                            help='Images to sample')
        parser.add_argument('--max-len', type=int, default=20,
                            help='Maximum length of generated caption')
        # Model Parameters
        parser.add_argument('--encoder-type', help='Encoder Type',
                            default='inc', choices=['inc', 'res'])
        parser.add_argument('--embed-size', help='Embedding Size',
                            type=int, default=50)
        parser.add_argument('--batch-size', help='Batch size',
                            type=int, default=32)
        parser.add_argument('--hidden-size', help='Hidden layer size',
                            type=int, default=50)
        parser.add_argument('--lstm-layers', help='LSTM layers',
                            type=int, default=3)

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
        sbatch_lines.append("#SBATCH --time=240:00")

        # Memory limit
        sbatch_lines.append("#SBATCH --mem-per-cpu=8G")

        # Set up notifications
        # send email when job begins
        sbatch_lines.append('#SBATCH --mail-type=begin')
        # send email when job ends
        sbatch_lines.append('#SBATCH --mail-type=end')
        # send email if job fails
        sbatch_lines.append('#SBATCH --mail-type=fail')
        sbatch_lines.append('#SBATCH --mail-user=chaider@purdue.edu')

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
        elif self.gen:
            sbatch_lines.append(
                "   -g --encoder-type {:s}\\".format(self.encoder_type),
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
            "   --device {:s} --random-seed {:d}\\".format(
                self.device, self.random_seed
            )
        )

        if not self.gen:

            vocab_file_name = "vocab_{}_CaptionsClean_nopunc_t.pkl"
            vocab_file_name = vocab_file_name.format(self.vthresh)
            sbatch_lines.append(
                "   --vocab-file {} \\".format(vocab_file_name)
            )
            if self.pretrained_embed:
                sbatch_lines.append(
                    "   --pretrained-embed g \\"
                )
            sbatch_lines.append(
                "   -e {:d}\\".format(self.num_epochs)
            )
            sbatch_lines.append(
                "   "\
                "--embed-size {:d} --batch-size {:d} "
                "--lstm-layers {:d} \\".format(
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
        dataset = MemeDataset(self.data_dir, self.cap_file,
                              self.vocab, self.transforms['train'])
        return dataset

    def get_loader(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=8,
                            drop_last=True, collate_fn=collate_memes,
                            pin_memory=True)

        return loader

    def get_vocabulary(self):
        """
        if self.pretrained_embed:
            vocab = GloveVocabulary(dim=self.embed_size)
            vocab.load_vocab('glove.6B', self.data_dir)
        else:
            vocab = Vocabulary()
            vocab.load_vocab(self.vocab_file, self.data_dir)
        """
        vocab = Vocabulary()
        vocab.load_vocab(self.vocab_file, self.data_dir)
        print('Vocabulary contains {} words'.format(vocab.length))
        return vocab

    def prepare_models(self):
        self.encoder = Encoder(self.embed_size, self.encoder_type,
                               self.data_dir)
        self.decoder = Decoder(self.embed_size, self.hidden_size,
                               self.vocab, self.lstm_layers,
                               pre_trained_embed=self.pretrained_embed,
                               data_dir=self.data_dir)

    def generate_title(self):
        if self.gen:
            title = self.encoder_type
        # TODO: add v_thresh as commandline argument.
        else:
            title = 'MCG_{:s}_{:d}_{:d}_{:d}_{:d}_{:s}'.format(
                self.encoder_type, self.embed_size,
                self.hidden_size, self.lstm_layers,
                self.vthresh,
                self.pretrained_embed if self.pretrained_embed else '0')
            print(title)
        return title

    def generate_image_embeddings(self):
        model = models.inception_v3(pretrained=True)
        in_size = model.fc.in_features
        model.fc = torch.nn.Identity(in_size, in_size)

        image_dir = os.path.join(self.data_dir, 'memes')
        image_list = os.listdir(image_dir)
        img_embeddings = torch.empty(len(image_list), 2048)

        model.eval()

        with torch.no_grad():
            for i, img in enumerate(image_list):
                print(i, img, flush=True)
                img = Image.open(os.path.join(image_dir, img)).convert('RGB')
                img = self.transforms['train'](img)
                img = img.unsqueeze(0)
                embed = model(img)
                del img
                # print(embed)
                # print(embed.shape, flush=True)
                img_embeddings[i][:] = embed[0][:]
                del embed

        torch.save(img_embeddings,
            os.path.join(self.data_dir, self.encoder_type + '.pkl')
        )

    def prepare_trainer(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        params = list(self.decoder.parameters())
        params += list(self.encoder.linear.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        self.step = 0

    def train_minibatch(self):
        print('minibatch start')
        batch_loss = 0
        i = 0
        for images, captions, lengths in self.loader:
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            images.to(self.device)
            captions.to(self.device)
            lengths.to(self.device)
            lengths = lengths - 1
            targets = pack_padded_sequence(captions[:, 1:],
                                           lengths,
                                           batch_first=True,
                                           enforce_sorted=False)
            captions = captions[:, :-1]
            features = self.encoder(images)
            prediction = self.decoder(features, captions, lengths)
            loss = self.criterion(prediction, targets.data)
            loss.backward()
            self.optimizer.step()
            self.step += 1
            batch_loss += loss.detach()
            print('Step {}/{} of mini-batch, Loss {}'.format(
                self.step, len(self.loader) * (self.num_epochs + 1), loss
            ), flush=True)

        return batch_loss/len(self.loader)

    def fit(self):
        print('Training started.')
        self.prepare_trainer()

        print('0 minibatch')
        loss = self.train_minibatch()
        best_loss = loss
        best_model_dict = {'encoder': deepcopy(self.encoder.state_dict()),
                           'decoder': deepcopy(self.decoder.state_dict())}

        for epoch in range(self.num_epochs):
            loss = self.train_minibatch()
            print('Epoch {}/{} Loss {}'.format(
                epoch, self.num_epochs, loss
            ))
            if loss < best_loss:
                best_loss = loss
                best_model_dict = {
                    'encoder': deepcopy(self.encoder.state_dict()),
                    'decoder': deepcopy(self.decoder.state_dict())
                }
                for model in ['encoder', 'decoder']:
                    torch.save(
                        {key: val.cpu() for key, val in
                         best_model_dict[model].items()},
                        os.path.join('logs', self.title, model + '_best.pt')
                    )
            # break

    def sample_caption(self):
        path = os.path.join('logs', self.title, 'encoder_best.pt')
        self.encoder.load_state_dict(torch.load(path))
        path = os.path.join('logs', self.title, 'decoder_best.pt')
        self.decoder.load_state_dict(torch.load(path))

        for img_file in self.sample_images:
            print(img_file)
            name, ext = os.path.splitext(img_file)
            index = torch.tensor([self.dataset.id2index[name]])
            feature = self.encoder(index)
            feature = feature.unsqueeze(1)
            caption = self.decoder.sample(feature, self.max_len)
            print(caption)
            caption = [self.vocab.idx2word[i] for i in caption]
            print(caption)


if __name__ == "__main__":
    Main().main()