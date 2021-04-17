import os
import logging
logger = logging.getLogger(__name__)
import nltk
from PIL import Image
from torch.utils import data as data
import torch
from torchvision import transforms
import string


class MemeDataset(data.Dataset):
    def __init__(self, image_dir, caption_file, vocab, transform=None):
        self.image_dir = image_dir
        self.caption_file = os.path.join(image_dir, caption_file)
        self.vocab = vocab
        self.img_captions = []
        self.ids = []
        self.captions = []
        self.transform = transform
        self.max_len = -1

    def load_dataset(self, num_samples):
        with open(self.caption_file) as f:
            for line in f:
                splits = line.split(' - ')

                # TODO: Should have modified it when vocabulary was updated.
                #  Doing here anyway. Should be removed.
                punc_table = dict(
                    (ord(char), None) for char in string.punctuation)
                img_name = splits[0].replace('_', ' ')
                img_name = img_name.translate(punc_table)
                img_name = img_name.replace(' ', '-')
                img_name = img_name.replace('--', '-')

                caption = splits[1]
                self.ids.append(img_name)

                caption = nltk.word_tokenize(caption)

                caption = [self.vocab(word) for word in caption]
                caption = [self.vocab('<start>')] + caption
                caption += [self.vocab('<end>')]
                caption = torch.tensor(caption)

                self.captions.append(caption)
                if len(self.ids) == num_samples:
                    break

        self.max_len = max([len(c) for c in self.captions])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        caption = self.captions[index]

        image = Image.open(os.path.join(self.image_dir,
                                        'memes',
                                        id + '.jpg'))
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, caption


def collate_memes(data):
    # print(data)
    images = torch.stack([t[0] for t in data], dim=0)

    captions = [t[1] for t in data]
    lengths = torch.tensor([len(c) for c in captions])
    max_len = torch.max(lengths)

    cap_tensor = torch.zeros((len(data), max_len), dtype=torch.long)

    for i, c in enumerate(captions):
        cap_tensor[i][:len(c)] = c

    # logger.debug(images.shape)
    # logger.debug(cap_tensor)
    # logger.debug(captions)
    # logger.debug(lengths)

    return images, cap_tensor, lengths
