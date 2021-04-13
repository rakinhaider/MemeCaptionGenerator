import os
import nltk
from PIL import Image
from torch.utils import data as data
import torch
from torchvision import transforms


class MemeDataset(data.Dataset):
    def __init__(self, image_dir, caption_file, vocab, transform=None):
        self.image_dir = image_dir
        self.caption_file = os.path.join(image_dir, caption_file)
        self.vocab = vocab
        self.img_captions = []
        self.ids = []
        self.captions = []
        self.load_dataset()
        self.transform = transform

    def load_dataset(self):
        with open(self.caption_file) as f:
            for line in f:
                splits = line.split(' - ')
                img_name = splits[0].replace(' ', '-')
                caption = splits[1]
                self.ids.append(img_name)

                caption = nltk.word_tokenize(caption)

                caption = [self.vocab(word) for word in caption]
                caption = [self.vocab('<start>')] + caption
                caption += [self.vocab('<end>')]
                caption = torch.tensor(caption)

                self.captions.append(caption)

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
