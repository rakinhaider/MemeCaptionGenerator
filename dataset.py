class MemeDataset():
    def __init__(self, image_dir, caption_path):
        self.image_dir = image_dir
        self.caption_path = caption_path
        self.img_captions = []
        self.load_image_captions()

    def load_image_captions(self):
        self.img_captions = []
        with open(self.caption_path) as f:
            line = f.readline()
            splits = line.split('-')
            img_name, caption = splits[0], splits[1]
            self.img_captions.append((img_name, caption))

        