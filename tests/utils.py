from PIL import Image
from torchvision import transforms


def load_test_image():
    img = Image.open('../data/memes/y-u-no.jpg').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = img.unsqueeze(0)

    return img