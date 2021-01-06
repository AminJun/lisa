import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
import pandas as pd
import pdb
import os


class SlowDataset(Dataset):
    csv_root = 'allAnnotations.csv'
    data_root = 'data'

    def _load_image(self, address: str, x1: int, y1: int, x2: int, y2: int) -> Image:
        image = Image.open(os.path.join(self.data_root, address))
        return torchvision.transforms.functional.resized_crop(image, y1, x1, y2 - y1, x2 - x1, 32)

    def __init__(self):
        table = pd.read_csv(self.csv_root, delimiter=';')
        self.to_tensor = torchvision.transforms.ToTensor()
        labels = []
        one_from_each_label = {}
        images = []

        for i, row in table.iterrows():
            file, label, x1, y1, x2, y2, *_ = row
            image = self._load_image(file, x1, y1, x2, y2)
            images.append(image)
            labels.append(label)
            if label not in one_from_each_label.keys():
                one_from_each_label[label] = i

        for label, i in one_from_each_label:
            image: Image = images[i]
            image.save(os.path.join('examples', f'{label}.png'))


def main():
    dataset = SlowDataset()


if __name__ == '__main__':
    main()
