import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
import pandas as pd
import pdb
import os
from tqdm import tqdm


class SlowDataset(Dataset):
    csv_root = 'allAnnotations.csv'
    data_root = 'data'

    def _load_image(self, address: str, x1: int, y1: int, x2: int, y2: int) -> Image:
        image = Image.open(os.path.join(self.data_root, address))
        return torchvision.transforms.functional.resized_crop(image, y1, x1, y2 - y1, x2 - x1, 32)

    def _read_images(self, table: pd.DataFrame) -> (list, list):
        labels = []
        images = []
        for i, row in tqdm(table.iterrows()):
            file, label, x1, y1, x2, y2, *_ = row
            image = self._load_image(file, x1, y1, x2, y2)
            images.append(image)
            labels.append(label)
        return images, labels

    def save_images(self, images: list, labels: list):
        for i, (image, label) in tqdm(enumerate(zip(images, labels))):
            image.save(os.path.join('desktop', f'{i}_{label}'))

    def __init__(self):
        table = pd.read_csv(self.csv_root, delimiter=';')
        self.to_tensor = torchvision.transforms.ToTensor()
        self.labels, self.images = self._read_images(table)
        self.save_images(self.images, self.labels)
        pdb.set_trace()


def main():
    SlowDataset()


if __name__ == '__main__':
    main()
