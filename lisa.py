import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import numpy as np
import pdb
import json
from tqdm import tqdm


class SlowDataset(Dataset):
    csv_root = 'allAnnotations.csv'
    data_root = 'data'

    def _load_image(self, address: str, x1: int, y1: int, x2: int, y2: int) -> Image:
        image = Image.open(os.path.join(self.data_root, address))
        return self.to_tensor(torchvision.transforms.functional.resized_crop(image, y1, x1, y2 - y1, x2 - x1, (32, 32)))

    def _read_images(self, table: pd.DataFrame) -> (list, list, list, dict):
        labels = []
        name_to_label = {}
        classes = []
        images = []
        for i, row in tqdm(table.iterrows()):
            file, label, x1, y1, x2, y2, *_ = row
            image = self._load_image(file, x1, y1, x2, y2)
            images.append(image)
            if label not in classes:
                name_to_label[label] = len(classes)
                classes.append(label)
            labels.append(name_to_label[label])
        return images, labels, classes, name_to_label

    def __init__(self):
        table = pd.read_csv(self.csv_root, delimiter=';')
        self.to_tensor = torchvision.transforms.ToTensor()
        images, labels, classes, name_to_label = self._read_images(table)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)
        self.meta = {'classes': classes, 'name_to_label': name_to_label}
        self.save(self.images, self.labels, self.meta)

    @staticmethod
    def save(images: torch.tensor, labels: torch.tensor, meta: dict):
        split = np.array_split(images, 3)
        for i, sub in enumerate(split):
            torch.save(sub, f'images_{i}.tensor')
        torch.save(labels, 'labels.tensor')
        with open('meta.js', 'w') as file:
            json.dump(meta, file)


def main():
    SlowDataset()


if __name__ == '__main__':
    main()
