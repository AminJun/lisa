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

    def _load_image(self, address: str, x1: int, y1: int, x2: int, y2: int) -> torch.tensor:
        image = Image.open(os.path.join(self.data_root, address))
        image = torchvision.transforms.functional.resized_crop(image, y1, x1, y2 - y1, x2 - x1, 32)
        return self.to_tensor(image)

    def __init__(self):
        table = pd.read_csv(self.csv_root, delimiter=';')
        self.to_tensor = torchvision.transforms.ToTensor()
        label = []

        for i, row in table.iterrows():
            file, label, x1, y1, x2, y2, *_ = row
            image = self._load_image(file, x1, y1, x2, y2)
            pdb.set_trace()
            print(row)


def main():
    dataset = SlowDataset()


if __name__ == '__main__':
    main()
