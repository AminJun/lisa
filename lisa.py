import json
import os
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class LISA(VisionDataset):
    base_folder = 'lisa-batches'
    url = "https://github.com/AminJun/lisa/releases/download/v1/lisa.tar.gz"

    zipped = {
        'filename': 'lisa.tar.gz',
        'md5': 'd3e7bd49dc55c2d9240d4b5473848dcb',
    }

    label_file = 'labels.tensor'
    meta_file = 'meta.js'
    images_list = ['images_0.tensor', 'images_1.tensor', 'images_2.tensor']

    checksum = {
        'images_0.tensor': 'ac59f173c4d374859e73be64cee9de41',
        'images_1.tensor': '13df95c1f3b05fc9a90a83cb0febe50f',
        'images_2.tensor': '235f29c99e67019b1ba47dfe2492b461',
        label_file: 'a68f3549adbf898b26f1ab76ab515d38',
        meta_file: 'c52f0f118ff7e03c366608f7ea960d8f',
    }

    def _get_path(self, file: str) -> str:
        return os.path.join(self.root, self.base_folder, file)

    def __init__(self, root, train: bool, download=False, transform=None, target_transform=None):
        super(LISA, self).__init__(root=root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self.images = torch.cat([torch.load(self._get_path(file)) for file in self.images_list], 0)
        self.labels = torch.load(self._get_path(self.label_file))
        self._load_meta()

        self.train = train
        self._train_test_split()

    def _load_meta(self):
        with open(self._get_path(self.meta_file), 'r') as file:
            data = json.load(file)
            self.classes = data['classes']
            self.class_to_idx = data['name_to_label']

    def __getitem__(self, index) -> (torch.tensor, torch.tensor):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.images[index], self.labels[index]
        img = img if self.transform is None else self.transform(img)
        target = target if self.target_transform is None else self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.images)

    def _check_integrity(self) -> bool:
        return all(check_integrity(self._get_path(filename), md5) for filename, md5 in self.checksum.items())

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, **self.zipped)

    def extra_repr(self) -> str:
        return "No Split Yet"

    def _train_test_split(self, test_percent: float = 0.16):
        classes = {}
        for i, cl in enumerate(self.labels.numpy()):
            arr = classes.get(cl, [])
            arr.append(i)
            classes[cl] = arr

        train, test = [], []
        for cl, arr in classes.items():
            split_index = int(len(arr) * test_percent)
            test = test + arr[:split_index]
            train = train + arr[split_index:]

        sub = train if self.train else test
        self.images, self.labels = self.images[sub], self.labels[sub]
