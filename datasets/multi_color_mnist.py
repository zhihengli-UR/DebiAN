import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MultiColorMNIST(Dataset):
    attribute_names = ["digit", "LColor", 'RColor']
    basename = 'multi_color_mnist'
    target_attr_index = 0
    left_color_bias_attr_index = 1
    right_color_bias_attr_index = 2

    def __init__(self, root, split, left_color_skew, right_color_skew, severity, transform=ToTensor()):
        super().__init__()

        assert split in ['train', 'valid']
        assert left_color_skew in [0.005, 0.01, 0.02, 0.05]
        assert right_color_skew in [0.005, 0.01, 0.02, 0.05]
        assert severity in [1, 2, 3, 4]

        root = os.path.join(root, self.basename, f'ColoredMNIST-SkewedA{left_color_skew}-SkewedB{right_color_skew}-Severity{severity}')
        assert os.path.exists(root), f'{root} does not exist'

        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)

        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        self.transform = transform

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, idx):
        image, attr = self.data[idx], self.attr[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, attr
