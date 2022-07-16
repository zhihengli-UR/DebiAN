import torch
import os


from PIL import Image
from torch.utils.data.dataset import Dataset
from glob import glob


class bFFHQDataset(Dataset):
    base_folder = 'bffhq'
    target_attr_index = 0
    bias_attr_index = 1

    def __init__(self, root, split, transform=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        root = os.path.join(root, self.base_folder)

        self.root = root

        if split == 'train':
            self.align = glob(os.path.join(root, split, 'align', "*", "*"))
            self.conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root, split, "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, split, "*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = self.data[index]
        first_attr = int(fpath.split('_')[-2])
        second_attr = int(fpath.split('_')[-1].split('.')[0])
        attr = torch.LongTensor([first_attr, second_attr])
        image = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr
