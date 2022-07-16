import os
import PIL
import json

from torchvision.datasets.vision import VisionDataset


class BAR(VisionDataset):
    base_folder = "bar"
    class_list = ['climbing', 'diving', 'fishing', 'racing', 'throwing', 'pole vaulting']

    def __init__(self, root, split="train", transform=None):
        super(BAR, self).__init__(root, transform=transform)
        assert split in ['train', 'test']
        self.split = split

        cls_name_to_index = {cls_name: i for i, cls_name in enumerate(self.class_list)}

        with open(os.path.join(self.root, self.base_folder, 'metadata.json')) as f:
            metadata = json.load(f)

        self.fname_lst = []
        self.cls_lst = []

        for fname, data in metadata.items():
            if split == 'train' and not data['train']:
                continue
            if split == 'test' and data['train']:
                continue

            self.fname_lst.append(fname + '.jpg')
            self.cls_lst.append(cls_name_to_index[data['cls']])

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.base_folder, self.split, self.fname_lst[index])
        X = PIL.Image.open(fpath).convert('RGB')
        target = self.cls_lst[index]
        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self):
        return len(self.fname_lst)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
