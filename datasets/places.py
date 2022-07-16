# adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/places365.py

import os
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor


scene_categories_lst = ['bedroom', 'bridge', 'church', 'classroom', 'conference_room',
                        'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']


class Places365(VisionDataset):
    r"""`Places365 <http://places2.csail.mit.edu/index.html>`_ classification dataset.

    Args:
        root (string): Root directory of the Places365 dataset.
        split (string, optional): The dataset split. Can be one of ``train-standard`` (default), ``train-challendge``,
            ``val``.
        small (bool, optional): If ``True``, uses the small images, i. e. resized to 256 x 256 pixels, instead of the
            high resolution ones.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    Raises:
        RuntimeError: If ``download is False`` and the meta files, i. e. the devkit, are not present or corrupted.
        RuntimeError: If ``download is True`` and the image archive is already extracted.
    """
    _SPLITS = ("train", "val")
    # (file, md5)
    _CATEGORIES_META = ("categories_places365.txt", "06c963b85866bd0649f97cb43dd16673")
    def __init__(
        self,
        root: str,
        categories: set = ('bedroom', 'bridge', 'church', 'classroom', 'conference_room',
                            'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'),
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        return_fpath: bool = False,
        return_original_img: bool = False
    ) -> None:
        super().__init__(root=os.path.join(root, 'places365'), transform=transform, target_transform=target_transform)

        assert split in self._SPLITS
        self.split = split

        assert len(categories) > 0
        self.categories_set = set(categories)
        self.categories_lst = list(categories)

        self.loader = loader

        class_to_idx = self.load_categories()
        self.places_idx_to_lsun_idx = {}
        self.lsun_categories_places_idx = []
        for i, c in enumerate(categories):
            if c != 'church':
                places_class_name = f'/{c[0]}/{c}'
            else:
                places_class_name = f'/{c[0]}/{c}/outdoor'
            places_idx = class_to_idx[places_class_name]
            self.places_idx_to_lsun_idx[places_idx] = i
            self.lsun_categories_places_idx.append(places_idx)
        self.imgs, self.targets = self.load_file_list()
        self.return_fpath = return_fpath
        self.return_original_img = return_original_img
        self.pil_to_tensor_func = ToTensor()

    _FILE_LIST_META = {
        "train": ("places365_train_standard.txt", "30f37515461640559006b8329efbed1a"),
        "val": ("places365_val.txt", "e9f2fd57bfd9d07630173f4e8708e4b1"),
    }

    def __getitem__(self, index: int):
        file, target = self.imgs[index]
        if self.return_fpath:
            return os.path.abspath(file), target

        image = self.loader(file)
        original_img = image

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.return_original_img:
            return image, target, self.pil_to_tensor_func(original_img)

        return image, target

    def __len__(self) -> int:
        return len(self.imgs)

    @property
    def variant(self) -> str:
        return "challenge" if "challenge" in self.split else "standard"

    @property
    def images_dir(self) -> str:
        if self.split == 'train':
            dir_name = f"data_256"
        else:
            dir_name = f"{self.split}_256"
        return path.join(self.root, dir_name)

    def load_categories(self) -> Dict[str, int]:
        def process(line: str) -> Tuple[str, int]:
            cls, idx = line.split()
            return cls, int(idx)

        file, md5 = self._CATEGORIES_META
        file = path.join(self.root, file)

        with open(file, "r") as fh:
            class_to_idx = dict(process(line) for line in fh)

        return class_to_idx

    def load_file_list(self) -> Tuple[List[Tuple[str, int]], List[int]]:
        def process(line: str, sep="/") -> Tuple[str, int]:
            image, idx = line.split()
            idx = self.places_idx_to_lsun_idx[int(idx)]
            return path.join(self.images_dir, image.lstrip(sep).replace(sep, os.sep)), idx

        file, md5 = self._FILE_LIST_META[self.split]
        file = path.join(self.root, file)

        with open(file, "r") as fh:
            images = []
            for line in fh:
                places_idx = int(line.split()[-1])
                if places_idx not in self.places_idx_to_lsun_idx:
                    continue
                images.append(process(line))

        _, targets = zip(*images)
        return images, list(targets)

    def extra_repr(self) -> str:
        return "\n".join(("Split: {split}", "Small: {small}")).format(**self.__dict__)
