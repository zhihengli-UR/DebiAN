import torch
import os
import PIL

from functools import partial
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision import transforms


attr_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


class CelebA(VisionDataset):
    base_folder = "celeba"

    def __init__(self, root, image_size, split="train", target_type="attr", transform=None, fpath_only=False, attribute_index=None, return_original_img=False):
        import pandas
        super(CelebA, self).__init__(root, transform=transform)
        self.split = split
        self.image_size = image_size
        self.target_type = target_type

        self.fpath_only = fpath_only

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split is None else (splits[1] == split)

        if target_type == 'identity':
            identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
            self.identity = torch.as_tensor(identity[mask].values)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='trunc')  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.attribute_index = attribute_index
        self.return_original_img = return_original_img
        self.pil_to_tensor_func = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        target = self.attr[index, :]
        if self.attribute_index is not None:
            target = target[self.attribute_index]

        if self.fpath_only:
            return os.path.abspath(fpath), target

        original_img = PIL.Image.open(fpath)

        if self.transform is not None:
            img = self.transform(original_img)
        else:
            img = original_img

        output = [img, target]

        if self.return_original_img:
            output.append(self.pil_to_tensor_func(original_img))

        return tuple(output)

    def __len__(self):
        return len(self.attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
