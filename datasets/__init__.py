import os
import torchvision.transforms as transforms


from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from .celeba import CelebA, attr_names as celeba_attr_names
from .places import Places365
from .lsun import LSUN
from .bar import BAR


def get_train_val(dataset_name, args, train_transform, val_transform):
    if dataset_name == 'celeba':
        train_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='train',
                           target_type='attr', transform=train_transform)
        val_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='valid',
                         target_type='attr', transform=val_transform)
    else:
        raise NotImplementedError

    return train_set, val_set


def get_train_val_test(dataset_name, args, train_transform, val_transform):
    if dataset_name == 'celeba':
        train_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='train',
                           target_type='attr', transform=train_transform)
        val_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='valid',
                         target_type='attr', transform=val_transform)
        test_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='test',
                          target_type='attr', transform=val_transform)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set


def get_train(dataset_name, args, train_transform, return_original_img, target_type='attr'):
    if dataset_name == 'celeba':
        train_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='train',
                           target_type=target_type, transform=train_transform,
                           return_original_img=return_original_img)
    else:
        raise NotImplementedError

    return train_set


def get_val(dataset_name, args, val_transform, return_original_img=False, target_type='attr'):
    if dataset_name == 'celeba':
        val_set = CelebA(root=args.dset_dir, image_size=args.image_size, split='valid',
                         target_type=target_type, transform=val_transform,
                         return_original_img=return_original_img)
    else:
        raise NotImplementedError

    return val_set


def get_scene_dataset_train_val(dataset_name, args, train_transform, val_transform):
    if dataset_name == 'places':
        train_set = Places365(root=args.dset_dir, split='train',
                              transform=train_transform)
        val_set = Places365(root=args.dset_dir, split='val',
                            transform=val_transform)
    elif dataset_name == 'lsun':
        train_set = LSUN(root=os.path.join(args.dset_dir, 'lsun'),
                         classes='train', transform=train_transform)
        val_set = LSUN(root=os.path.join(args.dset_dir, 'lsun'),
                       classes='val', transform=val_transform)
    else:
        raise NotImplementedError

    return train_set, val_set


def get_action_dataset_train_test(args, train_transform, test_transform):
    train_set = BAR(root=args.dset_dir, split='train', transform=train_transform)
    test_set = BAR(root=args.dset_dir, split='test', transform=test_transform)

    return train_set, test_set


def get_scene_dataset_train(dataset_name, args, train_transform, return_original_img=False):
    if dataset_name == 'places':
        train_set = Places365(root=args.dset_dir, split='train',
                              transform=train_transform,
                              return_original_img=return_original_img)
    elif dataset_name == 'lsun':
        train_set = LSUN(root=os.path.join(args.dset_dir, 'lsun'),
                         classes='train', transform=train_transform)
    else:
        raise NotImplementedError

    return train_set


def get_scene_dataset_val(dataset_name, args, val_transform, return_original_img=False):
    if dataset_name == 'places':
        val_set = Places365(root=args.dset_dir, split='val',
                            transform=val_transform,
                            return_original_img=return_original_img)
    elif dataset_name == 'lsun':
        val_set = LSUN(root=os.path.join(args.dset_dir, 'lsun'),
                       classes='val', transform=val_transform)
    else:
        raise NotImplementedError

    return val_set


def get_attr_names(dataset_name):
    if dataset_name == "celeba":
        return celeba_attr_names
    else:
        raise NotImplementedError


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


def get_transform_face():
    # code from https://github.com/kohpangwei/group_DRO/
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = 224

    val_transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            target_resolution,
            scale=(0.7, 1.0),
            ratio=(1.0, 1.3333333333333333),
            interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform
