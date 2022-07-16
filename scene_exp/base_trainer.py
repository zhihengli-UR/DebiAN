import os
import torch
import numpy as np


from datasets import get_scene_dataset_train_val, get_scene_dataset_val
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseSceneTrainer:
    def __init__(self, args):
        self.args = args
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset, val_dataset = get_scene_dataset_train_val(
            args.dset_name, args, train_transform, val_transform)

        val_ood_dataset = get_scene_dataset_val(
            args.ood_dset_name, args, val_transform)
        self.class_names = train_dataset.categories_lst

        self.train_dataset = train_dataset
        self.num_classes = self.num_scene_categories = len(train_dataset.categories_set)
        train_dataset = self._modify_train_set(train_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False, pin_memory=True)
        self.val_ood_loader = DataLoader(val_ood_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True)

        self.device = torch.device(0)

        self.total_epoch = args.epoch

        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        self._setup_method_name_and_default_name()

        if args.name is None:
            args.name = self.default_name
        else:
            args.name += f'_{self.default_name}'

        args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        if not os.path.isdir(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        self.ckpt_dir = args.ckpt_dir

        self.best_val_acc = -1
        self.cond_on_best_val_test_log_dict = {}

    def train(self, epoch):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _setup_models(self):
        raise NotImplementedError

    def _setup_criterion(self):
        raise NotImplementedError

    def _setup_optimizers(self):
        raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _save_ckpt(self, epoch):
        raise NotImplementedError

    def _loss_backward(self, loss):
        if self.args.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    def eval(self, epoch):
        log_dict = {}
        eval_dict = self.__eval_split(epoch, self.val_loader, self.args.dset_name)
        eval_ood_dict = self.__eval_split(
            epoch, self.val_ood_loader, self.args.ood_dset_name)
        log_dict.update(eval_dict)
        log_dict.update(eval_ood_dict)
        return log_dict

    @torch.no_grad()
    def __eval_split(self, epoch, loader, dset_name):
        self.classifier.eval()
        total_label = []
        total_pred = []

        pbar = tqdm(loader, dynamic_ncols=True,
                    desc='[{}/{}] evaluating on biased dataset ({})...'.format(epoch,
                                                                               self.total_epoch,
                                                                               dset_name))
        for img, label in pbar:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            cls_out = self.classifier(img)
            if isinstance(cls_out, tuple):
                logits = cls_out[0]
            else:
                logits = cls_out
            pred = logits.argmax(dim=1)
            total_label.append(label)
            total_pred.append(pred)

        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        total_pred = torch.cat(total_pred, dim=0).cpu().numpy()

        acc = np.mean(total_pred == total_label)

        log_dict = {
            f'{dset_name}_accuracy': acc
        }

        return log_dict

    def update_best_and_save_ckpt(self, epoch, log_dict):
        val_acc = log_dict[f'{self.args.dset_name}_accuracy']

        if val_acc <= self.best_val_acc \
                and epoch > 1 and len(self.cond_on_best_val_test_log_dict) > 0:
            log_dict.update(self.cond_on_best_val_test_log_dict)
            return

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            for key, value in log_dict.items():
                if key.startswith(f'{self.args.ood_dset_name}_'):
                    new_key = f'cond_{key}'
                    self.cond_on_best_val_test_log_dict[new_key] = value
            self._save_ckpt(epoch)

        log_dict.update(self.cond_on_best_val_test_log_dict)

    def run(self):
        eval_dict = self.eval(0)
        self.update_best_and_save_ckpt(0, eval_dict)
        print(eval_dict)

        for e in range(1, self.args.epoch + 1):
            log_dict = self.train(e)
            eval_dict = self.eval(e)
            log_dict.update(eval_dict)
            self.update_best_and_save_ckpt(e, log_dict)

            print(log_dict)
