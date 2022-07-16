import os
import torch

from datasets.bffhq import bFFHQDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.utils import MultiDimAverageMeter
from torchvision import transforms as T


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )
        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )
        train_dataset = bFFHQDataset(
            'data', 'train', transform=train_transform)
        valid_dataset = bFFHQDataset('data', 'valid', transform=test_transform)
        test_dataset = bFFHQDataset('data', 'test', transform=test_transform)

        attr_dims = [2, 2]
        self.target_attr_index = bFFHQDataset.target_attr_index
        self.bias_attr_index = bFFHQDataset.bias_attr_index

        self.num_classes = 1
        self.attr_dims = attr_dims
        self.eye_tsr = torch.eye(attr_dims[0]).long()

        self.train_dataset = train_dataset
        train_dataset = self._modify_train_set(train_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=args.pin_memory,
                                       persistent_workers=args.num_workers > 0)
        self.val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False, pin_memory=args.pin_memory,
                                     persistent_workers=args.num_workers > 0)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=False, pin_memory=args.pin_memory,
                                      persistent_workers=args.num_workers > 0)
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

        if args.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.best_val_acc = 0
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

    def _save_ckpt(self, epoch, name):
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
        log_dict = self.__eval_split(epoch, self.val_loader, 'val')
        test_log_dict = self.__eval_split(
            epoch, self.test_loader, 'test')
        log_dict.update(test_log_dict)
        eval_bias_pred_log_dict = self._eval_bias_pred(epoch,
                                                       self.test_loader,
                                                       self.args.dset_name)
        log_dict.update(eval_bias_pred_log_dict)
        return log_dict

    def _eval_bias_pred(self, epoch, loader, dset_name):
        return {}

    @torch.no_grad()
    def __eval_split(self, epoch, loader, split):
        self.classifier.eval()

        total_correct = 0
        total_num = 0

        attrwise_acc_meter = MultiDimAverageMeter([2, 2])

        pbar = tqdm(loader, dynamic_ncols=True,
                    desc='[{}/{}] evaluating ...'.format(epoch,
                                                         self.total_epoch))
        for img, all_attr_label in pbar:
            img = img.to(self.device, non_blocking=True)
            target_attr_label = all_attr_label[:, self.target_attr_index]
            target_attr_label = target_attr_label.to(
                self.device, non_blocking=True)
            cls_out = self.classifier(img)
            if isinstance(cls_out, tuple):
                logits = cls_out[0]
            else:
                logits = cls_out
            prob = torch.sigmoid(logits).squeeze(-1)
            pred = prob > 0.5
            correct = (pred == target_attr_label).long()
            total_correct += correct.sum().item()
            total_num += correct.size(0)
            attrwise_acc_meter.add(correct.cpu(), all_attr_label)

        global_acc = total_correct / total_num
        log_dict = {f'{split}_global_acc': global_acc}

        multi_dim_color_acc = attrwise_acc_meter.get_mean()
        confict_align = ['conflict', 'align']
        total_acc_align_conflict = 0
        for color in range(2):
            color_mask = self.eye_tsr == color
            acc = multi_dim_color_acc[color_mask].mean().item()
            align_conflict_str = confict_align[color]
            log_dict[f'{split}_{align_conflict_str}_acc'] = acc
            total_acc_align_conflict += acc

        log_dict[f'{split}_unbiased_acc'] = total_acc_align_conflict / 2

        return log_dict

    def update_best_and_save_ckpt(self, epoch, log_dict):
        self._save_ckpt(epoch, 'last')

        val_acc = log_dict['val_unbiased_acc']

        if val_acc <= self.best_val_acc \
                and epoch > 1 and len(self.cond_on_best_val_test_log_dict) > 0:
            log_dict.update(self.cond_on_best_val_test_log_dict)
            return

        prefix_lst = ['test']

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            for key, value in log_dict.items():
                for prefix in prefix_lst:
                    if key.startswith(prefix + '_'):
                        new_key = f'cond_{key}'
                        self.cond_on_best_val_test_log_dict[new_key] = value
            self._save_ckpt(epoch, 'best_val_acc')

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
            print(log_dict, step=e)
