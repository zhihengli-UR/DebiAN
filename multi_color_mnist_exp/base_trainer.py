import os
import torch


from datasets.multi_color_mnist import MultiColorMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.utils import MultiDimAverageMeter


class BaseTrainer:
    def __init__(self, args):
        self.args = args

        train_dataset = MultiColorMNIST(
            'data', 'train', args.left_color_skew, args.right_color_skew, args.severity)
        test_dataset = MultiColorMNIST(
            'data', 'valid', args.left_color_skew, args.right_color_skew, args.severity)

        attr_dims = []
        self.target_attr_index = MultiColorMNIST.target_attr_index
        self.left_color_bias_attr_index = MultiColorMNIST.left_color_bias_attr_index
        self.right_color_bias_attr_index = MultiColorMNIST.right_color_bias_attr_index

        train_target_attr = train_dataset.attr[:,
                                               MultiColorMNIST.target_attr_index]
        left_color_bias_attr = train_dataset.attr[:,
                                                  MultiColorMNIST.left_color_bias_attr_index]
        right_color_bias_attr = train_dataset.attr[:,
                                                   MultiColorMNIST.right_color_bias_attr_index]
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        attr_dims.append(torch.max(left_color_bias_attr).item() + 1)
        assert torch.max(right_color_bias_attr).item() + \
            1 == attr_dims[0] == attr_dims[1]
        self.num_classes = attr_dims[0]
        self.attr_dims = attr_dims
        self.eye_tsr = torch.eye(attr_dims[0]).long()

        self.train_dataset = train_dataset
        train_dataset = self._modify_train_set(train_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=args.pin_memory,
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
        log_dict = self.__eval_split(
            epoch, self.test_loader, self.args.dset_name)
        return log_dict

    @torch.no_grad()
    def __eval_split(self, epoch, loader, dset_name):
        self.classifier.eval()

        total_correct = 0
        total_num = 0

        attrwise_acc_meter = MultiDimAverageMeter([10, 10, 10])

        pbar = tqdm(loader, dynamic_ncols=True,
                    desc='[{}/{}] evaluating on ({})...'.format(epoch,
                                                                self.total_epoch,
                                                                dset_name))
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
            pred = logits.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == target_attr_label).long()
            total_correct += correct.sum().item()
            total_num += correct.size(0)
            attrwise_acc_meter.add(correct.cpu(), all_attr_label)

        global_acc = total_correct / total_num
        log_dict = {'global_acc': global_acc}

        multi_dim_color_acc = attrwise_acc_meter.get_mean()
        confict_align = ['conflict', 'align']
        total_acc_align_conflict = 0
        for left_color in range(2):
            for right_color in range(2):
                left_color_mask = (self.eye_tsr == left_color).unsqueeze(2)
                right_color_mask = (self.eye_tsr == right_color).unsqueeze(1)
                mask = left_color_mask * right_color_mask
                acc = multi_dim_color_acc[mask].mean().item()
                left_str = confict_align[left_color]
                right_str = confict_align[right_color]
                log_dict[f'left_{left_str}_right_{right_str}_acc'] = acc
                total_acc_align_conflict += acc

        left_conflict_acc = (log_dict['left_conflict_right_align_acc'] + log_dict['left_conflict_right_conflict_acc']) / 2
        right_conflict_acc = (log_dict['left_align_right_conflict_acc'] + log_dict['left_conflict_right_conflict_acc']) / 2
        log_dict['left_conflict_acc'] = left_conflict_acc
        log_dict['right_conflict_acc'] = right_conflict_acc
        log_dict['unbiased_acc'] = total_acc_align_conflict / 4

        return log_dict

    def save_ckpt(self, epoch):
        self._save_ckpt(epoch, 'last')

    def run(self):
        eval_dict = self.eval(0)
        self.save_ckpt(0)
        print(eval_dict)

        for e in range(1, self.args.epoch + 1):
            log_dict = self.train(e)
            eval_dict = self.eval(e)
            log_dict.update(eval_dict)
            self.save_ckpt(e)
            print(log_dict)
