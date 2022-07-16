import os
import torch
import numpy as np

from datasets import get_train_val_test, get_attr_names, get_transform_face
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class BaseFaceTrainer:
    def __init__(self, args):
        self.args = args
        train_transform, val_transform = get_transform_face()
        train_dataset, val_dataset, test_dataset = get_train_val_test(
            args.dset_name, args, train_transform, val_transform
        )
        train_dataset = self._modify_train_set(train_dataset)
        self.train_dataset = train_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

        self.all_attr_names = get_attr_names(args.dset_name)
        self.device = torch.device(0)

        if args.criterion in ["BCE", "GBCE"]:
            num_classes = 1
        elif args.criterion in ["CE", "GCE"]:
            num_classes = 2
        else:
            raise NotImplementedError
        self.num_classes = num_classes
        self.total_epoch = args.epoch

        self.attr_name = self.all_attr_names[args.attribute_index]
        args.attr_name = self.attr_name

        self.best_val_acc = 0
        self.cond_on_best_val_test_log_dict = {}

        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        self._setup_method_name_and_default_name()
        if self.args.name is None:
            self.args.name = self.default_name
        else:
            self.args.name += "_{}".format(self.default_name)
        print(f"run name: {self.args.name}")

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        if args.amp:
            self.scaler = torch.cuda.amp.GradScaler()

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
        val_log_dict = self.__eval_split(epoch, self.val_loader, "val")
        test_log_dict = self.__eval_split(epoch, self.test_loader, "test")

        log_dict.update(val_log_dict)
        log_dict.update(test_log_dict)

        return log_dict

    @torch.no_grad()
    def __eval_split(self, epoch, loader, split):
        self.classifier.eval()

        all_attr_label_lst = []
        pred_prob_lst = []
        log_dict = {}

        pbar = tqdm(
            loader,
            dynamic_ncols=True,
            desc="{} [{}/{}] attr_idx: {}, attr_name: {}, evaluating...".format(
                split,
                epoch,
                self.total_epoch,
                self.args.attribute_index,
                self.attr_name,
            ),
        )
        for img, all_attr_label in pbar:
            img = img.to(self.device)
            cls_out = self.classifier(img)
            if isinstance(cls_out, tuple):
                logit = cls_out[0]
            else:
                logit = cls_out

            if self.num_classes == 1:
                logit = logit.squeeze(-1)
                prob = torch.sigmoid(logit)
            else:
                two_cls_prob = torch.softmax(logit, dim=1)
                prob = two_cls_prob[:, 1]  # take positive class's probability

            pred_prob_lst.append(prob.cpu())
            all_attr_label_lst.append(all_attr_label.cpu())

        all_attr_label_lst = torch.cat(all_attr_label_lst, dim=0)
        np_all_attr_label_lst = all_attr_label_lst.numpy()
        target_label_lst = all_attr_label_lst[:, self.args.attribute_index]
        pred_prob_lst = torch.cat(pred_prob_lst, dim=0)

        np_target_label_lst = target_label_lst.numpy()
        np_pred_prob_lst = pred_prob_lst.numpy()
        np_pred_hard_lst = np_pred_prob_lst >= 0.5

        correct_predictions = np_pred_hard_lst == np_target_label_lst
        target_acc = correct_predictions.sum() / np_target_label_lst.shape[0]

        for idx_other_attr in range(all_attr_label_lst.shape[1]):
            if idx_other_attr == self.args.attribute_index:
                continue
            np_bias_label_lst = np_all_attr_label_lst[:, idx_other_attr]
            protected_attr_name = self.all_attr_names[idx_other_attr]

            acc_lst = []
            for bias_attr_val in range(2):
                for target_attr_val in range(2):
                    bool_value = (np_bias_label_lst == bias_attr_val) & (
                        np_target_label_lst == target_attr_val
                    )
                    if not any(bool_value):
                        continue
                    subgroup_correct_pred = correct_predictions[bool_value]
                    subgroup_num = bool_value.sum()
                    subgroup_acc = subgroup_correct_pred.sum() / subgroup_num

                    log_dict.update(
                        {
                            f"{split}_{idx_other_attr}_{protected_attr_name}_{bias_attr_val}_{self.attr_name}_{target_attr_val}_target_acc": subgroup_acc,
                        }
                    )
                    acc_lst.append(subgroup_acc)

            min_acc = np.min(acc_lst)
            log_dict[
                f"{split}_{idx_other_attr}_{protected_attr_name}_min_target_acc"
            ] = min_acc

            avg_acc = np.mean(acc_lst)
            log_dict[
                f"{split}_{idx_other_attr}_{protected_attr_name}_avg_target_acc"
            ] = avg_acc

        log_dict[f"{split}_target_acc"] = target_acc

        return log_dict

    def _save_ckpt(self, epoch, name):
        raise NotImplementedError

    def update_best_and_save_ckpt(self, epoch, log_dict):
        self._save_ckpt(epoch, "last")

        val_acc = log_dict["val_target_acc"]

        if (
            val_acc <= self.best_val_acc
            and epoch > 1
            and len(self.cond_on_best_val_test_log_dict) > 0
        ):
            log_dict.update(self.cond_on_best_val_test_log_dict)
            return

        prefix_lst = ["test"]

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            for key, value in log_dict.items():
                for prefix in prefix_lst:
                    if key.startswith(prefix + "_"):
                        new_key = f"cond_{key}"
                        self.cond_on_best_val_test_log_dict[new_key] = value
            self._save_ckpt(epoch, "best_val_acc")

        log_dict.update(self.cond_on_best_val_test_log_dict)

    def load_ckpt(self, resume_fpath):
        raise NotImplementedError

    def run(self):
        if self.args.resume is not None:
            start_epoch = self.load_ckpt(self.args.resume) + 1
        else:
            start_epoch = 1

            if not self.args.skip_eval:
                log_dict = self.eval(0)
                self.update_best_and_save_ckpt(0, log_dict)

            print(log_dict)

        for e in range(start_epoch, self.args.epoch + 1):
            log_dict = self.train(e)

            if not self.args.skip_eval:
                eval_log_dict = self.eval(e)
                log_dict.update(eval_log_dict)
                self.update_best_and_save_ckpt(e, log_dict)

            print(log_dict)
