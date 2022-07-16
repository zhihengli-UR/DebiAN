import os
import torch

from celeba_exp.face_args import get_parser, parse_and_check
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.binary_classifier import (
    get_resnet50_classifier,
    get_resnet18_classifier,
)
from common.constants import EPS
from celeba_exp.base_trainer import BaseFaceTrainer


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Trainer(BaseFaceTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.second_train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

    def _setup_models(self):
        if self.args.arch == "resnet50":
            self.bias_discover_net = get_resnet50_classifier(
                num_classes=self.num_classes
            ).to(self.device)
            self.classifier = get_resnet50_classifier(
                num_classes=self.num_classes
            ).to(self.device)
        elif self.args.arch == "resnet18":
            self.bias_discover_net = get_resnet18_classifier(
                num_classes=self.num_classes
            ).to(self.device)
            self.classifier = get_resnet18_classifier(
                num_classes=self.num_classes
            ).to(self.device)
        else:
            raise NotImplementedError

    def _setup_criterion(self):
        if self.args.criterion == "BCE":
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            assert args.criterion == "CE"
            self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _setup_optimizers(self):
        if self.args.optimizer == "sgd":
            self.optimizer_bias_discover_net = torch.optim.SGD(
                self.bias_discover_net.parameters(),
                args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
            self.optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            self.optimizer_bias_discover_net = torch.optim.Adam(
                self.bias_discover_net.parameters(),
                args.lr,
                weight_decay=args.weight_decay,
            )
            self.optimizer = torch.optim.Adam(
                self.classifier.parameters(),
                args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        args.method_name = "debian"
        self.default_name = (
            "debian_bs_{}_wd_{:.0E}_lr_{:.0E}_{}_{:02d}_{}".format(
                args.batch_size,
                args.weight_decay,
                args.lr,
                args.dset_name,
                args.attribute_index,
                self.attr_name,
            )
        )

    def train(self, epoch):
        total_cls_loss = 0
        total_ce_loss = 0
        total_bias_discover_loss = 0
        total_bias_discover_deo_loss = 0
        total_bias_discover_penalty = 0

        pbar = tqdm(
            zip(self.train_loader, self.second_train_loader),
            dynamic_ncols=True,
            total=len(self.train_loader),
        )
        for idx, (main_data, second_data) in enumerate(pbar):
            # ============= start: train classifier net ================
            self.bias_discover_net.eval()
            self.classifier.train()
            img, label = main_data
            img = img.to(self.device)
            label = label[:, args.attribute_index]
            label = label.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                with torch.no_grad():
                    spurious_logits = self.bias_discover_net(img)

                target_logits = self.classifier(img)

                if self.num_classes == 1:
                    label = label.float()
                    label = label.reshape(target_logits.shape)
                    p_vanilla = torch.sigmoid(target_logits)
                    p_spurious = torch.sigmoid(spurious_logits)
                else:
                    label = label.long()
                    label = label.reshape(target_logits.shape[0])
                    p_vanilla = torch.softmax(target_logits, dim=1)[:, 1]
                    p_spurious = torch.softmax(spurious_logits, dim=1)[:, 1]

                # standard CE or BCE loss
                ce_loss = self.criterion(target_logits, label)

                with torch.cuda.amp.autocast(enabled=False):
                    # reweight CE with DEO
                    for target_val in range(2):
                        batch_bool = label.long().flatten() == target_val
                        p_vanilla_w_same_t_val = p_vanilla[batch_bool]
                        if target_val == 0:
                            p_vanilla_w_same_t_val = 1 - p_vanilla_w_same_t_val

                        p_spurious_w_same_t_val = p_spurious[batch_bool]

                        positive_spurious_group_avg_p = (
                            p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                        ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                        negative_spurious_group_avg_p = (
                            (1 - p_spurious_w_same_t_val)
                            * p_vanilla_w_same_t_val
                        ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                        if (
                            negative_spurious_group_avg_p
                            < positive_spurious_group_avg_p
                        ):
                            p_spurious_w_same_t_val = (
                                1 - p_spurious_w_same_t_val
                            )

                        weight = p_spurious_w_same_t_val
                        weight += 1

                        ce_loss[batch_bool] *= weight

                    ce_loss = ce_loss.mean()
                    loss = ce_loss

            self.optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)

            total_cls_loss += loss.item()
            total_ce_loss += ce_loss.item()
            # avg_cls_loss = total_cls_loss / (idx + 1)
            avg_ce_loss = total_ce_loss / (idx + 1)
            # ============= end: train classifier net ================

            # ============= start: train bias discover net ================
            img, label = second_data
            img = img.to(self.device)
            label = label[:, args.attribute_index]
            label = label.to(self.device)

            self.bias_discover_net.train()
            self.classifier.eval()

            with torch.cuda.amp.autocast(self.args.amp):
                with torch.no_grad():
                    target_logits = self.classifier(img)

                spurious_logits = self.bias_discover_net(img)

                if self.num_classes == 1:
                    label = label.float()
                    label = label.reshape(target_logits.shape)
                    p_vanilla = torch.sigmoid(target_logits)
                    p_spurious = torch.sigmoid(spurious_logits)
                else:
                    label = label.long()
                    label = label.reshape(target_logits.shape[0])
                    p_vanilla = torch.softmax(target_logits, dim=1)[:, 1]
                    p_spurious = torch.softmax(spurious_logits, dim=1)[:, 1]

                # ==== deo loss ======
                with torch.cuda.amp.autocast(enabled=False):
                    sum_discover_net_deo_loss = 0
                    sum_penalty = 0
                    for target_val in range(2):
                        batch_bool = label.long().flatten() == target_val
                        p_vanilla_w_same_t_val = p_vanilla[batch_bool]
                        if target_val == 0:
                            p_vanilla_w_same_t_val = 1 - p_vanilla_w_same_t_val
                        p_spurious_w_same_t_val = p_spurious[batch_bool]

                        positive_spurious_group_avg_p = (
                            p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                        ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                        negative_spurious_group_avg_p = (
                            (1 - p_spurious_w_same_t_val)
                            * p_vanilla_w_same_t_val
                        ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                        discover_net_deo_loss = -torch.log(
                            EPS
                            + torch.abs(
                                positive_spurious_group_avg_p
                                - negative_spurious_group_avg_p
                            )
                        )

                        negative_p_spurious_w_same_t_val = (
                            1 - p_spurious_w_same_t_val
                        )
                        penalty = -torch.log(
                            EPS
                            + 1
                            - torch.abs(
                                p_spurious_w_same_t_val.mean()
                                - negative_p_spurious_w_same_t_val.mean()
                            )
                        )

                        sum_discover_net_deo_loss += discover_net_deo_loss
                        sum_penalty += penalty

                    sum_penalty *= self.args.lambda_penalty
                    loss_discover = sum_discover_net_deo_loss + sum_penalty

            self.optimizer_bias_discover_net.zero_grad(set_to_none=True)
            self._loss_backward(loss_discover)
            self._optimizer_step(self.optimizer_bias_discover_net)

            total_bias_discover_deo_loss += sum_discover_net_deo_loss.item()
            total_bias_discover_penalty += sum_penalty.item()
            total_bias_discover_loss += loss_discover.item()
            avg_discover_net_deo_loss = total_bias_discover_deo_loss / (idx + 1)
            avg_discover_net_penalty = total_bias_discover_penalty / (idx + 1)
            avg_bias_discover_loss = total_bias_discover_loss / (idx + 1)
            # ============= end: train bias discover net ================

            self._scaler_update()  # activated only when using amp

            pbar.set_description(
                "[{}/{}] {}_{}, ce: {:.3f}, l_dis: {:.3f}, deo: {:.3f}, penalty: {:.3f}".format(
                    epoch,
                    self.total_epoch,
                    self.args.attribute_index,
                    self.attr_name,
                    avg_ce_loss,
                    avg_bias_discover_loss,
                    avg_discover_net_deo_loss,
                    avg_discover_net_penalty,
                )
            )

        log_dict = {
            "loss": total_cls_loss / len(self.train_loader),
            "ce_loss": total_ce_loss / len(self.train_loader),
            "bias_discover_loss": total_bias_discover_loss
            / len(self.train_loader),
            "bias_discover_deo_loss": total_bias_discover_deo_loss
            / len(self.train_loader),
            "bias_discover_penalty_loss": total_bias_discover_penalty
            / len(self.train_loader),
        }

        return log_dict

    def _save_ckpt(self, epoch, name):
        bias_discover_net_state = {
            "model": self.bias_discover_net.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer_bias_discover_net.state_dict(),
        }

        classifier_net_state = {
            "model": self.classifier.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
        }
        best_val_acc_ckpt_fpath = os.path.join(self.ckpt_dir, f"{name}.pth")
        best_val_acc_bias_discover_net_ckpt_fpath = os.path.join(
            self.ckpt_dir, f"bias_discover_net_{name}.pth"
        )

        if getattr(self, "scaler", None) is not None:
            classifier_net_state["scaler"] = self.scaler.state_dict()

        torch.save(
            bias_discover_net_state, best_val_acc_bias_discover_net_ckpt_fpath
        )
        torch.save(classifier_net_state, best_val_acc_ckpt_fpath)

    def load_ckpt(self, resume_fpath):
        classifier_net_state = torch.load(resume_fpath)
        dir_fpath = os.path.dirname(resume_fpath)

        self.classifier.load_state_dict(classifier_net_state["model"])
        self.optimizer.load_state_dict(classifier_net_state["optimizer"])

        bias_discover_net_ckpt_fpath = os.path.join(
            dir_fpath, "bias_discover_net_last.pth"
        )
        assert os.path.exists(bias_discover_net_ckpt_fpath)

        bias_discover_net_state = torch.load(bias_discover_net_ckpt_fpath)

        self.bias_discover_net.load_state_dict(bias_discover_net_state["model"])
        self.optimizer_bias_discover_net.load_state_dict(
            bias_discover_net_state["optimizer"]
        )

        epoch = min(
            classifier_net_state["epoch"], bias_discover_net_state["epoch"]
        )

        if (
            getattr(self, "scaler", None) is not None
            and "scaler" in classifier_net_state
        ):
            self.scaler.load_state_dict(classifier_net_state["scaler"])

        return epoch


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default="exp/celeba")
    args = parse_and_check(parser)
    trainer = Trainer(args)
    trainer.run()
