import os
import torch


from torch.utils.data import DataLoader
from tqdm import tqdm
from models.binary_classifier import get_resnet18_classifier
from bar_exp.bar_args import get_parser, parse_and_check
from common.constants import EPS
from bar_exp.base_trainer import BaseActionTrainer


class Trainer(BaseActionTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.second_train_loader = DataLoader(self.train_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True,
                                              pin_memory=args.pin_memory,
                                              persistent_workers=args.num_workers > 0)

    def _setup_models(self):
        self.bias_discover_net = get_resnet18_classifier(
            num_classes=self.num_classes).to(self.device)
        self.classifier = get_resnet18_classifier(
            num_classes=self.num_classes).to(self.device)

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _setup_optimizers(self):
        self.optimizer_bias_discover_net = torch.optim.Adam(
            self.bias_discover_net.parameters(), args.lr)
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), args.lr)

    def _setup_method_name_and_default_name(self):
        self.args.method_name = 'debian'
        args = self.args
        default_name = '{}_{}_lr_{:.0E}_{}'.format(
            self.args.method_name,
            args.criterion,
            args.lr,
            args.dset_name)
        self.default_name = default_name

    def train(self, epoch):
        total_cls_loss = 0
        total_ce_loss = 0
        total_bias_discover_loss = 0
        total_bias_discover_deo_loss = 0
        total_bias_discover_penalty = 0

        pbar = tqdm(zip(self.train_loader, self.second_train_loader),
                    dynamic_ncols=True, total=len(self.train_loader))
        for idx, (main_data, second_data) in enumerate(pbar):
            # ============= start: train classifier net ================
            self.bias_discover_net.eval()
            self.classifier.train()
            img, label = main_data
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                spurious_logits = self.bias_discover_net(img)
            with torch.cuda.amp.autocast(enabled=self.args.amp):
                target_logits = self.classifier(img)
                label = label.long()
                label = label.reshape(target_logits.shape[0])
                p_vanilla = torch.softmax(target_logits, dim=1)
                p_spurious = torch.sigmoid(spurious_logits)

                # standard CE or BCE loss
                ce_loss = self.criterion(target_logits, label)

                # reweight CE with DEO
                for target_val in range(self.num_classes):
                    batch_bool = label.long().flatten() == target_val
                    p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                    p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                    positive_spurious_group_avg_p = (p_spurious_w_same_t_val * p_vanilla_w_same_t_val).sum() / (
                        p_spurious_w_same_t_val.sum() + EPS)
                    negative_spurious_group_avg_p = ((1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val).sum() / (
                        (1 - p_spurious_w_same_t_val).sum() + EPS)

                    if negative_spurious_group_avg_p < positive_spurious_group_avg_p:
                        p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val

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
            avg_ce_loss = total_ce_loss / (idx + 1)
            # ============= end: train classifier net ================

            # ============= start: train bias discover net ================
            img, label = second_data
            img = img.to(self.device)
            label = label.to(self.device)

            self.bias_discover_net.train()
            self.classifier.eval()

            with torch.no_grad():
                target_logits = self.classifier(img)
            with torch.cuda.amp.autocast(self.args.amp):
                spurious_logits = self.bias_discover_net(img)

                label = label.long()
                label = label.reshape(target_logits.shape[0])
                p_vanilla = torch.softmax(target_logits, dim=1)
                p_spurious = torch.sigmoid(spurious_logits)

                # ==== deo loss ======
                sum_discover_net_deo_loss = 0
                sum_penalty = 0
                for target_val in range(self.num_classes):
                    batch_bool = label.long().flatten() == target_val
                    p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                    p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                    positive_spurious_group_avg_p = (p_spurious_w_same_t_val * p_vanilla_w_same_t_val).sum() / (
                        p_spurious_w_same_t_val.sum() + EPS)
                    negative_spurious_group_avg_p = ((1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val).sum() / (
                        (1 - p_spurious_w_same_t_val).sum() + EPS)

                    discover_net_deo_loss = -torch.log(
                        EPS + torch.abs(positive_spurious_group_avg_p - negative_spurious_group_avg_p))

                    negative_p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val
                    penalty = -torch.log(
                        EPS + 1 - torch.abs(p_spurious_w_same_t_val.mean() - negative_p_spurious_w_same_t_val.mean()))

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
            avg_discover_net_deo_loss = total_bias_discover_deo_loss / \
                (idx + 1)
            avg_discover_net_penalty = total_bias_discover_penalty / (idx + 1)
            avg_bias_discover_loss = total_bias_discover_loss / (idx + 1)
            # ============= end: train bias discover net ================

            self._scaler_update()  # activated only when using amp

            pbar.set_description('[{}/{}] ce: {:.3f}, l_dis: {:.3f}, deo: {:.3f}, penalty: {:.3f}'.format(epoch,
                                                                                                          self.total_epoch,
                                                                                                          avg_ce_loss,
                                                                                                          avg_bias_discover_loss,
                                                                                                          avg_discover_net_deo_loss,
                                                                                                          avg_discover_net_penalty))

        log_dict = {
            'loss': total_cls_loss / len(self.train_loader),
            'ce_loss': total_ce_loss / len(self.train_loader),
            'bias_discover_loss': total_bias_discover_loss / len(self.train_loader),
            'bias_discover_deo_loss': total_bias_discover_deo_loss / len(self.train_loader),
            'bias_discover_penalty_loss': total_bias_discover_penalty / len(self.train_loader)
        }

        return log_dict

    def _save_ckpt(self, epoch):
        bias_discover_net_state = {
            'model': self.bias_discover_net.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer_bias_discover_net.state_dict()
        }

        classifier_net_state = {
            'model': self.classifier.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        best_val_acc_ckpt_fpath = os.path.join(self.ckpt_dir, 'best_val.pth')
        best_val_acc_bias_discover_net_ckpt_fpath = os.path.join(
            self.ckpt_dir, 'bias_discover_net_best_val.pth')

        torch.save(bias_discover_net_state,
                   best_val_acc_bias_discover_net_ckpt_fpath)
        torch.save(classifier_net_state, best_val_acc_ckpt_fpath)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--ckpt_dir', type=str,
                        default='exp/bar')
    args = parse_and_check(parser)
    trainer = Trainer(args)
    trainer.run()
