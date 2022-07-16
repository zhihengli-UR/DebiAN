import argparse
import os

from common.utils import initialize_seeds


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_dir', default='data', type=str)
    parser.add_argument('--dset_name', type=str, default='multi_color_mnist')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lambda_penalty', type=float, default=1.0)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--arch', type=str, default='mlp')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # dataset
    parser.add_argument('--left_color_skew', type=float, default=0.0, choices=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--right_color_skew', type=float, default=0.0, choices=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--severity', type=int, default=4)

    return parser


def parse_and_check(parser, required_args=None):
    args = parser.parse_args()
    assert args.left_color_skew <= args.right_color_skew
    # set seeds
    initialize_seeds(args.seed)

    if required_args is not None:
        if isinstance(required_args, str):
            required_args = [required_args]
        for a in required_args:
            assert getattr(args, a, None) is not None, f'{a} is required.'

    if getattr(args, 'ckpt_dir', None) is not None:
        assert os.path.isdir(args.ckpt_dir)

    return args
