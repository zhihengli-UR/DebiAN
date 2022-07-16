# modified based on https://github.com/alinlab/LfF/blob/master/make_dataset.py

import os
import pickle
import numpy as np
import torch

from torchvision.datasets import MNIST
from tqdm import tqdm
from imageio import imwrite


colors_path = "datasets/make_datasets/colors.th"
left_mean_color = torch.load(colors_path)  # 0-1
right_mean_color = 1 - left_mean_color  # 0-1


def colorize(raw_image, severity, left_color_label, right_color_label):
    raw_image = raw_image.unsqueeze(0).float()  # 0-255
    # raw_image: (1, 28, 28)

    std_color = [0.05, 0.02, 0.01, 0.005, 0.002][severity - 1]
    left_color = torch.clamp(
        left_mean_color[left_color_label] + torch.randn((3, 1, 1)) * std_color,
        0,
        1,
    )
    right_color = torch.clamp(
        right_mean_color[right_color_label]
        + torch.randn((3, 1, 1)) * std_color,
        0,
        1,
    )

    img_size = raw_image.shape[-1]
    assert raw_image.shape[1] == raw_image.shape[2]
    assert (img_size % 2) == 0

    img_bg_mask = raw_image != 255

    left_bg_mask = torch.zeros(1, img_size, img_size)
    left_bg_mask[:, :, : img_size // 2] = 1
    left_bg_mask *= img_bg_mask
    right_bg_mask = torch.zeros(1, img_size, img_size)
    right_bg_mask[:, :, img_size // 2 :] = 1
    right_bg_mask *= img_bg_mask

    image = (
        raw_image
        + left_bg_mask * 255 * left_color
        + right_bg_mask * 255 * right_color
    )
    image = torch.clamp(image, 0, 255)

    return image


def make_attr_labels(
    target_labels, left_color_bias_aligned_ratio, right_color_bias_aligned_ratio
):
    num_classes = target_labels.max().item() + 1
    num_samples_per_class = np.array(
        [
            torch.sum(target_labels == label).item()
            for label in range(num_classes)
        ]
    )
    left_bias_ratios_per_class = left_color_bias_aligned_ratio * np.eye(
        num_classes
    ) + (1 - left_color_bias_aligned_ratio) / (num_classes - 1) * (
        1 - np.eye(num_classes)
    )
    left_bias_ratios_per_class = left_bias_ratios_per_class[:, :, None]

    right_bias_ratios_per_class = right_color_bias_aligned_ratio * np.eye(
        num_classes
    ) + (1 - right_color_bias_aligned_ratio) / (num_classes - 1) * (
        1 - np.eye(num_classes)
    )
    right_bias_ratios_per_class = right_bias_ratios_per_class[:, None, :]

    ratios_per_class = left_bias_ratios_per_class * right_bias_ratios_per_class
    ratios_per_class = ratios_per_class.reshape(10, 100)

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis]
        * np.cumsum(ratios_per_class, axis=1)
    ).round()

    left_color_attr_labels = torch.zeros_like(target_labels)
    right_color_attr_labels = torch.zeros_like(target_labels)
    for label in range(10):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            index = np.min(
                np.nonzero(corruption_milestones > corruption_idx)[0]
            ).item()
            left_color_attr_labels[idx] = index // 10
            right_color_attr_labels[idx] = index % 10

    assert left_color_attr_labels.unique().shape[0] == 10
    assert right_color_attr_labels.unique().shape[0] == 10

    return left_color_attr_labels, right_color_attr_labels


def make_colored_mnist(
    data_dir,
    skewed_ratio_bias_left_color,
    skewed_ratio_bias_right_color,
    severity,
):
    mnist_dir = os.path.join(data_dir, "MNIST")
    colored_mnist_dir = os.path.join(
        data_dir,
        f"ColoredMNIST-SkewedA{skewed_ratio_bias_left_color}-SkewedB{skewed_ratio_bias_right_color}-Severity{severity}",
    )
    os.makedirs(colored_mnist_dir, exist_ok=True)
    print(colored_mnist_dir)
    attr_names = ["digit", "LColor", "RColor"]
    attr_names_path = os.path.join(colored_mnist_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["valid", "train"]:
        dataset = MNIST(mnist_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(colored_mnist_dir, split), exist_ok=True)

        if split == "train":
            left_color_bias_aligned_ratio = 1.0 - skewed_ratio_bias_left_color
            right_color_bias_aligned_ratio = 1.0 - skewed_ratio_bias_right_color
        else:
            left_color_bias_aligned_ratio = (
                0.1  # balanced because of 10 classes
            )
            right_color_bias_aligned_ratio = 0.1

        left_color_labels, right_color_labels = make_attr_labels(
            torch.LongTensor(dataset.targets),
            left_color_bias_aligned_ratio,
            right_color_bias_aligned_ratio,
        )

        images, attrs = [], []
        for img, target_label, left_color_label, right_color_label in tqdm(
            zip(
                dataset.data,
                dataset.targets,
                left_color_labels,
                right_color_labels,
            ),
            total=len(left_color_labels),
        ):
            colored_img = colorize(
                img, severity, left_color_label.item(), right_color_label.item()
            )
            colored_img = np.moveaxis(
                np.array(colored_img).astype(np.uint8), 0, 2
            )

            images.append(colored_img)
            attrs.append([target_label, left_color_label, right_color_label])

        images = np.array(images)
        attrs = np.array(attrs)

        image_path = os.path.join(colored_mnist_dir, split, "images.npy")
        np.save(image_path, images)
        attr_path = os.path.join(colored_mnist_dir, split, "attrs.npy")
        np.save(attr_path, attrs)

        if split == "valid":
            # save examples
            align_conflict_str_list = ["conflict", "align"]
            left_color_aligned = attrs[:, 0] == attrs[:, 1]
            right_color_aligned = attrs[:, 0] == attrs[:, 2]

            for idx_cls in range(10):
                cls_mask = attrs[:, 0] == idx_cls

                for left_align in range(2):
                    for right_align in range(2):
                        cur_align_conflict_mask = (
                            left_color_aligned == left_align
                        ) * (right_color_aligned == right_align)
                        left_str = align_conflict_str_list[left_align]
                        right_str = align_conflict_str_list[right_align]

                        dir_path = os.path.join(
                            colored_mnist_dir,
                            "examples",
                            f"left_{left_str}_right_{right_str}",
                        )
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        fpath = os.path.join(dir_path, f"{idx_cls}.png")

                        idx_img = (
                            cls_mask * cur_align_conflict_mask
                        ).nonzero()[0]

                        assert len(idx_img) > 0
                        idx_img = idx_img[0]
                        img = images[idx_img]
                        imwrite(fpath, img)
            example_dir = os.path.join(colored_mnist_dir, "examples")
            print(f"examples write to {example_dir}")


def main():
    severity = 4
    skewed_ratio_lst = [0.005, 0.01, 0.02, 0.05]
    for left_color in range(len(skewed_ratio_lst)):
        for right_color in range(left_color, len(skewed_ratio_lst)):
            left_color_skewed_ratio = skewed_ratio_lst[left_color]
            right_color_skewed_ratio = skewed_ratio_lst[right_color]
            assert left_color_skewed_ratio <= right_color_skewed_ratio
            make_colored_mnist(
                "data/multi_color_mnist",
                left_color_skewed_ratio,
                right_color_skewed_ratio,
                severity,
            )


if __name__ == "__main__":
    main()
