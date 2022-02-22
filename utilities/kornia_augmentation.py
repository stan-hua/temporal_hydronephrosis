import torch
from kornia.augmentation import *
from torch import nn


class DataAugmentation(nn.Module):
    """Module to perform batch data augmentation on torch tensors using Kornia. If evaluation, only perform transforms
    and not augmentations."""

    def __init__(self, normalize=False, random_rotation=False, color_jitter=False,
                 random_gaussian_blur=False, random_motion_blur=False, random_noise=False,
                 prob=0.) -> None:
        super().__init__()
        self._mode = 'train'
        self.transforms = torch.nn.Sequential()
        self.augmentations = torch.nn.Sequential()

        if normalize:
            self.transforms.add_module("normalize", Normalize(mean=0.0, std=1.0, p=1.0))

        if random_rotation:
            self.augmentations.add_module("rotate", RandomRotation((-0.15, 0.15), p=prob))

        if color_jitter:
            self.augmentations.add_module("color_jitter", ColorJitter(brightness=[0.5, 2], contrast=[0.5, 2], p=prob))

        if random_gaussian_blur:
            self.augmentations.add_module("gaussian_blur", RandomGaussianBlur((3, 3), (0.1, 2.0), p=prob))

        if random_motion_blur:
            self.augmentations.add_module("motion_blur", RandomMotionBlur(3, 35., 0.5, p=prob))

        if random_noise:
            self.augmentations.add_module("gaussian_noise", RandomGaussianNoise(p=prob))

    def set_mode(self, mode='eval'):
        """Set mode of augmentation. If training, include augmentation. If evaluation, disable augmentation"""
        self._mode = mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        if self._mode == 'eval':
            x = self.augmentations(x)
        return x


def create_augmentation_str(args):
    """Create string from augmentations enabled."""
    if not args.augment_training:
        return ""

    augmentations = []

    if args.normalize:
        augmentations.append("normalize")
    if args.random_rotation:
        augmentations.append("rotate")
    if args.color_jitter:
        augmentations.append("color_jitter")
    if args.random_gaussian_blur:
        augmentations.append("gaussian_blur")
    if args.random_motion_blur:
        augmentations.append("motion_blur")
    if args.random_noise:
        augmentations.append("gaussian_noise")

    augmentations_str = "-".join(augmentations)

    return augmentations_str


def parse_augmentation_str(s: str):
    s_split = s.split("-")

    augmentations = {}
    for aug in ["normalize", "random_rotation", "color_jitter", "random_gaussian_blur", "random_motion_blur",
                "random_noise"]:
        augmentations[aug] = aug in s_split

    return augmentations


def instantiate_augmenter(hyperparams):
    """Instantiate image data augmentation object based on arguments. Return augmenter and name (given by all
    augmentation types enabled)."""
    if hyperparams is None:
        return None

    if not hyperparams['augmented'] or hyperparams['model'] is not 'baseline':
        return None

    augs = parse_augmentation_str(hyperparams['augmentations_str'])
    augmenter = DataAugmentation(normalize=augs['normalize'], random_rotation=augs['random_rotation'],
                                 color_jitter=augs['color_jitter'], random_gaussian_blur=augs['random_gaussian_blur'],
                                 random_motion_blur=augs['random_motion_blur'], random_noise=augs['random_noise'],
                                 prob=hyperparams['augment_probability'])

    return augmenter