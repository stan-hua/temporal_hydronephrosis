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
