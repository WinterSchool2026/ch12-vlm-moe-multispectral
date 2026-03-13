"""
Multiple transformation classes for BEN-LS dataset.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""


class CenterCrop40:
    """Crop the central 40x40 pixels of a (C,H,W) image tensor."""

    def __call__(self, img):
        _, h, w = img.shape
        top = (h - 40) // 2
        left = (w - 40) // 2
        return img[:, top : top + 40, left : left + 40]


class ToFloat:
    """Convert image to float tensor."""

    def __call__(self, img):
        return img.float()


class ZScoreNormalize:
    """Normalize per-band with given mean and std tensors."""

    def __init__(self, mean, std):
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std
