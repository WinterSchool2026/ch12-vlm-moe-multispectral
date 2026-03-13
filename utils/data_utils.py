"""
Utility functions used in differnt scripts.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import math
from matplotlib import pyplot as plt
import yaml
import numpy as np
import torch


def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def unnormalize(img, mean, std):
    mean = torch.as_tensor(mean, device=img.device).view(-1, 1, 1)
    std = torch.as_tensor(std, device=img.device).view(-1, 1, 1)
    return img * std + mean


def show_side_by_side(
    original,
    reconstruction,
    output_path,
    title1="Original",
    title2="Reconstruction",
    max_values=np.array([65454.0, 65454.0, 65330.308], dtype=np.float32),
    gamma=0.8,
):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, img, title in zip(axes, [original, reconstruction], [title1, title2]):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = img_np[..., [3, 2, 1]] / max_values
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = img_np.clip(0, 1)
        img_np = img_np ** (1 / gamma)
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
