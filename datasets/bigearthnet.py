"""
PyTorch Dataset for loading and preprocessing
BigEarthNet-Landsat satellite imagery and metadata.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import json
import os
from datetime import datetime
import pandas as pd
import rasterio
import pyproj
import torch
from torch.utils.data import Dataset
from utils.data_utils import normalize_latlon, normalize_timestamp
from utils.data_config import BigEarthNetInfo


class BigEarthNetDatasetLS(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, supervised=False, num_classes=19
    ):
        """
        Args:
            csv_file (str): Path to CSV containing image IDs.
            root_dir (str): Root directory with images.
            transform (callable, optional): Optional transform for images.
        """
        self.data = pd.read_csv(csv_file)
        self.imgs_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.class2idx = {c: i for i, c in enumerate(BigEarthNetInfo.CLASSES[43])}
        self.supervised = supervised
        self.num_classes = num_classes

    def _crop_center_40(self, img):
        """
        Crops the central 40x40 patch from (C, H, W).
        """
        _, h, w = img.shape
        top = (h - 40) // 2
        left = (w - 40) // 2
        return img[:, top : top + 40, left : left + 40]

    def _parse_datetime(self, filename):
        # Example: S2A_MSIL2A_20170717T113321_28_89
        parts = filename.split("_")

        try:
            date_part = parts[2][0:10]
            dt = datetime.strptime(date_part, "%Y%m%dT%H")
            norm_week, norm_hour = normalize_timestamp(dt)
            return norm_week, norm_hour
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse datetime from filename '{filename}': {e}")
            return None, None

    def _get_latlon(self, filepath):
        with rasterio.open(filepath) as src:
            center_x = (src.bounds.left + src.bounds.right) / 2
            center_y = (src.bounds.top + src.bounds.bottom) / 2
            if src.crs.to_string() != "EPSG:4326":
                transformer = pyproj.Transformer.from_crs(
                    src.crs, "EPSG:4326", always_xy=True
                )
                lon, lat = transformer.transform(center_x, center_y)
            else:
                lon, lat = center_x, center_y
        norm_lat, norm_lon = normalize_latlon(lat, lon)
        return norm_lat, norm_lon

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        img_path = os.path.join(self.imgs_dir, f"{img_id}.tif")
        label_path = os.path.join(self.labels_dir, f"{img_id}_labels_metadata.json")

        # Load image
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)

        img = torch.from_numpy(img)

        # Parse metadata
        week, hour = self._parse_datetime(img_id)
        lat, lon = self._get_latlon(img_path)

        meta_week = torch.tensor(week, dtype=torch.float32)
        meta_hour = torch.tensor(hour, dtype=torch.float32)
        meta_lat = torch.tensor(lat, dtype=torch.float32)
        meta_lon = torch.tensor(lon, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        if self.supervised:
            with open(label_path, "r", encoding="utf-8") as f:
                labels = json.load(f)["labels"]
            indices = [self.class2idx[label] for label in labels]
            if self.num_classes == 19:
                indices_optional = [
                    BigEarthNetInfo.MATCH_LABELS.get(idx) for idx in indices
                ]
                indices = [idx for idx in indices_optional if idx is not None]
            target = torch.zeros(self.num_classes, dtype=torch.long)
            target[indices] = 1
        else:
            target = torch.tensor([])
        return img, target, meta_week, meta_hour, meta_lat, meta_lon


if __name__ == "__main__":
    dataset = BigEarthNetDatasetLS(
        "/mnt/storage/data/bigearthnet-l/bigearthnet-train.csv",
        "/mnt/storage/data/bigearthnet-l/bigearthnet-l",
    )
