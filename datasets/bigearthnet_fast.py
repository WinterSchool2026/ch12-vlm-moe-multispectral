"""
Optimized BigEarthNet-Landsat dataset loader.

Changes vs datasets/bigearthnet.py:
- opens TIFF once per sample (image + georef metadata in same handle)
- caches CRS transformers to reduce repeated pyproj setup
"""

import json
import os
from datetime import datetime

import pandas as pd
import pyproj
import rasterio
import torch
from torch.utils.data import Dataset

from utils.data_config import BigEarthNetInfo
from utils.data_utils import normalize_latlon, normalize_timestamp


class BigEarthNetDatasetLSFast(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, supervised=False, num_classes=19
    ):
        self.data = pd.read_csv(csv_file)
        self.imgs_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.class2idx = {c: i for i, c in enumerate(BigEarthNetInfo.CLASSES[43])}
        self.supervised = supervised
        self.num_classes = num_classes
        self._transformer_cache = {}

    def _parse_datetime(self, filename):
        parts = filename.split("_")
        try:
            date_part = parts[2][0:10]
            dt = datetime.strptime(date_part, "%Y%m%dT%H")
            norm_week, norm_hour = normalize_timestamp(dt)
            return norm_week, norm_hour
        except (IndexError, ValueError):
            return 0.0, 0.0

    def _latlon_from_src(self, src):
        center_x = (src.bounds.left + src.bounds.right) / 2
        center_y = (src.bounds.top + src.bounds.bottom) / 2

        crs_str = src.crs.to_string()
        if crs_str != "EPSG:4326":
            transformer = self._transformer_cache.get(crs_str)
            if transformer is None:
                transformer = pyproj.Transformer.from_crs(
                    src.crs, "EPSG:4326", always_xy=True
                )
                self._transformer_cache[crs_str] = transformer
            lon, lat = transformer.transform(center_x, center_y)
        else:
            lon, lat = center_x, center_y

        return normalize_latlon(lat, lon)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        img_path = os.path.join(self.imgs_dir, f"{img_id}.tif")
        label_path = os.path.join(self.labels_dir, f"{img_id}_labels_metadata.json")

        with rasterio.open(img_path) as src:
            img = src.read()
            lat, lon = self._latlon_from_src(src)

        img = torch.from_numpy(img)

        week, hour = self._parse_datetime(img_id)
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
                    BigEarthNetInfo.MATCH_LABELS.get(class_idx) for class_idx in indices
                ]
                indices = [class_idx for class_idx in indices_optional if class_idx is not None]
            target = torch.zeros(self.num_classes, dtype=torch.long)
            target[indices] = 1
        else:
            target = torch.tensor([])

        return img, target, meta_week, meta_hour, meta_lat, meta_lon
