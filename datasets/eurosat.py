"""
PyTorch Dataset for loading and preprocessing
EuroSAT-Landsat images and class labels.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import pathlib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio


class EuroSATDatasetLS(Dataset):
    """
    Images live in <root_dir>/<ClassName>/*.{tif,tiff,jpg,jpeg,png,bmp}
    Split file has one entry per line, either:
      - "basename.ext"  (e.g., "Residential_2029.jpg")
      - "ClassName/basename.ext"

    Matching is case-insensitive and by stem (basename without extension).
    """

    def __init__(
        self,
        root_dir,
        split_file,
        transform=None,
        return_one_hot=False,
        allowed_exts=(".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"),
        strict=False,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.split_file = pathlib.Path(split_file)
        self.transform = transform
        self.return_one_hot = return_one_hot
        self.strict = strict
        allowed_exts = tuple(e.lower() for e in allowed_exts)

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        # 1) Discover classes (immediate subfolders)
        class_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class subfolders under: {self.root_dir}")
        self.class_names = sorted([p.name for p in class_dirs])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # 2) Index images: by stem and by class/stem (lowercase)
        by_stem = {}
        by_class_stem = {}
        for cls_name in self.class_names:
            cdir = self.root_dir / cls_name
            for path in cdir.rglob("*"):
                if path.is_file() and path.suffix.lower() in allowed_exts:
                    stem = path.stem.lower()
                    cls_l = cls_name.lower()
                    label = self.class_to_idx[cls_name]
                    by_stem[stem] = (path, label)
                    by_class_stem[f"{cls_l}/{stem}"] = (path, label)

        if not by_stem:
            raise RuntimeError(f"No images with {allowed_exts} under: {self.root_dir}")

        # 3) Resolve split entries
        with open(self.split_file, "r", encoding="utf-8") as f:
            requested = [ln.strip() for ln in f if ln.strip()]

        samples = []
        missing = []
        for entry in requested:
            key = entry.replace("\\", "/").strip().strip("/").lower()
            if "/" in key:
                cls, filepart = key.split("/", 1)
                stem = pathlib.Path(filepart).stem.lower()
                rec = by_class_stem.get(f"{cls}/{stem}") or by_stem.get(stem)
            else:
                stem = pathlib.Path(key).stem.lower()
                rec = by_stem.get(stem)

            (samples if rec else missing).append(rec or entry)

        if not samples:
            msg = (
                f"No split entries matched under {self.root_dir}. "
                "Check root_dir level, file extensions, and class names."
            )
            if strict:
                raise FileNotFoundError(msg)
            print("[EuroSATDataset] WARNING:", msg)

        if missing:
            msg = (
                f"{len(missing)} entries in {self.split_file.name} not matched. "
                f"Example: {missing[:5]}"
            )
            if strict:
                raise FileNotFoundError(msg)
            print("[EuroSATDataset] WARNING:", msg)

        self.samples = samples  # list of (path, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # rasterio returns (bands, H, W) -> good for torchvision tensor transforms
        with rasterio.open(path) as src:
            arr = src.read()  # numpy, shape (C,H,W), dtype depends on file
        img = torch.from_numpy(arr[:-1, ...])

        if self.transform is not None:
            img = self.transform(img)  # works with torchvision tensor transforms

        if self.return_one_hot:
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            y[target] = 1.0
            return img, y

        return img, target

    @property
    def classes(self):
        return self.class_names


if __name__ == "__main__":
    # Example usage

    root = "/mnt/storage/data/eurosat-l/eurosat-l"
    split_txt = "/mnt/storage/data/eurosat-l/eurosat-train.txt"

    tfm = transforms.Compose(
        [
            transforms.Resize((40, 40)),
        ]
    )

    train_set = EuroSATDatasetLS(
        root_dir=root,
        split_file=split_txt,
        transform=tfm,
        return_one_hot=True,  # set True if you want one-hot labels
        strict=False,  # skip missing names instead of raising
    )

    img, label = train_set[0]  # img: Tensor, label: int (or one-hot Tensor)
    print(train_set.classes, len(train_set))
