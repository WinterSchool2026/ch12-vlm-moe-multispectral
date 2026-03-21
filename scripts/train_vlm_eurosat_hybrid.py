import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.eurosat import EuroSATDatasetLS
from models.moe_mae import MOEMAE, build_model
from models.vlm import VLM, clip_multilabel_loss
from transformation.transformer import ToFloat, ZScoreNormalize
from utils.data_config import BigEarthNetInfo
from utils.data_utils import load_model


def _device_type_str(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _configure_encoder_unfreeze(model: VLM, n_last_layers: int) -> None:
    if n_last_layers <= 0:
        return
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise ValueError("Model encoder does not expose .layers for selective unfreezing.")

    layers = list(model.encoder.layers)
    if not layers:
        return
    n = min(n_last_layers, len(layers))
    for lyr in layers[-n:]:
        _set_requires_grad(lyr, True)


def _build_optimizer(
    model: nn.Module, lr: float, weight_decay: float, encoder_lr: float
) -> tuple[torch.optim.Optimizer, List[torch.nn.Parameter]]:
    enc_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("base_vlm.encoder."):
            enc_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    if enc_params:
        groups.append(
            {"params": enc_params, "lr": encoder_lr, "weight_decay": weight_decay}
        )
    if not groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(groups), other_params + enc_params


def _save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_loss: float,
    class_names: List[str],
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_names": class_names
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def _meta_zeros(batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    z = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
    return {"week": z.clone(), "hour": z.clone(), "lat": z.clone(), "lon": z.clone()}


@dataclass
class EuroBatch:
    imgs: torch.Tensor
    labels: torch.Tensor
    prompts: List[str]
    meta: Dict[str, torch.Tensor]


def _prepare_batch(
    batch,
    *,
    device: torch.device,
    class_names: List[str],
    prompt_template: str,
) -> EuroBatch:
    imgs, labels = batch
    imgs = imgs.to(device).float()
    labels = labels.to(device).long()
    prompts = [
        prompt_template.format(_normalize_class_text(class_names[int(y)]))
        for y in labels.detach().cpu()
    ]
    meta = _meta_zeros(batch_size=imgs.size(0), device=device)
    return EuroBatch(imgs=imgs, labels=labels, prompts=prompts, meta=meta)


def _normalize_class_text(name: str) -> str:
    text = name.strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = " ".join(text.split()).lower()
    if text == "sea lake":
        return "sea or lake"
    return text


def positives_from_class(labels: torch.Tensor) -> torch.Tensor:
    # labels: (B,) int class ids
    return labels.view(-1, 1).eq(labels.view(1, -1))


class EuroSATHybridVLM(nn.Module):
    def __init__(self, base_vlm: VLM, num_classes: int):
        super().__init__()
        self.base_vlm = base_vlm
        self.classifier = nn.Linear(base_vlm.text_dim, num_classes)

    def forward(self, prompts: List[str], imgs: torch.Tensor, meta: Dict[str, torch.Tensor]):
        logits_i2t = self.base_vlm(prompts, imgs, meta["week"], meta["hour"], meta["lat"], meta["lon"])
        _, _, z_img = self.base_vlm.image_encoder(imgs, meta["week"], meta["hour"], meta["lat"], meta["lon"])
        cls_image_embeddings = z_img[:, 4]
        prj_image_embeddings = self.base_vlm.image_proj(cls_image_embeddings)
        z_img_norm = F.normalize(prj_image_embeddings, dim=-1)

        cls_logits = self.classifier(z_img_norm)
        return logits_i2t, cls_logits


def _build_train_loader(
    root_dir: str,
    split_file: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    num_samples: Optional[int],
    drop_last: bool,
) -> tuple[DataLoader, List[str]]:
    tfm = transforms.Compose(
        [
            transforms.Resize((40, 40)),
            ToFloat(),
            ZScoreNormalize(
                BigEarthNetInfo.STATISTICS["mean"],
                BigEarthNetInfo.STATISTICS["std"],
            ),
        ]
    )
    ds = EuroSATDatasetLS(
        root_dir=root_dir,
        split_file=split_file,
        transform=tfm,
        return_one_hot=False,
        strict=False,
    )

    sampler = None
    shuffle = True
    if num_samples is not None:
        n = min(int(num_samples), len(ds))
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=n, replace=False)
        sampler = SubsetRandomSampler(indices.tolist())
        shuffle = False

    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 4

    loader = DataLoader(**kwargs)
    return loader, ds.classes


@torch.no_grad()
def _eval_avg_loss(
    model: EuroSATHybridVLM,
    loader: DataLoader,
    *,
    class_names: List[str],
    prompt_template: str,
    device: torch.device,
    clip_weight: float,
    cls_weight: float,
    label_smoothing: float,
) -> float:
    model.eval()
    running = 0.0
    n_steps = 0
    for batch in loader:
        tb = _prepare_batch(
            batch,
            device=device,
            class_names=class_names,
            prompt_template=prompt_template,
        )
        logits_i2t, cls_logits = model(tb.prompts, tb.imgs, tb.meta)
        pos_mask = positives_from_class(tb.labels)
        loss_clip = clip_multilabel_loss(logits_i2t, pos_mask)
        loss_cls = F.cross_entropy(
            cls_logits, tb.labels, label_smoothing=label_smoothing
        )
        loss = clip_weight * loss_clip + cls_weight * loss_cls
        running += float(loss.item())
        n_steps += 1
    return running / max(1, n_steps)


def train(
    model: EuroSATHybridVLM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    class_names: List[str],
    prompt_template: str,
    device: str,
    epochs: int,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
    cls_weight: float,
    clip_weight: float,
    label_smoothing: float,
    amp: bool,
    log_every: int,
    save_dir: str,
    save_name: str,
    save_last: bool,
    args: argparse.Namespace,
) -> EuroSATHybridVLM:
    dev = torch.device(device)
    device_type = _device_type_str(dev)
    use_amp = bool(amp and device_type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device_type == "cuda" else None
    optimizer, params = _build_optimizer(model, lr, weight_decay, encoder_lr)

    best_metric = float("inf")
    best_path = os.path.join(save_dir, f"{save_name}_best.pt")
    last_path = os.path.join(save_dir, f"{save_name}_last.pt")
    step = 0

    for ep in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}")

        for batch in pbar:
            tb = _prepare_batch(
                batch,
                device=dev,
                class_names=class_names,
                prompt_template=prompt_template,
            )

            optimizer.zero_grad(set_to_none=True)

            if device_type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            else:
                autocast_ctx = torch.autocast(device_type=device_type, enabled=False)

            with autocast_ctx:
                logits_i2t, cls_logits = model(tb.prompts, tb.imgs, tb.meta)
                pos_mask = positives_from_class(tb.labels)
                loss_clip = clip_multilabel_loss(logits_i2t, pos_mask)
                loss_cls = F.cross_entropy(
                    cls_logits, tb.labels, label_smoothing=label_smoothing
                )
                loss = clip_weight * loss_clip + cls_weight * loss_cls

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            running += float(loss.item())

            if step % log_every == 0:
                with torch.no_grad():
                    cls_acc = (cls_logits.argmax(dim=1) == tb.labels).float().mean().item()
                    pred_txt_idx = logits_i2t.argmax(dim=1)
                    r1_i2t_class = (
                        tb.labels[pred_txt_idx].eq(tb.labels).float().mean().item()
                    )
                    pred_img_idx = logits_i2t.t().argmax(dim=1)
                    r1_t2i_class = (
                        tb.labels[pred_img_idx].eq(tb.labels).float().mean().item()
                    )
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    loss_cls=f"{loss_cls.item():.4f}",
                    loss_clip=f"{loss_clip.item():.4f}",
                    cls_acc=f"{cls_acc:.3f}",
                    R1_i2t_cls=f"{r1_i2t_class:.3f}",
                    R1_t2i_cls=f"{r1_t2i_class:.3f}",
                    best=f"{best_metric:.4f}" if best_metric < float("inf") else "inf",
                )
            step += 1

        train_loss = running / max(1, len(train_loader))

        val_loss = None
        if val_loader is not None:
            val_loss = _eval_avg_loss(
                model,
                val_loader,
                class_names=class_names,
                prompt_template=prompt_template,
                device=dev,
                clip_weight=clip_weight,
                cls_weight=cls_weight,
                label_smoothing=label_smoothing,
            )
            model.train()

        metric = val_loss if val_loss is not None else train_loss
        if val_loss is None:
            print(f"Epoch {ep} finished - train_loss={train_loss:.4f}")
        else:
            print(
                f"Epoch {ep} finished - train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

        if metric < best_metric:
            best_metric = metric
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=ep,
                best_loss=best_metric,
                class_names=class_names,
                args=args,
            )
            print(f"Saved BEST checkpoint: {best_path}")

        if save_last:
            _save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=ep,
                best_loss=best_metric,
                class_names=class_names,
                args=args,
            )
            print(f"Saved LAST checkpoint: {last_path}")

    return model


def main():
    ap = argparse.ArgumentParser(
        description="Train EuroSAT hybrid VLM: single-label CE + text contrastive."
    )
    ap.add_argument("--root_dir", required=True, help="EuroSAT root with class subfolders.")
    ap.add_argument("--split_file", required=True, help="Train split txt file.")
    ap.add_argument(
        "--val_split_file",
        default="",
        help="Optional validation split txt. If set, best checkpoint is selected by val loss.",
    )
    ap.add_argument("--moe_ckpt", required=True, help="Pretrained MoE-MAE checkpoint.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--drop_last", action="store_true")

    ap.add_argument("--text_backend", choices=["sbert", "openclip"], default="openclip")
    ap.add_argument("--text_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--text_dim", type=int, default=384)
    ap.add_argument("--openclip_model", default="ViT-B-32")
    ap.add_argument("--openclip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--pool", choices=["cls", "mean_patches", "mean_all"], default="cls")
    ap.add_argument("--proj_hidden", type=int, default=512)
    ap.add_argument("--proj_layernorm", action="store_true")

    ap.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--freeze_text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--unfreeze_last_encoder_layers", type=int, default=0)

    ap.add_argument("--temp_init", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--clip_weight", type=float, default=0.3)
    ap.add_argument("--cls_weight", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--prompt_template", default="a multispectral satellite image showing {}")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--save_name", default="vlm_eurosat_hybrid")
    ap.add_argument(
        "--save_last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {args.device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, class_names = _build_train_loader(
        root_dir=args.root_dir,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        num_samples=args.num_samples,
        drop_last=args.drop_last,
    )
    print(f"Loaded {len(class_names)} EuroSAT classes: {class_names}")

    val_loader = None
    if str(args.val_split_file).strip():
        val_loader, val_classes = _build_train_loader(
            root_dir=args.root_dir,
            split_file=args.val_split_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            num_samples=None,
            drop_last=False,
        )
        if list(val_classes) != list(class_names):
            raise ValueError("Train/val class ordering mismatch in EuroSAT folders.")
        print(f"Validation split enabled: {args.val_split_file}")

    # Init model
    encoder = build_model(size="S", img_size=40, patch_size=4, in_chans=7)
    moe = MOEMAE(encoder).to(args.device)
    moe = load_model(moe, args.moe_ckpt, args.device)
    encoder = moe.encoder

    text_encoder_str = "bert"
    base_vlm = VLM(encoder, text_encoder_str, args.temp_init, device=args.device, freeze_encoders=True).to(args.device)
    _configure_encoder_unfreeze(base_vlm, n_last_layers=0)

    model = EuroSATHybridVLM(base_vlm=base_vlm, num_classes=len(class_names)).to(args.device)

    train(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        prompt_template=args.prompt_template,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        cls_weight=args.cls_weight,
        clip_weight=args.clip_weight,
        label_smoothing=args.label_smoothing,
        amp=args.amp,
        log_every=args.log_every,
        save_dir=args.save_dir,
        save_name=args.save_name,
        save_last=args.save_last,
        args=args,
    )


if __name__ == "__main__":
    main()
