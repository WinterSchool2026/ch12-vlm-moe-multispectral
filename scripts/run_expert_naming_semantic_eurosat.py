from __future__ import annotations
from debugpy.common.json import default

import argparse
import heapq
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.eurosat import EuroSATDatasetLS
from models.moe_mae import MOEMAE, build_model
from models.vlm import VLM
from scripts.train_vlm_eurosat_hybrid import EuroSATHybridVLM
from transformation.transformer import ToFloat, ZScoreNormalize
from utils.data_config import BigEarthNetInfo
from utils.data_utils import load_model


def import_analysis_utils(path: str):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    return __import__(os.path.splitext(os.path.basename(path))[0])


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _normalize_class_text(name: str) -> str:
    text = name.strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = " ".join(text.split()).lower()
    if text == "sea lake":
        return "sea or lake"
    return text


def canonicalize_query_label(q: str) -> str:
    s = q.strip()
    prefixes = [
        "satellite view of ",
        "remote sensing image of ",
        "aerial image of ",
    ]
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p) :]
            break
    if s.endswith(" landscape"):
        s = s[: -len(" landscape")]
    return s.strip()


def build_dataset(root_dir: str, split_file: str) -> EuroSATDatasetLS:
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
    return EuroSATDatasetLS(
        root_dir=root_dir,
        split_file=split_file,
        transform=tfm,
        return_one_hot=False,
        strict=False,
    )


def load_hybrid_model(
    ckpt_path: str, device: torch.device
) -> Tuple[EuroSATHybridVLM, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]

    encoder = build_model(size="S", img_size=40, patch_size=4, in_chans=7)

    base = VLM(
        encoder, 
        "bert", 
        0.07, 
        device=device, 
        freeze_encoders=True
        ).to(device)

    model = EuroSATHybridVLM(base_vlm=base, num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


class EuroSATAdapterDataset:
    """
    Adapter to match expert-prototype function expectations:
    returns (img, y, week, hour, lat, lon).
    """

    def __init__(self, ds: EuroSATDatasetLS):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, y = self.ds[int(idx)]
        z = torch.zeros(2, dtype=torch.float32)
        return (
            img,
            torch.tensor([int(y)], dtype=torch.float32),
            z.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
        )


@torch.no_grad()
def compute_expert_prototypes_top1_share(
    *,
    au,
    model: VLM,
    dataset,
    ds_idx: List[int],
    layer: int,
    device: torch.device,
    bs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder_layer_states = getattr(au, "encoder_layer_states", None)
    encoder_tokens_per_layer = getattr(au, "encoder_tokens_per_layer", None)
    routing_assign_and_usage_for_layer = getattr(
        au, "routing_assign_and_usage_for_layer"
    )
    if encoder_layer_states is None and encoder_tokens_per_layer is None:
        raise RuntimeError(
            "analysis_utils must provide encoder_layer_states(...) or encoder_tokens_per_layer(...)."
        )

    Z_e: Optional[torch.Tensor] = None
    mass: Optional[torch.Tensor] = None

    for i in tqdm(range(0, len(ds_idx), bs), desc=f"top1-share prototypes L{layer}"):
        batch_ids = ds_idx[i : i + bs]
        imgs, mw, mh, mlat, mlon = [], [], [], [], []
        for ds in batch_ids:
            img, _y, w, h, la, lo = dataset[int(ds)]
            imgs.append(img)
            mw.append(w)
            mh.append(h)
            mlat.append(la)
            mlon.append(lo)

        imgs_t = torch.stack(imgs, dim=0).to(device).float()
        meta = dict(
            week=torch.stack(mw).to(device).float(),
            hour=torch.stack(mh).to(device).float(),
            lat=torch.stack(mlat).to(device).float(),
            lon=torch.stack(mlon).to(device).float(),
        )

        z_img = model.encode_images(imgs_t, meta["week"], meta["hour"], meta["lat"], meta["lon"]).detach().cpu()

        with au.deterministic_routing(model):
            if encoder_layer_states is not None:
                states_by_layer, _, nmeta = encoder_layer_states(
                    model.encoder,
                    imgs_t,
                    meta["week"],
                    meta["hour"],
                    meta["lat"],
                    meta["lon"],
                )
                x = states_by_layer[layer]["z_pre_moe"]
            else:
                tokens_by_layer, _, nmeta = encoder_tokens_per_layer(
                    model.encoder,
                    imgs_t,
                    meta["week"],
                    meta["hour"],
                    meta["lat"],
                    meta["lon"],
                )
                x = tokens_by_layer[layer]

            moe = model.encoder.layers[layer].moe
            pg = imgs_t.shape[-1] // model.encoder.patch_size
            assign_maps, _usage = routing_assign_and_usage_for_layer(
                moe, x, pg, num_meta_tokens=nmeta, has_cls=True
            )
            assign_maps = np.asarray(assign_maps)
            bsz, hp, wp = assign_maps.shape
            n_experts = int(getattr(moe, "num_experts"))
            shares = torch.zeros(bsz, n_experts, dtype=torch.float32)
            denom = float(hp * wp)
            for b in range(bsz):
                a = assign_maps[b].reshape(-1)
                for e in range(n_experts):
                    shares[b, e] = float((a == e).sum()) / denom

        if Z_e is None:
            d = z_img.shape[1]
            Z_e = torch.zeros(n_experts, d, dtype=torch.float32)
            mass = torch.zeros(n_experts, dtype=torch.float32)

        Z_e += shares.T @ z_img
        mass += shares.sum(dim=0)

    assert Z_e is not None and mass is not None
    Z_e = Z_e / Z_e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return Z_e, mass


@torch.no_grad()
def compute_expert_prototypes_topk_instances(
    *,
    au,
    model: VLM,
    dataset,
    ds_idx: List[int],
    layer: int,
    device: torch.device,
    bs: int,
    topk_instances: int = 200,
    score_source: str = "top1_share",
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder_layer_states = getattr(au, "encoder_layer_states", None)
    encoder_tokens_per_layer = getattr(au, "encoder_tokens_per_layer", None)
    routing_assign_and_usage_for_layer = getattr(
        au, "routing_assign_and_usage_for_layer", None
    )
    if encoder_layer_states is None and encoder_tokens_per_layer is None:
        raise RuntimeError(
            "analysis_utils must provide encoder_layer_states(...) or encoder_tokens_per_layer(...)."
        )

    heaps = None
    counter = 0

    for i in tqdm(
        range(0, len(ds_idx), bs), desc=f"topk-instance prototypes L{layer}"
    ):
        batch_ids = ds_idx[i : i + bs]
        imgs, mw, mh, mlat, mlon = [], [], [], [], []
        for ds in batch_ids:
            img, _y, w, h, la, lo = dataset[int(ds)]
            imgs.append(img)
            mw.append(w)
            mh.append(h)
            mlat.append(la)
            mlon.append(lo)

        imgs_t = torch.stack(imgs, dim=0).to(device).float()
        meta = dict(
            week=torch.stack(mw).to(device).float(),
            hour=torch.stack(mh).to(device).float(),
            lat=torch.stack(mlat).to(device).float(),
            lon=torch.stack(mlon).to(device).float(),
        )

        z_img = model.encode_images(imgs_t, meta["week"], meta["hour"], meta["lat"], meta["lon"]).detach().cpu()

        with au.deterministic_routing(model):
            if encoder_layer_states is not None:
                states_by_layer, _, nmeta = encoder_layer_states(
                    model.image_encoder,
                    imgs_t,
                    meta["week"],
                    meta["hour"],
                    meta["lat"],
                    meta["lon"],
                )
                x = states_by_layer[layer]["z_pre_moe"]
            else:
                tokens_by_layer, _, nmeta = encoder_tokens_per_layer(
                    model.image_encoder,
                    imgs_t,
                    meta["week"],
                    meta["hour"],
                    meta["lat"],
                    meta["lon"],
                )
                x = tokens_by_layer[layer]

            moe = model.image_encoder.layers[layer].moe
            pg = imgs_t.shape[-1] // model.image_encoder.patch_size
            n_experts = int(getattr(moe, "num_experts"))
            bsz = z_img.shape[0]

            if score_source == "top1_share":
                if routing_assign_and_usage_for_layer is None:
                    raise RuntimeError(
                        "analysis_utils must provide routing_assign_and_usage_for_layer(...) for top1_share."
                    )
                assign_maps, _usage = routing_assign_and_usage_for_layer(
                    moe, x, pg, num_meta_tokens=nmeta, has_cls=True
                )
                assign_maps = np.asarray(assign_maps)
                hp, wp = assign_maps.shape[1], assign_maps.shape[2]
                denom = float(hp * wp)
                scores = torch.zeros(bsz, n_experts, dtype=torch.float32)
                for b in range(bsz):
                    a = assign_maps[b].reshape(-1)
                    for e in range(n_experts):
                        scores[b, e] = float((a == e).sum()) / denom
            else:
                contrib = au.get_expert_contributions_for_layer(moe, x)
                maps = au.tokens_to_patch_maps(contrib, pg, num_meta_tokens=nmeta)
                scores = torch.zeros(bsz, n_experts, dtype=torch.float32)
                for b in range(bsz):
                    vals = [
                        float(np.mean(np.asarray(maps[b][e], dtype=np.float32)))
                        for e in range(n_experts)
                    ]
                    row = torch.tensor(vals, dtype=torch.float32).clamp(min=0.0)
                    s = row.sum().item()
                    if s > 0:
                        row /= s
                    scores[b] = row

        if heaps is None:
            heaps = [[] for _ in range(n_experts)]

        for b in range(bsz):
            emb_np = z_img[b].numpy()
            for e in range(n_experts):
                sc = float(scores[b, e].item())
                item = (sc, counter, emb_np)
                counter += 1
                hp = heaps[e]
                if len(hp) < topk_instances:
                    heapq.heappush(hp, item)
                elif sc > hp[0][0]:
                    heapq.heapreplace(hp, item)

    assert heaps is not None
    n_experts = len(heaps)
    d = int(heaps[0][0][2].shape[0]) if heaps and heaps[0] else int(model.logit_scale.numel())
    Z_e = torch.zeros(n_experts, d, dtype=torch.float32)
    mass = torch.zeros(n_experts, dtype=torch.float32)

    for e in range(n_experts):
        hp = heaps[e]
        if not hp:
            continue
        ws = np.asarray([max(0.0, t[0]) for t in hp], dtype=np.float32)
        xs = np.stack([t[2] for t in hp], axis=0).astype(np.float32)
        s = float(ws.sum())
        if s > 0:
            proto = (ws[:, None] * xs).sum(axis=0) / s
            mass[e] = s
        else:
            proto = xs.mean(axis=0)
            mass[e] = float(len(hp))
        Z_e[e] = torch.from_numpy(proto)

    Z_e = Z_e / Z_e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return Z_e, mass


@torch.no_grad()
def compute_expert_prototypes_patch_weighted(
    *,
    au,
    model: VLM,
    dataset,
    ds_idx: List[int],
    layer: int,
    device: torch.device,
    bs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder_layer_states = getattr(au, "encoder_layer_states", None)
    encoder_tokens_per_layer = getattr(au, "encoder_tokens_per_layer", None)
    if encoder_layer_states is None and encoder_tokens_per_layer is None:
        raise RuntimeError(
            "analysis_utils must provide encoder_layer_states(...) or encoder_tokens_per_layer(...)."
        )

    Z_e: Optional[torch.Tensor] = None
    mass: Optional[torch.Tensor] = None

    for i in tqdm(
        range(0, len(ds_idx), bs), desc=f"patch-weighted prototypes L{layer}"
    ):
        batch_ids = ds_idx[i : i + bs]
        imgs, mw, mh, mlat, mlon = [], [], [], [], []
        for ds in batch_ids:
            img, _y, w, h, la, lo = dataset[int(ds)]
            imgs.append(img)
            mw.append(w)
            mh.append(h)
            mlat.append(la)
            mlon.append(lo)

        imgs_t = torch.stack(imgs, dim=0).to(device).float()
        meta = dict(
            week=torch.stack(mw).to(device).float(),
            hour=torch.stack(mh).to(device).float(),
            lat=torch.stack(mlat).to(device).float(),
            lon=torch.stack(mlon).to(device).float(),
        )

        z_img = model.encode_images(imgs_t, meta["week"], meta["hour"], meta["lat"], meta["lon"]).detach().cpu()

        with au.deterministic_routing(model):
            if encoder_layer_states is not None:
                states_by_layer, _, nmeta = encoder_layer_states(
                    model.encoder,
                    imgs_t,
                    meta["week"],
                    meta["hour"],
                    meta["lat"],
                    meta["lon"],
                )
                x = states_by_layer[layer]["z_pre_moe"]
            else:
                tokens_by_layer, _, nmeta = encoder_tokens_per_layer(
                    model.encoder,
                    imgs_t,
                    meta["week"],
                    meta["hour"],
                    meta["lat"],
                    meta["lon"],
                )
                x = tokens_by_layer[layer]

            moe = model.encoder.layers[layer].moe
            pg = imgs_t.shape[-1] // model.encoder.patch_size
            contrib = au.get_expert_contributions_for_layer(moe, x)
            maps = au.tokens_to_patch_maps(contrib, pg, num_meta_tokens=nmeta)
            bsz = len(maps)
            n_experts = len(maps[0]) if bsz > 0 else int(getattr(moe, "num_experts"))

            weights = torch.zeros(bsz, n_experts, dtype=torch.float32)
            for b in range(bsz):
                vals = [
                    float(np.mean(np.asarray(maps[b][e], dtype=np.float32)))
                    for e in range(n_experts)
                ]
                row = torch.tensor(vals, dtype=torch.float32).clamp(min=0.0)
                s = row.sum().item()
                if s <= 0:
                    row.fill_(1.0 / max(n_experts, 1))
                else:
                    row /= s
                weights[b] = row

        if Z_e is None:
            d = z_img.shape[1]
            Z_e = torch.zeros(n_experts, d, dtype=torch.float32)
            mass = torch.zeros(n_experts, dtype=torch.float32)

        Z_e += weights.T @ z_img
        mass += weights.sum(dim=0)

    assert Z_e is not None and mass is not None
    Z_e = Z_e / Z_e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return Z_e, mass


@torch.no_grad()
def name_experts(
    model: VLM,
    Z_e: torch.Tensor,
    queries: List[str],
    device: torch.device,
    topk: int = 8,
):
    z_txt = model.encode_text(queries, normalize_embeddings=True).detach().cpu()
    sims = Z_e @ z_txt.T
    names = []
    names_raw = []
    names_zscore = []
    names_raw_prompt = []
    names_zscore_prompt = []
    top1_margin = []
    top1_margin_z = []
    uncertain_by_z = []
    names_with_uncertainty = []
    for e in range(sims.size(0)):
        row = sims[e]
        mu = row.mean()
        std = row.std(unbiased=False).clamp(min=1e-6)
        row_z = (row - mu) / std

        vals_r, idx_r = torch.topk(row, k=min(topk, row.numel()))
        vals_z, idx_z = torch.topk(row_z, k=min(topk, row_z.numel()))
        prompt_rank_raw = [
            (queries[i], float(vals_r[j])) for j, i in enumerate(idx_r.tolist())
        ]
        prompt_rank_z = [
            (queries[i], float(vals_z[j])) for j, i in enumerate(idx_z.tolist())
        ]
        names_raw_prompt.append(prompt_rank_raw)
        names_zscore_prompt.append(prompt_rank_z)

        cls_scores_raw = {}
        cls_scores_z = {}
        for qi, q in enumerate(queries):
            c = canonicalize_query_label(q)
            r = float(row[qi].item())
            z = float(row_z[qi].item())
            if c not in cls_scores_raw:
                cls_scores_raw[c] = r
                cls_scores_z[c] = z
            else:
                cls_scores_raw[c] = max(cls_scores_raw[c], r)
                cls_scores_z[c] = max(cls_scores_z[c], z)

        agg_raw = sorted(cls_scores_raw.items(), key=lambda x: x[1], reverse=True)
        agg_z = sorted(cls_scores_z.items(), key=lambda x: x[1], reverse=True)
        names_raw.append([(k, float(v)) for k, v in agg_raw[:topk]])
        names_zscore.append([(k, float(v)) for k, v in agg_z[:topk]])

        names.append(names_raw[-1])

        m_r = (
            float(names_raw[-1][0][1] - names_raw[-1][1][1])
            if len(names_raw[-1]) >= 2
            else 0.0
        )
        m_z = (
            float(names_zscore[-1][0][1] - names_zscore[-1][1][1])
            if len(names_zscore[-1]) >= 2
            else 0.0
        )
        top1_margin.append(m_r)
        top1_margin_z.append(m_z)
        uncertain_by_z.append(False)
        names_with_uncertainty.append(names[-1])

    return {
        "names": names,
        "names_raw": names_raw,
        "names_zscore": names_zscore,
        "names_raw_prompt": names_raw_prompt,
        "names_zscore_prompt": names_zscore_prompt,
        "top1_margin": top1_margin,
        "top1_margin_zscore": top1_margin_z,
        "uncertain_by_zscore_margin": uncertain_by_z,
        "names_with_uncertainty": names_with_uncertainty,
    }


def load_query_bank(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return [x.strip() for x in obj if x.strip()]
    raise ValueError("Query bank JSON must be a list of strings.")


@torch.no_grad()
def compute_routing_label_names(
    *,
    au,
    model: VLM,
    ds_raw: EuroSATDatasetLS,
    ds_idx: List[int],
    layer: int,
    class_names: List[str],
    device: torch.device,
    bs: int,
    topk: int,
) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
    if not hasattr(au, "encoder_layer_states") or not hasattr(
        au, "routing_assign_and_usage_for_layer"
    ):
        raise RuntimeError(
            "analysis_utils must provide encoder_layer_states(...) and routing_assign_and_usage_for_layer(...)."
        )

    class_text = [_normalize_class_text(x) for x in class_names]
    n_classes = len(class_text)
    label_mass = None
    expert_mass = None

    for i in range(0, len(ds_idx), bs):
        batch_ids = ds_idx[i : i + bs]
        imgs = []
        ys = []
        for di in batch_ids:
            img, y = ds_raw[int(di)]
            imgs.append(img)
            ys.append(int(y))
        imgs_t = torch.stack(imgs, dim=0).to(device).float()

        z = torch.zeros(imgs_t.size(0), 2, device=device, dtype=torch.float32)
        meta = {"week": z.clone(), "hour": z.clone(), "lat": z.clone(), "lon": z.clone()}

        with au.deterministic_routing(model):
            states_by_layer, _, nmeta = au.encoder_layer_states(
                model.encoder,
                imgs_t,
                meta["week"],
                meta["hour"],
                meta["lat"],
                meta["lon"],
            )
            x = states_by_layer[layer]["z_pre_moe"]
            moe = model.encoder.layers[layer].moe
            patch_grid = imgs_t.shape[-1] // model.encoder.patch_size
            assign_maps, _usage = au.routing_assign_and_usage_for_layer(
                moe, x, patch_grid, num_meta_tokens=nmeta, has_cls=True
            )

        assign_maps = np.asarray(assign_maps)
        bsz, hp, wp = assign_maps.shape
        n_experts = int(getattr(moe, "num_experts"))
        shares = np.zeros((bsz, n_experts), dtype=np.float32)
        denom = float(hp * wp)
        for b in range(bsz):
            a = assign_maps[b].reshape(-1)
            for e in range(n_experts):
                shares[b, e] = float((a == e).sum()) / denom

        if label_mass is None:
            label_mass = np.zeros((n_experts, n_classes), dtype=np.float32)
            expert_mass = np.zeros((n_experts,), dtype=np.float32)

        for b in range(bsz):
            y = ys[b]
            for e in range(n_experts):
                w = shares[b, e]
                label_mass[e, y] += w
                expert_mass[e] += w

    assert label_mass is not None and expert_mass is not None
    names = []
    for e in range(label_mass.shape[0]):
        row = label_mass[e]
        s = float(row.sum())
        probs = row / s if s > 0 else row
        idx = np.argsort(-probs)[: max(1, topk)]
        names.append([(class_text[int(i)], float(probs[int(i)])) for i in idx])
    return names, expert_mass.tolist()


def main():
    ap = argparse.ArgumentParser(
        description="EuroSAT semantic expert naming (minimal self-contained)."
    )
    ap.add_argument("--analysis_utils", required=False, default="/home/jordi.morales/Desktop/ELLIS Winter School/scripts/utils/analysis_utils.py")
    ap.add_argument("--csv", required=False, default="/home/jordi.morales/Desktop/ELLIS Winter School/data/eurosat-l/eurosat-test.txt")
    ap.add_argument("--root", required=False, default="/home/jordi.morales/Desktop/ELLIS Winter School/data/eurosat-l/eurosat-l")
    ap.add_argument("--vlm_ckpt", required=False, default="weights/vlm_100epochs_best.pt")
    ap.add_argument("--layers", type=int, nargs="+", required=False, default=[1])
    ap.add_argument("--proto_n", type=int, required=False, default=100)
    ap.add_argument("--proto_bs", type=int, required=False, default=128)
    ap.add_argument("--name_topk", type=int, required=False, default=10)
    ap.add_argument(
        "--prototype_mode",
        choices=["top1_share", "patch_weighted", "topk_instances"],
        required=False,
        default="topk_instances"
    )
    ap.add_argument("--topk_instances", type=int, required=False, default=10)
    ap.add_argument(
        "--instance_score_source",
        choices=["top1_share", "contrib_mean"],
        required=False,
        default="contrib_mean"
    )
    ap.add_argument("--query_bank_json", required=False, default="data/query_bank_eurosat_basic.json" )
    ap.add_argument(
        "--naming_source",
        choices=["text", "routing_labels", "hybrid"],
        required=False,
        default="hybrid"
    )
    ap.add_argument("--out", required=False, default="results/eurosat_expert_naming.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    au = import_analysis_utils(args.analysis_utils)
    ds_raw = build_dataset(args.root, args.csv)
    dataset = EuroSATAdapterDataset(ds_raw)
    hybrid_model, ckpt = load_hybrid_model(args.vlm_ckpt, device)
    model = hybrid_model.base_vlm

    all_idx = list(range(len(dataset)))
    random.Random(0).shuffle(all_idx)
    proto_idx = all_idx[: min(args.proto_n, len(all_idx))]

    class_names = list(ckpt["class_names"])
    query_bank = load_query_bank(args.query_bank_json)

    out = {
        "config": {
            "layers": args.layers,
            "proto_n": len(proto_idx),
            "proto_bs": args.proto_bs,
            "name_topk": args.name_topk,
            "prototype_mode": args.prototype_mode,
            "topk_instances": args.topk_instances,
            "instance_score_source": args.instance_score_source,
            "naming_source": args.naming_source,
            "query_bank_size": len(query_bank),
            "ckpt": args.vlm_ckpt,
            "root_dir": args.root,
            "split_file": args.csv,
        },
        "query_bank": query_bank,
        "layers": {},
    }

    for layer in args.layers:
        z_e, mass = compute_expert_prototypes_topk_instances(
            au=au,
            model=model,
            dataset=dataset,
            ds_idx=proto_idx,
            layer=layer,
            device=device,
            bs=args.proto_bs,
            topk_instances=max(1, int(args.topk_instances)),
            score_source=args.instance_score_source,
        )

        naming = name_experts(
            model,
            z_e,
            query_bank,
            device=device,
            topk=args.name_topk,
        )
        """routing_names, routing_mass = compute_routing_label_names(
            au=au,
            model=model,
            ds_raw=ds_raw,
            ds_idx=proto_idx,
            layer=layer,
            class_names=class_names,
            device=device,
            bs=args.proto_bs,
            topk=args.name_topk,
        )"""

        """if args.naming_source == "hybrid":
            final_names = []
            for e in range(len(naming["names"])):
                txt = list(naming["names"][e])
                if routing_names[e]:
                    top_r = routing_names[e][0]
                    txt = [top_r] + [x for x in txt if x[0] != top_r[0]]
                    txt = txt[: args.name_topk]
                final_names.append(txt)
            final_mass = mass.tolist()
        else:
            final_names = routing_names
            final_mass = routing_mass"""

        out["layers"][str(layer)] = {
            #"mass": final_mass,
            #"names": final_names,
            "names_raw": naming["names_raw"],
            "names_zscore": naming["names_zscore"],
            "names_raw_prompt": naming["names_raw_prompt"],
            "names_zscore_prompt": naming["names_zscore_prompt"],
            "top1_margin": naming["top1_margin"],
            "top1_margin_zscore": naming["top1_margin_zscore"],
            "uncertain_by_zscore_margin": naming["uncertain_by_zscore_margin"],
            "names_with_uncertainty": naming["names_with_uncertainty"],
            #"names_routing_labels": routing_names,
        }

    save_json(out, args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
