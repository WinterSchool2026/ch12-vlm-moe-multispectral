from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.eurosat import EuroSATDatasetLS
from models.moe_mae import build_model
from models.vlm import VLM
from scripts.train_vlm_eurosat_hybrid import EuroSATHybridVLM
from transformation.transformer import ToFloat, ZScoreNormalize
from utils.data_config import BigEarthNetInfo


def import_analysis_utils(path: str):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    return __import__(os.path.splitext(os.path.basename(path))[0])


def _meta_zeros(batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    z = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
    return {"week": z.clone(), "hour": z.clone(), "lat": z.clone(), "lon": z.clone()}


def _to_rgb_fallback(img: torch.Tensor) -> np.ndarray:
    x = img.detach().cpu().numpy()
    c = x.shape[0]
    if c >= 4:
        rgb = x[[3, 2, 1], :, :]
    elif c >= 3:
        rgb = x[[2, 1, 0], :, :]
    else:
        rgb = np.repeat(x[:1, :, :], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    lo = np.percentile(rgb, 2.0)
    hi = np.percentile(rgb, 98.0)
    rgb = (rgb - lo) / max(1e-6, hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def _plot_routing(
    rgb: np.ndarray,
    assign_map: np.ndarray,
    patch_size: int,
    title: str,
    n_experts: Optional[int] = None,
):
    hp, wp = assign_map.shape
    if n_experts is None:
        n_experts = int(assign_map.max()) + 1
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.gca()
    ax.imshow(rgb)
    ys, xs = np.mgrid[0:hp, 0:wp]
    xs = (xs + 0.5) * patch_size
    ys = (ys + 0.5) * patch_size
    cmap = plt.get_cmap("tab20", n_experts)
    norm = matplotlib.colors.BoundaryNorm(
        np.arange(-0.5, n_experts + 0.5, 1), cmap.N
    )
    ax.scatter(
        xs.flatten(),
        ys.flatten(),
        c=assign_map.flatten(),
        s=24,
        cmap=cmap,
        norm=norm,
        edgecolors="k",
        linewidths=0.25,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(n_experts))
    cbar.set_label("expert id")
    ax.set_title(title)
    ax.axis("off")
    return fig


def _plot_heat(
    patch_map: np.ndarray,
    patch_size: int,
    title: str,
    cmap: str = "turbo",
):
    heat = np.kron(patch_map, np.ones((patch_size, patch_size), dtype=np.float32))
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.gca()
    im = ax.imshow(heat, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
    return fig


def _plot_expert_bars(scores: List[float], title: str):
    fig = plt.figure(figsize=(6.2, 2.8))
    ax = plt.gca()
    ax.bar(np.arange(len(scores)), scores)
    ax.set_title(title)
    ax.set_xlabel("expert id")
    ax.set_ylabel("score")
    return fig


def _try_plot_routing_with_analysis_utils(
    au,
    rgb: np.ndarray,
    assign_map: np.ndarray,
    patch_size: int,
    title: str,
    out_path: str,
) -> bool:
    fn = getattr(au, "plot_routing_assignment_with_dots", None)
    if fn is None:
        return False
    try:
        fn(
            base_rgb=rgb,
            assignment_map=assign_map,
            patch_size=patch_size,
            output_path=out_path,
            title=title,
        )
        return True
    except Exception:
        return False


def _load_layer_expert_names(path: str) -> Dict[int, Dict[int, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}

    out: Dict[int, Dict[int, str]] = {}
    if isinstance(obj, dict):
        for lk, lv in obj.items():
            if not str(lk).isdigit() or not isinstance(lv, dict):
                continue
            layer = int(lk)
            names = lv.get("names")
            if not isinstance(names, list):
                continue
            out[layer] = {}
            for e, cand in enumerate(names):
                name = None
                if isinstance(cand, list) and len(cand) > 0:
                    first = cand[0]
                    if isinstance(first, list) and len(first) > 0:
                        name = str(first[0])
                    elif isinstance(first, str):
                        name = first
                if name:
                    out[layer][e] = name
    return out


@st.cache_resource(show_spinner=False)
def _load_model(ckpt_path: str, device_str: str) -> Tuple[EuroSATHybridVLM, Dict]:
    device = torch.device(device_str)
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


def _build_dataset(root_dir: str, split_file: str) -> EuroSATDatasetLS:
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


@torch.no_grad()
def _encode_gallery(
    model: EuroSATHybridVLM,
    ds: EuroSATDatasetLS,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    z_all: List[torch.Tensor] = []
    y_all: List[int] = []
    idx_all: List[int] = []
    offset = 0
    for imgs, labels in tqdm(loader, desc="Encoding gallery"):
        bsz = imgs.size(0)
        imgs = imgs.to(device).float()
        meta = _meta_zeros(batch_size=bsz, device=device)
        z = model.base_vlm.encode_images(imgs, meta["week"], meta["hour"], meta["lat"], meta["lon"]).detach().cpu()
        z_all.append(z)
        y_all.extend(labels.tolist())
        idx_all.extend(list(range(offset, offset + bsz)))
        offset += bsz
    z = torch.cat(z_all, dim=0)
    return z, y_all, idx_all


def _cache_path(cache_dir: str, ckpt: str, split_file: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    ckpt_p = Path(ckpt).resolve()
    split_p = Path(split_file).resolve()
    st = ckpt_p.stat() if ckpt_p.exists() else None
    ckpt_sig = (
        f"{ckpt_p}:{st.st_size}:{st.st_mtime_ns}"
        if st is not None
        else f"{ckpt_p}:missing"
    )
    key = f"{ckpt_sig}::{split_p}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return p / f"eurosat_gallery_cache_{h}.pt"


@torch.no_grad()
def _interpret_layer(
    *,
    au,
    base_vlm: GeoMoEMAETextAlign,
    ds: EuroSATDatasetLS,
    ds_idx: int,
    layer: int,
    query: str,
    device: torch.device,
) -> Dict:
    img, label = ds[ds_idx]
    imgs = img.unsqueeze(0).to(device).float()
    meta = _meta_zeros(batch_size=1, device=device)

    make_rgb = getattr(au, "to_rgb_landsat", None) or getattr(au, "make_rgb", None)
    if make_rgb is not None:
        rgb = make_rgb(
            img,
            max_values=np.array([65454.0, 65454.0, 65330.308], dtype=np.float32),
            bands=(3, 2, 1),
        )
    else:
        rgb = _to_rgb_fallback(img)
    z_img = base_vlm.encode_images(imgs, meta["week"], meta["hour"], meta["lat"], meta["lon"]).detach().cpu().squeeze(0)
    z_q = base_vlm.encode_text([query]).detach().cpu().squeeze(0)
    sim = float((z_img @ z_q).item())

    det = au.deterministic_routing(base_vlm)
    if hasattr(det, "__enter__") and hasattr(det, "__exit__"):
        ctx = det
    else:
        @contextlib.contextmanager
        def _wrap_gen(g):
            try:
                next(g)
                yield
            finally:
                try:
                    next(g)
                except StopIteration:
                    pass
        ctx = _wrap_gen(det)

    with ctx:
        if not hasattr(au, "encoder_layer_states"):
            raise RuntimeError("analysis_utils must provide encoder_layer_states(...)")

        states_by_layer, _, nmeta = au.encoder_layer_states(
            base_vlm.image_encoder,
            imgs,
            meta["week"],
            meta["hour"],
            meta["lat"],
            meta["lon"],
        )
        st_layer = states_by_layer[layer]
        x = st_layer["z_pre_moe"]
        residual = st_layer["x_after_attn"]
        moe = base_vlm.image_encoder.layers[layer].moe
        patch_grid = imgs.shape[-1] // base_vlm.image_encoder.patch_size
        patch_size = base_vlm.image_encoder.patch_size

        assign_maps, _usage = au.routing_assign_and_usage_for_layer(
            moe, x, patch_grid, num_meta_tokens=nmeta, has_cls=True
        )
        assign_map = np.asarray(assign_maps[0], dtype=np.int64)

        contrib = au.get_expert_contributions_for_layer(moe, x)
        maps = au.tokens_to_patch_maps(contrib, patch_grid, num_meta_tokens=nmeta)[0]
        means = [float(np.mean(m)) for m in maps]

        tail = (
            base_vlm.image_encoder.layers[layer + 1 :]
            if (layer + 1) < len(base_vlm.image_encoder.layers)
            else None
        )
        ab = au.compute_ablation_heatmaps(
            moe,
            x,
            patch_grid,
            num_meta_tokens=nmeta,
            propagate_through=tail,
            layer_norm=base_vlm.image_encoder.norm,
            residual_base_tokens=residual,
        )
        ab_np = np.asarray(ab)[0]
        ab_means = [float(np.mean(ab_np[e])) for e in range(ab_np.shape[0])]

    return {
        "rgb": rgb,
        "label": int(label),
        "sim": sim,
        "assign_map": assign_map,
        "patch_size": patch_size,
        "maps": maps,
        "means": means,
        "ab_np": ab_np,
        "ab_means": ab_means,
        "n_experts": len(maps),
    }


def main():
    ap = argparse.ArgumentParser(description="EuroSAT retrieval + expert visualization app")
    ap.add_argument("--analysis_utils", default="utils/analysis_utils.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_dir", default=".cache_vlm_app_eurosat")
    ap.add_argument("--gallery_bs", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--layer_names_json",
        default="",
        help="Optional expert naming JSON from run_expert_naming_semantic.py",
    )
    args, _unknown = ap.parse_known_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)

    st.set_page_config(page_title="EuroSAT Retrieval + Experts", layout="wide")
    st.title("EuroSAT Text Retrieval + Expert Visualization")
    st.caption(f"Device: {device}")

    au = import_analysis_utils(args.analysis_utils)
    model, ckpt = _load_model(args.ckpt, str(device))
    class_names = ckpt["class_names"]
    ds = _build_dataset(args.root_dir, args.split_file)
    layer_expert_names = _load_layer_expert_names(args.layer_names_json)

    n_layers = len(model.base_vlm.image_encoder.layers)
    layer_options = list(range(n_layers))

    with st.sidebar:
        st.header("Query")
        query_bank = [
            "residential area",
            "dense urban neighborhood",
            "industrial buildings",
            "highway",
            "forest",
            "river",
            "sea or lake",
            "pasture land",
            "annual crop fields",
            "permanent crop plantations",
            "herbaceous vegetation",
            "hedges"
        ]
        dropdown_q = st.selectbox("Pick a query", query_bank, index=0)
        free_q = st.text_input("Or type your own query (overrides dropdown)", value="")
        query = free_q.strip() if free_q.strip() else dropdown_q
        topk = st.slider("Top-K", min_value=1, max_value=30, value=12)
        class_filter = st.selectbox("Filter by class", ["All"] + class_names, index=0)
        st.divider()
        st.header("Expert View")
        compare_mode = st.checkbox("Compare multiple layers", value=False)
        if compare_mode:
            all_layers = st.checkbox("Use all model layers", value=False)
            if all_layers:
                sel_layers = layer_options
            else:
                sel_layers = st.multiselect(
                    "Layers",
                    layer_options,
                    default=[0],
                    help="Choose one or multiple layers.",
                )
        else:
            sel_layers = [
                st.selectbox(
                    "Active layer",
                    layer_options,
                    index=0,
                    help="Single-layer view is easier to inspect.",
                )
            ]
        rebuild = st.button("Rebuild cache")

    cpath = _cache_path(args.cache_dir, args.ckpt, args.split_file)
    if rebuild and cpath.exists():
        cpath.unlink()

    if cpath.exists():
        cache = torch.load(cpath, map_location="cpu")
        z = cache["Z"]
        y_all = cache["y_all"]
        idx_all = cache["idx_all"]
    else:
        with st.spinner("Encoding gallery and building cache..."):
            z, y_all, idx_all = _encode_gallery(
                model,
                ds,
                device=device,
                batch_size=args.gallery_bs,
                num_workers=args.num_workers,
            )
            torch.save({"Z": z, "y_all": y_all, "idx_all": idx_all}, cpath)

    st.caption(f"Gallery size: {len(idx_all)} | Cache: {cpath}")
    if not query.strip():
        st.stop()

    with torch.no_grad():
        zq = model.base_vlm.encode_text([query]).detach().cpu().squeeze(0)
        if z.ndim != 2 or z.shape[1] != zq.numel():
            with st.spinner("Cache embedding dimension mismatch. Rebuilding gallery cache..."):
                z, y_all, idx_all = _encode_gallery(
                    model,
                    ds,
                    device=device,
                    batch_size=args.gallery_bs,
                    num_workers=args.num_workers,
                )
                torch.save({"Z": z, "y_all": y_all, "idx_all": idx_all}, cpath)
        sims = z @ zq

    if class_filter != "All":
        keep = [i for i, y in enumerate(y_all) if class_names[int(y)] == class_filter]
    else:
        keep = list(range(len(y_all)))
    if not keep:
        st.warning("No samples for selected filter.")
        st.stop()

    keep_t = torch.tensor(keep, dtype=torch.long)
    sims_f = sims[keep_t]
    k = min(topk, sims_f.numel())
    top_sim, top_pos_local = torch.topk(sims_f, k=k)
    top_pos = keep_t[top_pos_local].tolist()

    st.subheader(f"Top-{k} for: {query}")
    cols = st.columns(4)
    for rank, (glob_pos, score) in enumerate(zip(top_pos, top_sim.tolist()), start=1):
        ds_idx = idx_all[glob_pos]
        label_idx = y_all[glob_pos]
        label_name = class_names[int(label_idx)]
        img, _ = ds[ds_idx]
        rgb = _to_rgb_fallback(img)
        path, _ = ds.samples[ds_idx]
        with cols[(rank - 1) % 4]:
            st.image(rgb, caption=f"#{rank} sim={score:.3f} | {label_name}", use_container_width=True)
            st.caption(str(path))

    selected_rank = st.radio(
        "Select a retrieved sample to inspect",
        list(range(1, k + 1)),
        horizontal=True,
        index=0,
    )
    chosen_global = top_pos[selected_rank - 1]
    chosen_ds_idx = idx_all[chosen_global]
    chosen_label = class_names[int(y_all[chosen_global])]
    st.write(f"Selected sample: `ds_idx={chosen_ds_idx}` | class=`{chosen_label}`")

    if not sel_layers:
        st.info("Pick at least one layer to visualize.")
        st.stop()

    st.divider()
    st.subheader("Expert Visualization")
    with st.spinner("Computing routing/contribution/ablation for selected layers..."):
        layer_out = {}
        for layer in sel_layers:
            layer_out[int(layer)] = _interpret_layer(
                au=au,
                base_vlm=model.base_vlm,
                ds=ds,
                ds_idx=int(chosen_ds_idx),
                layer=int(layer),
                query=query,
                device=device,
            )

    tabs = st.tabs([f"Layer {l}" for l in sel_layers])
    for tab, layer in zip(tabs, sel_layers):
        out = layer_out[int(layer)]
        maps = out["maps"]
        means = out["means"]
        ab_np = out["ab_np"]

        with tab:
            e = int(np.argmax(means))
            layer_name_map = layer_expert_names.get(int(layer), {})
            exp_name = layer_name_map.get(e, f"expert_{e}")
            top_routing_expert = int(np.bincount(out["assign_map"].reshape(-1)).argmax())
            top_routing_name = layer_name_map.get(
                top_routing_expert, f"expert_{top_routing_expert}"
            )

            st.caption(
                f"Layer={layer} | sim(query)={out['sim']:.3f} | selected expert={e} ({exp_name})"
            )
            st.write(
                f"Routing top-1 dominant expert: `{top_routing_expert}` ({top_routing_name})"
            )

            left, right = st.columns([1, 1])
            with left:
                st.image(
                    out["rgb"],
                    caption=f"Selected tile | class={class_names[out['label']]} | sim={out['sim']:.3f}",
                    use_container_width=True,
                )
            with right:
                routing_title = f"Routing top-1 (L{layer}) | dom={top_routing_expert} ({top_routing_name})"
                qhash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
                tmp = (
                    Path(args.cache_dir)
                    / f"tmp_routing_eurosat_L{layer}_{chosen_ds_idx}_{qhash}.png"
                )
                tmp.parent.mkdir(parents=True, exist_ok=True)
                ok = _try_plot_routing_with_analysis_utils(
                    au,
                    out["rgb"],
                    out["assign_map"],
                    out["patch_size"],
                    routing_title,
                    str(tmp),
                )
                if ok and tmp.exists():
                    st.image(str(tmp), caption="Routing top-1 (analysis_utils)", use_container_width=True)
                else:
                    fig = _plot_routing(
                        out["rgb"],
                        out["assign_map"],
                        out["patch_size"],
                        routing_title,
                        n_experts=out["n_experts"],
                    )
                    st.pyplot(fig, clear_figure=True)

            top2 = np.argsort(means)[::-1][:2].tolist()
            contrib_stack = np.stack([np.asarray(m, np.float32) for m in maps], axis=0)
            ab_stack = np.asarray(ab_np, np.float32)
            c_min, c_max = float(contrib_stack.min()), float(contrib_stack.max())
            a_min, a_max = float(ab_stack.min()), float(ab_stack.max())
            if np.isclose(c_min, c_max):
                c_max = c_min + 1e-6
            if np.isclose(a_min, a_max):
                a_max = a_min + 1e-6

            st.caption(
                f"Shared ranges (layer {layer}) | "
                f"Contribution [{c_min:.4f}, {c_max:.4f}] | "
                f"Ablation Δ [{a_min:.4f}, {a_max:.4f}]"
            )

            c0, c1, c2, c3 = st.columns(4)
            with c0:
                st.image(out["rgb"], caption=f"RGB | layer {layer}", use_container_width=True)
            with c1:
                e1 = int(top2[0])
                n1 = layer_name_map.get(e1, f"expert_{e1}")
                fig = _plot_heat(
                    np.asarray(maps[e1]),
                    out["patch_size"],
                    f"Contribution | e={e1} ({n1})",
                )
                st.pyplot(fig, clear_figure=True)
            with c2:
                if len(top2) > 1:
                    e2 = int(top2[1])
                    n2 = layer_name_map.get(e2, f"expert_{e2}")
                    fig = _plot_heat(
                        np.asarray(maps[e2]),
                        out["patch_size"],
                        f"Contribution | e={e2} ({n2})",
                    )
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Only one expert available.")
            with c3:
                fig = _plot_heat(
                    np.asarray(ab_np[int(top2[0])]),
                    out["patch_size"],
                    f"Ablation Δ | e={int(top2[0])}",
                )
                st.pyplot(fig, clear_figure=True)

            st.subheader(f"Expert attribution scores (layer {layer})")
            b1, b2 = st.columns(2)
            with b1:
                fig = _plot_expert_bars(means, f"Mean contribution per expert (L{layer})")
                st.pyplot(fig, clear_figure=True)
            with b2:
                fig = _plot_expert_bars(out["ab_means"], f"Mean ablation Δ per expert (L{layer})")
                st.pyplot(fig, clear_figure=True)

            st.subheader(f"Per-expert maps (layer {layer})")
            n_experts = out["n_experts"]
            expert_options = ["Top-1 contribution"] + [str(i) for i in range(n_experts)]
            default_choice = st.session_state.get(
                f"expert_choice_layer_{layer}", "Top-1 contribution"
            )
            with st.form(key=f"expert_form_layer_{layer}", clear_on_submit=False):
                expert_choice = st.selectbox(
                    f"Expert for per-expert maps (layer {layer})",
                    expert_options,
                    index=expert_options.index(default_choice)
                    if default_choice in expert_options
                    else 0,
                    key=f"expert_choice_layer_{layer}",
                )
                st.form_submit_button("Update per-expert maps")
            if expert_choice == "Top-1 contribution":
                e_sel = int(np.argmax(means))
            else:
                e_sel = int(expert_choice)
            exp_name_sel = layer_name_map.get(e_sel, f"expert_{e_sel}")
            d0, d1, d2 = st.columns(3)
            with d0:
                st.image(out["rgb"], caption="RGB", use_container_width=True)
            with d1:
                fig = _plot_heat(
                    np.asarray(maps[e_sel]),
                    out["patch_size"],
                    f"Contribution (selected e={e_sel}, {exp_name_sel})",
                )
                st.pyplot(fig, clear_figure=True)
            with d2:
                fig = _plot_heat(
                    np.asarray(ab_np[e_sel]),
                    out["patch_size"],
                    f"Ablation Δ (selected e={e_sel}, {exp_name_sel})",
                )
                st.pyplot(fig, clear_figure=True)

            st.write(
                f"Top contribution expert for layer {layer}: `{int(np.argmax(means))}` | "
                f"Top ablation expert: `{int(np.argmax(out['ab_means']))}`"
            )


if __name__ == "__main__":
    main()
