from contextlib import contextmanager
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils.data_config import BigEarthNetInfo


@contextmanager
def deterministic_routing(model):
    """
    Temporarily zero the noise in NoisyTopKGate for reproducible analysis.
    """
    original_forwards = []
    gates = []

    def make_det_forward(gate):
        def det_forward(x):
            logits = gate.Wg(x)  # [BN, E]
            noise_scale = F.softplus(  # pylint:disable=E1102
                gate.Wnoise(x)
            )  # unused when deterministic
            H = logits  # noise removed
            topk_vals, topk_idx = torch.topk(H, gate.k, dim=-1)
            topk_softmax = F.softmax(topk_vals, dim=-1)
            gates_tensor = x.new_zeros(H.shape)
            gates_tensor.scatter_(-1, topk_idx, topk_softmax)  # sparse gates
            return gates_tensor, H, topk_idx, noise_scale, logits

        return det_forward

    for mod in model.modules():
        if mod.__class__.__name__ == "NoisyTopKGate":
            gates.append(mod)
            original_forwards.append(mod.forward)
            mod.forward = make_det_forward(mod)

    try:
        yield
    finally:
        for g, f in zip(gates, original_forwards):
            g.forward = f


@torch.no_grad()
def encoder_layer_states(
    encoder, imgs, meta_week=None, meta_hour=None, meta_lat=None, meta_lon=None
):
    """
    Rebuilds encoder.forward to capture per-layer states:
      - x_after_attn: residual stream right before norm2/moe
      - z_pre_moe: norm2(x_after_attn), actual input to moe.gate/expert
      - x_out: residual stream after adding moe output

    Returns:
      states_by_layer: list[dict] with tensors [B, N, C]
      total_moe_loss: scalar tensor
      num_meta_tokens: int
    """
    B = imgs.shape[0]
    device = imgs.device

    p = encoder.patch_proj(imgs).flatten(2).transpose(1, 2)  # (B,N,embed_dim)

    if meta_week is None:
        meta_week = torch.zeros(B, 2, device=device)
    if meta_hour is None:
        meta_hour = torch.zeros(B, 2, device=device)
    if meta_lat is None:
        meta_lat = torch.zeros(B, 2, device=device)
    if meta_lon is None:
        meta_lon = torch.zeros(B, 2, device=device)

    week = encoder.week_proj(meta_week).unsqueeze(1)
    hour = encoder.hour_proj(meta_hour).unsqueeze(1)
    lat = encoder.lat_proj(meta_lat).unsqueeze(1)
    lon = encoder.lon_proj(meta_lon).unsqueeze(1)
    meta = torch.cat([week, hour, lat, lon], dim=1)  # (B,M,embed)
    cls = encoder.cls_token.expand(B, -1, -1)

    x = torch.cat([meta, cls, p], dim=1)
    x = x + encoder.pos_embed[:, : x.shape[1], :]

    total_moe_loss = x.new_zeros(1).sum()
    states_by_layer = []

    for layer in encoder.layers:
        y = layer.norm1(x)
        y = layer.attn(y)
        x_after_attn = x + y
        z_pre_moe = layer.norm2(x_after_attn)
        moe_out, l = layer.moe(z_pre_moe)
        x = x_after_attn + moe_out
        total_moe_loss = total_moe_loss + l
        states_by_layer.append(
            {
                "x_after_attn": x_after_attn,
                "z_pre_moe": z_pre_moe,
                "x_out": x,
            }
        )

    x = encoder.norm(x)
    states_by_layer[-1]["x_out"] = x  # keep parity with encoder.forward return

    return states_by_layer, total_moe_loss, encoder.num_meta_tokens


@torch.no_grad()
def encoder_tokens_per_layer(
    encoder, imgs, meta_week=None, meta_hour=None, meta_lat=None, meta_lon=None
):
    """
    Backward-compatible wrapper returning tokens after each layer.
    """
    states_by_layer, total_moe_loss, num_meta_tokens = encoder_layer_states(
        encoder, imgs, meta_week, meta_hour, meta_lat, meta_lon
    )
    tokens_by_layer = [st["x_out"] for st in states_by_layer]
    return tokens_by_layer, total_moe_loss, num_meta_tokens


@torch.no_grad()
def routing_assign_and_usage_for_layer(
    moe_layer, x_tokens, patch_grid, num_meta_tokens=4, has_cls=True
):
    """
    Return:
      assign_maps: [B, H, W] with top-1 expert id per patch (0..E-1)
      usage:       [B, E] number of patch tokens routed to each expert (appears in top-k)
    """
    B, N, C = x_tokens.shape
    flat = x_tokens.reshape(-1, C)
    gates, _, topk_idx, *_ = moe_layer.gate(
        flat
    )  # gates: [BN, E] (nonzero only for top-k)
    E = moe_layer.num_experts

    start = num_meta_tokens + (1 if has_cls else 0)
    N_patch = patch_grid * patch_grid

    # Top-1 expert per patch (argmax over gate probs)
    gates_np = gates.detach().cpu().numpy().reshape(B, N, E)
    assign_maps = []
    usage = []

    for b in range(B):
        Gp = gates_np[b, start : start + N_patch]  # [N_patch, E]
        a = Gp.argmax(axis=-1).reshape(patch_grid, patch_grid)
        assign_maps.append(a)

        row = topk_idx[b * N + start : b * N + start + N_patch]  # [N_patch, k]
        cnt = np.zeros(E, dtype=int)
        for e in range(E):
            cnt[e] = (row == e).any(dim=1).sum().item()
        usage.append(cnt)

    return np.stack(assign_maps, 0), np.stack(usage, 0)  # [B,H,W], [B,E]


def plot_expert_usage_bars(usage, title="Expert usage"):
    """
    usage: [B, E] token counts per expert (patch tokens only)
    """
    arr = np.asarray(usage)
    mean, std = arr.mean(0), arr.std(0)
    x = np.arange(len(mean))
    plt.figure(figsize=(4.8, 3.3))
    plt.bar(x, mean, yerr=std, capsize=3)
    plt.xticks(x, [f"E{i}" for i in x])
    plt.ylabel("# patch tokens")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def safe_unnormalize(x, mean=None, std=None):
    """
    x: [C,H,W] tensor. If mean/std not provided, returns x unchanged.
    """
    if mean is None or std is None:
        return x
    mean_t = _to_tensor(mean).to(x.device).view(-1, 1, 1)
    std_t = _to_tensor(std).to(x.device).view(-1, 1, 1)
    return x * std_t + mean_t


try:
    _BEN_MEAN = BigEarthNetInfo.STATISTICS["mean"]
    _BEN_STD = BigEarthNetInfo.STATISTICS["std"]
except Exception:
    _BEN_MEAN = None
    _BEN_STD = None


def to_rgb_landsat(
    img_tensor, max_values, bands=(3, 2, 1), mean=_BEN_MEAN, std=_BEN_STD
):
    x = safe_unnormalize(img_tensor, mean, std)
    img = x.detach().cpu().float().permute(1, 2, 0).numpy()  # H,W,C

    bands = tuple(int(b) for b in bands)
    bands = tuple([b for b in bands if b < img.shape[-1]])
    if len(bands) == 0:
        chans = min(3, img.shape[-1])
        bands = tuple(range(chans))
    img = img[..., bands]

    mv = np.asarray(max_values, dtype=np.float32)
    mv = mv[: img.shape[-1]] if mv.ndim == 1 else mv
    mv = np.maximum(mv, 1e-6)
    img = img / mv

    img = np.clip(img, 0, 1)
    lo, hi = np.percentile(img, (1, 99))
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


def make_rgb(imgs, index, max_values, bands=(3, 2, 1), mean=_BEN_MEAN, std=_BEN_STD):
    return to_rgb_landsat(
        imgs[index], max_values=np.asarray(max_values), bands=bands, mean=mean, std=std
    )


def plot_routing_assignment_with_dots(
    base_rgb,  # (H*ps, W*ps, 3) float in [0,1]
    *,
    assignment_map,  # (H_p, W_p) ints in {0..E-1}
    num_experts=None,
    patch_size: int = 16,
    title: str = "Routing: top-1 expert per patch",
    border_color=(1, 1, 1, 0.55),  # white with alpha for borders
    border_lw=0.6,
    dot_size=18,  # matplotlib points^2
    dot_edge_color="k",
    dot_edge_lw=0.25,
    output_path: str = None,
):
    """
    Renders patch grid as borders and a dot in the center of each patch,
    colored by the top-1 expert for that patch.
    """
    H_p, W_p = assignment_map.shape
    H_img, W_img = base_rgb.shape[:2]
    assert (
        H_img == H_p * patch_size and W_img == W_p * patch_size
    ), f"Image size ({H_img}x{W_img}) must match patches ({H_p}x{W_p}) × ps ({patch_size})."

    if num_experts is None:
        max_id = int(assignment_map.max()) if assignment_map.size else -1
        num_experts = max(0, max_id) + 1

    # expert colormap (tab20 cycling)
    base_tab = mpl.colormaps.get_cmap("tab20")(np.arange(20))
    if num_experts > 20:
        reps = int(np.ceil(num_experts / 20))
        colors = np.vstack([base_tab for _ in range(reps)])[:num_experts]
    else:
        colors = base_tab[:num_experts]

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.imshow(base_rgb)
    ax.set_title(title)
    ax.axis("off")

    # Draw patch borders
    for gy in range(0, H_img + 1, patch_size):
        ax.plot(
            [0, W_img], [gy - 0.5, gy - 0.5], color=border_color, linewidth=border_lw
        )
    for gx in range(0, W_img + 1, patch_size):
        ax.plot(
            [gx - 0.5, gx - 0.5], [0, H_img], color=border_color, linewidth=border_lw
        )

    # Compute patch centers
    ys, xs = np.mgrid[0:H_p, 0:W_p]
    xs = (xs + 0.5) * patch_size
    ys = (ys + 0.5) * patch_size

    # Colors per expert id
    a = assignment_map.flatten()
    dot_colors = np.zeros((a.size, 4))
    for e in range(num_experts):
        dot_colors[a == e] = colors[e]

    # Scatter dots at centers
    ax.scatter(
        xs.flatten(),
        ys.flatten(),
        s=dot_size,
        c=dot_colors,
        edgecolors=dot_edge_color,
        linewidths=dot_edge_lw,
    )

    # Legend (E0..E{E-1})
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[e],
            markeredgecolor="k",
            markeredgewidth=dot_edge_lw,
            markersize=6,
            linestyle="None",
            label=f"E{e}",
        )
        for e in range(num_experts)
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False
    )
    fig.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def get_expert_contributions_for_layer(moe_layer, x_tokens):
    """
    Per-token contribution (||expert_out * gate||) for each expert for ONE MoE layer.
    Returns: [B, N, E]
    """
    B, N, C = x_tokens.shape
    flat = x_tokens.reshape(-1, C)
    gates, _, topk_idx, *_ = moe_layer.gate(flat)

    contrib = torch.zeros(B * N, moe_layer.num_experts, device=x_tokens.device)
    for e_idx, expert in enumerate(moe_layer.experts):
        mask = (topk_idx == e_idx).any(dim=1)
        idx = torch.where(mask)[0]
        if idx.numel() > 0:
            out = expert(flat[idx]) * gates[idx, e_idx].unsqueeze(-1)
            contrib[idx, e_idx] = out.norm(dim=-1)
    return contrib.reshape(B, N, moe_layer.num_experts)


def tokens_to_patch_maps(contributions, patch_grid, num_meta_tokens=4, has_cls=True):
    """
    contributions: [B, N, E] -> list[B][E] -> (H,W)
    """
    B, N, E = contributions.shape
    start = num_meta_tokens + (1 if has_cls else 0)
    patch_tok = contributions[:, start:, :]
    assert (
        patch_tok.shape[1] == patch_grid * patch_grid
    ), f"Patch count mismatch: got {patch_tok.shape[1]}, expected {patch_grid**2}"
    maps = []
    for b in range(B):
        maps_b = []
        for e in range(E):
            m = (
                patch_tok[b, :, e]
                .detach()
                .cpu()
                .numpy()
                .reshape(patch_grid, patch_grid)
            )
            maps_b.append(m)
        maps.append(maps_b)
    return maps


def _shared_scale(maps_list):
    vmin = min([m.min() for m in maps_list]) if len(maps_list) else 0.0
    vmax = max([m.max() for m in maps_list]) if len(maps_list) else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def show_original_plus_expert_heatmaps(
    maps_for_image,
    base_rgb,
    title="",
    patch_size=16,
    alpha=0.4,
    cmap="plasma",
    routed_counts=None,
    clip_percentile=99.0,
    show_standalone=True,
):
    """
    For each expert (row):
       Col 1: Original
       Col 2: Overlay (heatmap on RGB)
       Col 3: Standalone heatmap (with shared colorbar)
    """
    E = len(maps_for_image)
    maps = [m.copy() for m in maps_for_image]

    if clip_percentile is not None and E > 0:
        all_vals = np.concatenate([m.flatten() for m in maps])
        lo = np.percentile(all_vals, 0)
        hi = np.percentile(all_vals, clip_percentile)
        hi = max(hi, lo + 1e-6)
        maps = [np.clip(m, lo, hi) for m in maps]

    vmin, vmax = _shared_scale(maps)
    ncols = 3 if show_standalone else 2
    fig, axes = plt.subplots(E, ncols, figsize=(5 * ncols, 3 * E))
    if E == 1:
        axes = np.expand_dims(axes, 0)

    last_im = None
    for e in range(E):
        map_resized = np.kron(maps[e], np.ones((patch_size, patch_size)))
        axes[e, 0].imshow(base_rgb)
        axes[e, 0].set_title("Original")
        axes[e, 0].axis("off")
        axes[e, 1].imshow(base_rgb)
        last_im = axes[e, 1].imshow(
            map_resized, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax
        )
        title_e = f"Expert {e}"
        if routed_counts is not None:
            title_e += f" (tokens: {routed_counts[e]})"
            if routed_counts[e] == 0:
                title_e += " — No tokens"
        axes[e, 1].set_title(title_e)
        axes[e, 1].axis("off")
        if show_standalone:
            axes[e, 2].imshow(map_resized, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[e, 2].set_title(f"Expert {e} Heatmap")
            axes[e, 2].axis("off")

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes.ravel().tolist(), shrink=0.7, fraction=0.03, pad=0.02
        )
        cbar.ax.set_ylabel("Activation / Δ", rotation=270, labelpad=12)
    plt.show()


@torch.no_grad()
def compute_ablation_heatmaps(
    layer_moe,
    x_tokens,
    patch_grid,
    num_meta_tokens,
    has_cls=True,
    propagate_through=None,
    layer_norm=None,
    residual_base_tokens=None,
):
    """
    For each expert in THIS MoE layer:
      1) run normally through this MoE, then (optionally) through the rest of the encoder
      2) ablate that expert in THIS MoE, then (optionally) through the rest
      3) measure Δ = ||final_normal - final_ablated|| (per token), map to patches.

    Returns: np.ndarray [B, E, H, W]
    """
    B, N, C = x_tokens.shape
    E = len(layer_moe.experts)

    def run_moe_once(tokens, ablate_idx=None):
        flat = tokens.reshape(-1, C)
        gates, _, topk_idx, *_ = layer_moe.gate(flat)
        out = torch.zeros_like(flat)
        for e_idx, expert in enumerate(layer_moe.experts):
            if ablate_idx is not None and e_idx == ablate_idx:
                continue
            mask = (topk_idx == e_idx).any(dim=1)
            if mask.any():
                idx = torch.where(mask)[0]
                out.index_add_(
                    0, idx, expert(flat[idx]) * gates[idx, e_idx].unsqueeze(-1)
                )
        return out.reshape(B, N, C)

    def propagate(tokens, tail_layers):
        if tail_layers is None:
            return tokens
        y = tokens
        for lyr in tail_layers:
            y, _ = lyr(y)  # ignore MoE loss here
        return y

    heat = []
    # normal path
    normal_after = run_moe_once(x_tokens, ablate_idx=None)
    normal_layer_out = (
        normal_after if residual_base_tokens is None else residual_base_tokens + normal_after
    )
    normal_final = propagate(normal_layer_out, propagate_through)
    if layer_norm is not None:
        normal_final = layer_norm(normal_final)

    for e in range(E):
        ablated_after = run_moe_once(x_tokens, ablate_idx=e)
        ablated_layer_out = (
            ablated_after
            if residual_base_tokens is None
            else residual_base_tokens + ablated_after
        )
        ablated_final = propagate(ablated_layer_out, propagate_through)
        if layer_norm is not None:
            ablated_final = layer_norm(ablated_final)

        diff = (normal_final - ablated_final).norm(dim=-1)  # [B,N]

        start = num_meta_tokens + (1 if has_cls else 0)
        patch_diff = diff[:, start:]  # [B, H*W]
        maps = patch_diff.reshape(B, patch_grid, patch_grid).detach().cpu().numpy()
        heat.append(maps)

    return np.stack(heat, axis=1)  # [B,E,H,W]


def layer_report_simple(
    model,
    imgs,
    meta_week=None,
    meta_hour=None,
    meta_lat=None,
    meta_lon=None,
    image_index: int = 0,
    layer_index: int = 0,
    max_values=np.array([65454.0, 65454.0, 65330.308], dtype=np.float32),
    rgb_bands=(3, 2, 1),
    device="cuda",
):
    """
    Produce three easy-to-explain views for ONE encoder layer:
      (i) expert usage bars (mean±std across batch)
      (ii) routing overlay (top-1 expert per patch) for one image
      (iii) per-expert contribution maps (||g_e·f_e||) and ablation Δ maps (||y - y_(-e)||)

    Requires the following helpers from your code to be available:
      - deterministic_routing
      - encoder_tokens_per_layer
      - get_expert_contributions_for_layer
      - tokens_to_patch_maps
      - compute_ablation_heatmaps
      - show_original_plus_expert_heatmaps
      - make_rgb
    """
    model.eval()
    model.to(device)
    imgs = imgs.to(device)
    if meta_week is not None:
        meta_week = meta_week.to(device)
    if meta_hour is not None:
        meta_hour = meta_hour.to(device)
    if meta_lat is not None:
        meta_lat = meta_lat.to(device)
    if meta_lon is not None:
        meta_lon = meta_lon.to(device)

    encoder = model.encoder
    H = imgs.shape[2]
    patch_grid = H // encoder.patch_size

    with torch.no_grad(), deterministic_routing(model):
        states_by_layer, total_moe_loss, M = encoder_layer_states(
            encoder, imgs, meta_week, meta_hour, meta_lat, meta_lon
        )

        # states at selected layer
        st = states_by_layer[layer_index]
        z_L = st["z_pre_moe"]  # [B, N, C], true input to MoE gate/experts
        x_after_attn = st["x_after_attn"]  # [B, N, C], residual branch
        moe = encoder.layers[layer_index].moe
        num_experts = moe.num_experts

        # (i) Expert usage + (ii) top-1 assignment map
        assign_maps, usage = routing_assign_and_usage_for_layer(
            moe, z_L, patch_grid, num_meta_tokens=M
        )
        plot_expert_usage_bars(
            usage, title=f"Layer {layer_index} — Expert routing (patch tokens)"
        )

        # RGB for the chosen image
        base_rgb = make_rgb(imgs, image_index, max_values=max_values, bands=rgb_bands)
        plot_routing_assignment_with_dots(
            base_rgb,
            assignment_map=assign_maps[image_index],
            num_experts=num_experts,
            patch_size=encoder.patch_size,
            title=f"Layer {layer_index} — Top-1 expert per patch",
        )

        # (iii-a) Per-expert contribution maps: || g_e · f_e(x) ||
        contrib = get_expert_contributions_for_layer(moe, z_L)  # [B, N, E]
        contrib_maps = tokens_to_patch_maps(contrib, patch_grid, num_meta_tokens=M)
        counts_img = usage[image_index]  # [E]

        show_original_plus_expert_heatmaps(
            [contrib_maps[image_index][e] for e in range(num_experts)],
            base_rgb=base_rgb,
            title=f"Layer {layer_index} • Contribution (||g_e·f_e||)",
            patch_size=encoder.patch_size,
            alpha=0.35,
            cmap="plasma",
            routed_counts=counts_img,
            clip_percentile=99.0,
            show_standalone=True,
        )

        # (iii-b) Per-expert ablation Δ maps: || y - y_(-e) ||
        tail_layers = (
            encoder.layers[layer_index + 1 :]
            if (layer_index + 1) < len(encoder.layers)
            else None
        )
        ablation_maps = compute_ablation_heatmaps(
            moe,
            z_L,
            patch_grid,
            num_meta_tokens=M,
            propagate_through=tail_layers,
            layer_norm=encoder.norm,
            residual_base_tokens=x_after_attn,
        )  # [B, E, H, W]

        show_original_plus_expert_heatmaps(
            [ablation_maps[image_index, e] for e in range(num_experts)],
            base_rgb=base_rgb,
            title=f"Layer {layer_index} • Ablation Δ (||y - y_(-e)||)",
            patch_size=encoder.patch_size,
            alpha=0.35,
            cmap="viridis",
            routed_counts=counts_img,
            clip_percentile=99.0,
            show_standalone=True,
        )
    return {
        "num_experts": int(num_experts),
        "usage_image": counts_img,
        "moe_loss": float(total_moe_loss.item()),
    }
