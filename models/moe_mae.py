"""
Lightweight MoE-MAE Vision Transformer implementation (PyTorch)
inspired from "How Lightweight Can a Vision Transformer Be"
and adapted for EO data.

This file contains:
- SwiGLU FFN with options to share V and W2 across experts
- NoisyTopK gating (Softmax-k) and gating probability utilities
- MoE layer: dispatcher/gatherer using sparse routing
- Grouped Query Attention (GQA)
- MoE-based Transformer Encoder Layer
- MoE-MAE encoder model builder with depth-wise scaling and staged expert increases
- MoE-MAE (encoder + lightweight decoder)

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- SwiGLU FFN --------------------------------


class SwiGLU(nn.Module):
    """SwiGLU feedforward used as expert network.
    Optionally share V and W2 across experts by passing shared modules.
    As described in the paper, V and W2 can be shared to reduce parameters.
    """

    def __init__(
        self,
        m: int,
        dh: int,
        shared_V: Optional[nn.Linear] = None,
        shared_W2: Optional[nn.Linear] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.m = m
        self.dh = dh
        self.W = nn.Linear(m, dh)
        self.V = shared_V if shared_V is not None else nn.Linear(m, dh)
        self.W2 = shared_W2 if shared_W2 is not None else nn.Linear(dh, m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        xW = self.W(x)
        xV = self.V(x)
        x_act = F.silu(xW) * xV
        out = self.W2(x_act)
        return self.dropout(out)


# --------------------------- NoisyTopK Gating -----------------------------


class NoisyTopKGate(nn.Module):
    """Noisy Top-K gating as in Shazeer et al.
    Produces sparse routing weights for 't' experts and selects top-k for each token.
    Returns: gates (B*N, t) sparse, top-k noisy values H, indices of chosen experts, and noise_scale
    """

    def __init__(self, m: int, t_experts: int, k: int = 2, eps: float = 1e-9):
        super().__init__()
        self.Wg = nn.Linear(m, t_experts, bias=True)
        self.Wnoise = nn.Linear(m, t_experts, bias=True)
        self.t = t_experts
        self.k = k
        self.eps = eps

    def forward(self, x):
        # x: (B*N, m)
        logits = self.Wg(x)  # (B*N, t)
        noise_scale = F.softplus(self.Wnoise(x))  # pylint:disable=E1102
        noise = torch.randn_like(logits)
        H = logits + noise * noise_scale

        # top-k values and indices
        topk_vals, topk_idx = torch.topk(H, self.k, dim=-1)
        topk_softmax = F.softmax(topk_vals, dim=-1)

        # build sparse gates
        gates = x.new_zeros(H.shape)
        gates.scatter_(-1, topk_idx, topk_softmax)

        return gates, H, topk_idx, noise_scale, logits


# ------------------------------ MoE Layer --------------------------------


class MoELayer(nn.Module):
    """Mixture-of-Experts layer."""

    def __init__(
        self,
        m: int,
        dh: int,
        num_experts: int = 3,
        k: int = 2,
        share_V_W2: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.m = m
        self.dh = dh
        self.num_experts = num_experts
        self.k = k

        shared_V = nn.Linear(m, dh) if share_V_W2 else None
        shared_W2 = nn.Linear(dh, m) if share_V_W2 else None

        self.experts = nn.ModuleList(
            [
                SwiGLU(m, dh, shared_V=shared_V, shared_W2=shared_W2, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        self.gate = NoisyTopKGate(m, num_experts, k=k)
        self.wload = 1e-2
        self.wimportance = 1e-2

    def coefficient_of_variation(self, x):
        mean = x.mean()
        std = x.std(unbiased=False)
        return std / (mean + 1e-6)

    def forward(self, x):
        B, N, m = x.shape
        flat = x.reshape(B * N, m)
        gates, H_vals, topk_idx, xW_noise_softplus, xW_g = self.gate(flat)
        importance = gates.sum(dim=0)
        L_importance = self.wimportance * self.coefficient_of_variation(importance)
        topk_vals, _ = torch.topk(H_vals, k=self.k + 1, dim=1)
        l_k = topk_vals[:, self.k - 1 : self.k]
        l_k_plus_1 = topk_vals[:, self.k : self.k + 1]
        l_k_broadcast = l_k.repeat(1, H_vals.shape[1])
        l_k_plus_1_broadcast = l_k_plus_1.repeat(1, H_vals.shape[1])
        psi_H = torch.where(
            H_vals > l_k_broadcast,
            l_k_broadcast,
            torch.where(H_vals <= l_k_plus_1_broadcast, l_k_plus_1_broadcast, H_vals),
        )
        normal_dist = torch.distributions.Normal(
            torch.zeros_like(xW_g), torch.ones_like(xW_g)
        )
        numerator = xW_g - psi_H
        denominator = xW_noise_softplus
        P_x = normal_dist.cdf(numerator / denominator)
        load = P_x.sum(dim=0)
        L_load = self.wload * self.coefficient_of_variation(load)
        L_moe = L_importance + L_load
        flat_out = flat.new_zeros(flat.shape)
        for e_idx, expert in enumerate(self.experts):
            expert_mask = (topk_idx == e_idx).any(dim=1)
            if not expert_mask.any():
                continue
            token_indices = torch.where(expert_mask)[0]
            inp = flat[token_indices]
            expert_gates_vals = gates[token_indices, e_idx].unsqueeze(-1)
            expert_output = expert(inp) * expert_gates_vals
            flat_out.index_add_(0, token_indices, expert_output)
        y = flat_out.reshape(B, N, m)
        return y, L_moe


# ------------------------ Grouped Query Attention -------------------------


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)."""

    def __init__(
        self, dim, num_heads=8, num_groups=4, attn_dropout=0.0, proj_dropout=0.0
    ):
        super().__init__()
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(dim, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_groups, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_groups, self.head_dim)
        k_exp = (
            k.unsqueeze(2)
            .repeat(1, 1, self.num_heads // self.num_groups, 1, 1)
            .view(B, N, self.num_heads, self.head_dim)
        )
        v_exp = (
            v.unsqueeze(2)
            .repeat(1, 1, self.num_heads // self.num_groups, 1, 1)
            .view(B, N, self.num_heads, self.head_dim)
        )

        q = q.permute(0, 2, 1, 3)  # (B, H, N, head_dim)
        k_exp = k_exp.permute(0, 2, 3, 1)  # (B, H, head_dim, N)
        v_exp = v_exp.permute(0, 2, 1, 3)  # (B, H, N, head_dim)
        attn = (q @ k_exp) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v_exp
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out


# --------------------- MoE Transformer Encoder Layer ----------------------


class MoETransformerEncoderLayer(nn.Module):
    """A single transformer encoder layer with Grouped Query Attention and an MoE block."""

    def __init__(
        self,
        m: int,
        dh: int,
        num_experts: int = 3,
        k: int = 2,
        num_heads: int = 8,
        num_groups: int = 4,
        share_V_W2: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(m)
        self.attn = GroupedQueryAttention(m, num_heads=num_heads, num_groups=num_groups)
        self.norm2 = nn.LayerNorm(m)
        self.moe = MoELayer(
            m, dh, num_experts=num_experts, k=k, share_V_W2=share_V_W2, dropout=dropout
        )

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        x = x + y
        z = self.norm2(x)
        moe_out, moe_loss = self.moe(z)
        x = x + moe_out
        return x, moe_loss


# --------------------------- MOEEncoder Model ----------------------------------


class MOEEncoder(nn.Module):
    """The GEO-MoE-MAE Vision Transformer Encoder model."""

    def __init__(
        self,
        *,
        img_size=36,
        patch_size=3,
        in_chans=3,
        embed_dim=144,
        depth=15,
        dfirsth=144,
        dlasth=72,
        num_heads=8,
        num_groups=4,
        experts_config: List[int] = None,
        experts_per_stage: Tuple[int, int] = (3, 5),
        k=2,
        share_V_W2=True,
        dropout=0.1,
        num_meta_tokens=4,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_grid = img_size // patch_size
        self.num_patches = self.patch_grid * self.patch_grid
        self.num_meta_tokens = num_meta_tokens
        self.patch_proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.week_proj = nn.Linear(2, embed_dim)
        self.hour_proj = nn.Linear(2, embed_dim)
        self.lat_proj = nn.Linear(2, embed_dim)
        self.lon_proj = nn.Linear(2, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1 + self.num_meta_tokens, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.depth = depth
        self.dh_list = [
            int(((depth - 1 - i) / (depth - 1)) * (dfirsth - dlasth)) + dlasth
            for i in range(depth)
        ]
        if experts_config is None:
            experts_config = []
            num_stages = 3
            stage_size = depth // num_stages
            base_experts, final_experts = experts_per_stage

            for i in range(depth):
                stage = min(i // stage_size, num_stages - 1)
                num_experts = base_experts + stage * (final_experts - base_experts - 1)
                experts_config.append(num_experts)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            layer = MoETransformerEncoderLayer(
                m=embed_dim,
                dh=self.dh_list[i],
                num_experts=experts_config[i],
                k=k,
                num_heads=num_heads,
                num_groups=num_groups,
                share_V_W2=share_V_W2,
                dropout=dropout,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, meta_week=None, meta_hour=None, meta_lat=None, meta_lon=None):
        B = x.shape[0]
        p = self.patch_proj(x)
        p = p.flatten(2).transpose(1, 2)
        device = x.device
        if meta_week is None:
            meta_week = torch.zeros(B, 2, device=device)
        if meta_hour is None:
            meta_hour = torch.zeros(B, 2, device=device)
        if meta_lat is None:
            meta_lat = torch.zeros(B, 2, device=device)
        if meta_lon is None:
            meta_lon = torch.zeros(B, 2, device=device)

        week_token = self.week_proj(meta_week).unsqueeze(1)  # (B, 1, embed_dim)
        hour_token = self.hour_proj(meta_hour).unsqueeze(1)
        lat_token = self.lat_proj(meta_lat).unsqueeze(1)
        lon_token = self.lon_proj(meta_lon).unsqueeze(1)

        meta_tokens = torch.cat([week_token, hour_token, lat_token, lon_token], dim=1)

        cls = self.cls_token.expand(B, -1, -1)

        x_tokens = torch.cat([meta_tokens, cls, p], dim=1)  # cls token is at index 0
        x_tokens = x_tokens + self.pos_embed[:, : x_tokens.shape[1], :]

        total_moe_loss = x_tokens.new_zeros(1, device=x.device).sum()
        for layer in self.layers:
            x_tokens, l = layer(x_tokens)
            total_moe_loss = total_moe_loss + l

        ln_x_tokens = self.norm(x_tokens)
        return ln_x_tokens, total_moe_loss, x_tokens


# --------------------------- GEO-MOE-MAE --------------------------------


class MOEMAE(nn.Module):
    """Masked Autoencoder wrapper MoEEncoder and a small decoder."""

    def __init__(
        self,
        encoder: MOEEncoder,
        decoder_layers: int = 2,
        decoder_embed=108,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_embed = self.encoder.patch_proj
        self.decoder_embed = nn.Linear(self.encoder.embed_dim, decoder_embed)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.encoder.num_patches, decoder_embed)
        )
        self.decoder_layers = nn.ModuleList(
            [
                MoETransformerEncoderLayer(
                    m=decoder_embed,
                    dh=decoder_embed // 2,
                    num_experts=3,
                    k=2,
                    num_heads=6,
                    num_groups=3,
                    share_V_W2=True,
                    dropout=0.1,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed)
        self.decoder_pred = nn.Linear(
            decoder_embed, encoder.patch_size * encoder.patch_size * encoder.in_chans
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def random_masking(self, x, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B, N, C = x.shape  # pylint:disable=W0612
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([B, N], device=x.device)
        mask.scatter_(1, ids_keep, 0)
        return ids_keep, ids_restore, mask

    def forward(
        self, imgs, meta_week=None, meta_hour=None, meta_lat=None, meta_lon=None
    ):
        """
        imgs: (B, C, H, W)
        meta_*: (B, 2) sin/cos or zeros
        returns:
            pred: (B, N, patch_area * in_chans)
            mask: (B, N) with 1 for masked patches
            total_moe_loss: scalar
        """
        patches = self.patch_embed(imgs)
        patches = patches.flatten(2).transpose(1, 2)
        B, N, C = patches.shape
        ids_keep, ids_restore, mask = self.random_masking(patches)
        kept_patches = torch.gather(
            patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C)
        )

        device = imgs.device
        if meta_week is None:
            meta_week = torch.zeros(B, 2, device=device)
        if meta_hour is None:
            meta_hour = torch.zeros(B, 2, device=device)
        if meta_lat is None:
            meta_lat = torch.zeros(B, 2, device=device)
        if meta_lon is None:
            meta_lon = torch.zeros(B, 2, device=device)

        week_token = self.encoder.week_proj(meta_week).unsqueeze(1)
        hour_token = self.encoder.hour_proj(meta_hour).unsqueeze(1)
        lat_token = self.encoder.lat_proj(meta_lat).unsqueeze(1)
        lon_token = self.encoder.lon_proj(meta_lon).unsqueeze(1)
        meta_tokens = torch.cat([week_token, hour_token, lat_token, lon_token], dim=1)
        cls = self.encoder.cls_token.expand(B, -1, -1)

        encoder_input = torch.cat([meta_tokens, cls, kept_patches], dim=1)

        M = self.encoder.num_meta_tokens
        d = self.encoder.embed_dim
        pos_meta_cls = self.encoder.pos_embed[:, : M + 1, :].expand(B, -1, -1)
        base_patch_pos = self.encoder.pos_embed[
            :, M + 1 : M + 1 + self.encoder.num_patches, :
        ]
        pos_kept_patches = base_patch_pos.expand(B, -1, -1).gather(
            dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d)
        )
        pos_full = torch.cat([pos_meta_cls, pos_kept_patches], dim=1)
        encoder_input = encoder_input + pos_full
        encoded = encoder_input
        total_moe_loss = torch.zeros(1, device=imgs.device)
        for layer in self.encoder.layers:
            encoded, l = layer(encoded)
            total_moe_loss = total_moe_loss + l
        encoded = self.encoder.norm(encoded)
        M = self.encoder.num_meta_tokens
        encoded_meta = encoded[:, : M + 1, :]
        encoded_patch_tokens = encoded[:, M + 1 :, :]
        dec_patches = self.decoder_embed(encoded_patch_tokens)
        num_mask = N - dec_patches.shape[1]
        if num_mask > 0:
            mask_tokens = self.mask_token.repeat(B, num_mask, 1)
            dec_patches_all = torch.cat([dec_patches, mask_tokens], dim=1)
        else:
            dec_patches_all = dec_patches
        dec_patches_restored = torch.gather(
            dec_patches_all,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_patches_all.shape[-1]),
        )
        meta_cls_proj = self.decoder_embed(encoded_meta[:, : M + 1, :])
        decoder_patches_proj = dec_patches_restored  # already in decoder dim
        decoder_seq = torch.cat([meta_cls_proj, decoder_patches_proj], dim=1)
        meta_pos = torch.zeros(
            1, M + 1, self.decoder_pos_embed.shape[-1], device=device
        )
        patch_pos = self.decoder_pos_embed  # (1, N, D_dec)
        pos_full = torch.cat([meta_pos, patch_pos], dim=1)
        decoder_seq = decoder_seq + pos_full[:, : decoder_seq.shape[1], :]
        x = decoder_seq
        for layer in self.decoder_layers:
            x, _ = layer(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x[:, M + 1 :, :])
        return pred, mask, ids_restore, total_moe_loss


def build_model(size="XXS", img_size=36, patch_size=3, in_chans=3):
    configs = {
        "S": dict(
            embed_dim=144, depth=15, dfirsth=144, dlasth=72, num_heads=8, num_groups=4
        ),
        "XS": dict(
            embed_dim=128, depth=12, dfirsth=96, dlasth=32, num_heads=8, num_groups=4
        ),
        "XXS": dict(
            embed_dim=108, depth=9, dfirsth=81, dlasth=27, num_heads=6, num_groups=3
        ),
    }
    c = configs[size]
    model = MOEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=c["embed_dim"],
        depth=c["depth"],
        dfirsth=c["dfirsth"],
        dlasth=c["dlasth"],
        num_heads=c["num_heads"],
        num_groups=c["num_groups"],
    )
    return model
