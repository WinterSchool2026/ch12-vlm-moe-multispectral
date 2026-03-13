# Lightweight Vision-Language Mixture-of-Experts for Interpretable Multispectral Satellite Representation Learning

## Description

Earth Observation imagery exhibits strong heterogeneity across land-cover classes, making it inefficient for a single compact model to represent all patterns equally well. Modern Earth Observation models increasingly rely on large opaque encoders to achieve strong performance. On the other hand, Mixture-of-Experts architectures promise computational efficiency and specialization. The conditional routing of these models enables different experts to specialize in distinct spectral-spatial regimes. However, their internal behavior remains poorly understood.
VLMs introduce a powerful new capability, which is the semantic alignment between visual features and natural language.
 In this challenge, participants will explore a lightweight metadata-aware Mixture-of-Experts Masked Autoencoder (Geo-MoE-MAE) pretrained on multispectral Landsat imagery. Building on this foundation, the group will develop a lightweight vision-language interface using existing multilabel annotations (i.e., BigEarthNet labels).
The core idea is to convert these labels into text prompts and align image representations with a frozen lightweight language encoder. This will enable text-to-image retrieval, and produce spatially localized experts with semantically meaningful specialization, while remaining computationally efficient.
Practical steps:

- Participants will combine:
  - A pretrained lightweight MoE-MAE vision encoder (patch-based, sparse experts),
  - A small text encoder,
  - The resulting model will be trained with contrastive VLM alignment (CLIP-style).

- The resulting model to be analyzed through:
  - Routing: top-1 expert assignment for each patch.
  - Contribution: how strongly each expert influences each patch.
  - Ablation: how much the final representation changes when a specific expert is removed.
  - Expert naming: because image and text live in the same feature space, experts can be named by comparing expert-conditioned image embeddings to a small query bank of basic land cover classes.

The goal is to demonstrate how conditional routing and weak language supervision together provide an interpretable, resource-efficient representation model for optical remote sensing.

## Recommended reading material

- [https://arxiv.org/pdf/2111.06377](https://arxiv.org/pdf/2111.06377)
- [https://arxiv.org/pdf/1701.06538](https://arxiv.org/pdf/1701.06538)
- [https://arxiv.org/pdf/2509.10919](https://arxiv.org/pdf/2509.10919)
- [https://arxiv.org/pdf/2103.00020](https://arxiv.org/pdf/2103.00020)
