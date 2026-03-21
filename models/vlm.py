from sentence_transformers import SentenceTransformer
from torch import nn
import torch
import torch.nn.functional as F

# This is a placeholder file for the Vision-Language Mixture-of-Experts (VLM-MoE) model. The actual implementation will be added here in the future.
class VLM(nn.Module):
    def __init__(self, image_encoder, text_encoder, temperature_init, text_dim=384, device=None, freeze_encoders=False):
        super().__init__()
        self.device = device

        # Set image encoder
        self.image_encoder = image_encoder

        # Set text encoder
        self.text_dim = text_dim
        if text_encoder == "bert":
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_encoder.eval()
        elif text_encoder == "clip":
            assert False, "Not implemented!"
        
        # If specified
        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # Projection head (to text encoder space)
        proj_layers = [nn.Linear(self.image_encoder.embed_dim, self.text_dim)]
        proj_layers.extend([nn.GELU(), nn.Linear(self.text_dim, self.text_dim)])
        self.image_proj = nn.Sequential(*proj_layers)

        # ---- CLIP-style temperature parameter ----
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature_init).log())


    def forward(self, text, image, meta_week=None, meta_hour=None, meta_lat=None, meta_lon=None):
        # get image embeddings
        image_embeddings = self.encode_images(image, meta_week, meta_hour, meta_lat, meta_lon, normalize_embeddings=True)

        # get text embeddings
        text_embeddings = self.encode_text(text, normalize_embeddings=True)

        # Compute similarity matrix
        sim_mat_image = self.compute_similarity_matrix(image_embeddings, text_embeddings)

        return sim_mat_image

    def encode_images(self, image, meta_week=None, meta_hour=None, meta_lat=None, meta_lon=None, normalize_embeddings=True):
        _, _, image_embeddings = self.image_encoder(image, meta_week, meta_hour, meta_lat, meta_lon)
        # get CLS embeddings
        cls_image_embeddings = image_embeddings[:, 4]
        
        # project to target space
        prj_image_embeddings = self.image_proj(cls_image_embeddings)

        if normalize_embeddings:
            prj_image_embeddings = F.normalize(prj_image_embeddings, dim=-1)

        return prj_image_embeddings

    def encode_text(self, text, normalize_embeddings=True):
        text_embeddings = self.text_encoder.encode(text, normalize_embeddings=normalize_embeddings, convert_to_tensor=True, device=self.device)
        text_embeddings = text_embeddings.clone()
        return text_embeddings

    def compute_similarity_matrix(self, image_embeddings, text_embeddings):
        return self.logit_scale.exp().clamp(1e-3, 100.0) * image_embeddings @ text_embeddings.t()


def clip_multilabel_loss(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,B) image->text similarities (already scaled)
    pos_mask: (B,B) bool, True where j is a positive for i
    """
    device = logits.device
    B = logits.size(0)

    # Ensure each row has at least one positive (itself is always positive)
    eye = torch.eye(B, device=device, dtype=torch.bool)
    pos_mask = pos_mask | eye

    def row_loss(logits_mat, mask):
        # logsumexp over all candidates
        denom = torch.logsumexp(logits_mat, dim=1)  # (B,)
        # logsumexp over positives only
        neg_inf = torch.tensor(-1e9, device=device, dtype=logits_mat.dtype)
        pos_logits = torch.where(mask, logits_mat, neg_inf)
        num = torch.logsumexp(pos_logits, dim=1)  # (B,)
        return (denom - num).mean()
    
    loss_i2t = row_loss(logits, pos_mask)
    loss_t2i = row_loss(logits.t(), pos_mask.t())
    return 0.5 * (loss_i2t + loss_t2i)