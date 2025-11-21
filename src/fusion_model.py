import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class TextEncoder(nn.Module):


    # Loading pre-trained CLIP text encoder
    def __init__(self, device="cuda"):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.device = device

    def forward(self, text: str):
        """
        text: python string
        returns: (1, D) embedding tensor
        """
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding  # Shape: (1, 512)
        
class CrossAttentionFusion(nn.Module):

    # Cross-attention module between text embeddings and Gaussian features: queries from text, keys/values from Gaussians
    # Useful for conditioning Gaussian attributes on text meaning
    
    def __init__(self, gaussian_dim=64, text_dim=512, hidden_dim=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Project both modalities to same hidden dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.gauss_proj = nn.Linear(gaussian_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, gaussian_dim)

    def forward(self, text_emb, gaussian_feats):
        """
        text_emb:       (1, text_dim)
        gaussian_feats: (N, gaussian_dim)   <-- features for each Gaussian point

        returns: (N, gaussian_dim) fused features
        """
        # Project into attention space
        text_q = self.text_proj(text_emb).unsqueeze(0) # (1, 1, hidden)
        gauss_kv = self.gauss_proj(gaussian_feats).unsqueeze(1)  # (N, 1, hidden)

        # In MultiheadAttention: sequence dimension is first
        # So we must permute:
        gauss_kv = gauss_kv.permute(1, 0, 2)  # (1, N, hidden)

        # Perform cross-attention: text queries attend to Gaussian keys
        attn_output, _ = self.attn(query=text_q, key=gauss_kv, value=gauss_kv)

        # Broadcast attention output to all Gaussians
        attn_output = attn_output.repeat(gaussian_feats.size(0), 1, 1).squeeze(1)

        # Fuse
        fused = gaussian_feats + self.out_proj(attn_output)  # residual skip
        return fused

class TextGaussianFusionModel(nn.Module):
    """
    Top-level module:
    - Encodes text prompt
    - Cross-attends with Gaussian features
    - Returns fused Gaussian features for use in renderer/training
    """
    def __init__(self, gaussian_dim=64, device="cuda"):
        super().__init__()
        self.text_encoder = TextEncoder(device=device)
        self.cross_attn = CrossAttentionFusion(gaussian_dim=gaussian_dim)

    def forward(self, text_prompt, gaussian_features):
        """
        text_prompt: python string
        gaussian_features: (N, F) Gaussian feature tensor from GS

        returns fused_features: (N, F)
        """
        text_emb = self.text_encoder(text_prompt)  # (1, D)
        fused = self.cross_attn(text_emb, gaussian_features)
        return fused