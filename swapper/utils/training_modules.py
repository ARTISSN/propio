import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer

class GatingMLP(nn.Module):
    def __init__(self, in_dim, num_modalities):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_modalities)
    def forward(self, x):
        # x: [B, seq_len, in_dim]  → logits [B, seq_len, M]
        logits = self.proj(x)
        return torch.sigmoid(logits).unsqueeze(-1)  # [B,seq_len,M,1]

class AdaptiveLoraLayer(LoraLayer):
    """
    Extends PEFT’s LoraLayer to accept a gating vector at forward time.
    """
    def __init__(self, orig_layer: LoraLayer, cond_dim: int):
        super().__init__(**orig_layer._config.to_dict())
        # copy weights from orig_layer
        self.r      = orig_layer.r
        self.lora_A = orig_layer.lora_A
        self.lora_B = orig_layer.lora_B
        self.scaling = orig_layer.scaling
        # gating network:
        self.gate_net = nn.Sequential(
            nn.Linear(cond_dim, self.lora_A.shape[0]), 
            nn.Sigmoid()
        )

    def forward(self, x, gate_context=None):
        # x: [*, in_features], gate_context: [*, cond_dim]
        result = super().forward(x)  # W x + B A x
        if gate_context is not None:
            g = self.gate_net(gate_context)      # [*, r]
            # scale only the low-rank part (B·A·x)
            lora_part = (self.lora_B @ (self.lora_A @ x.unsqueeze(-1))).squeeze(-1) * g
            # original Lin: W x + scaling * lora_part, so replace:
            return F.linear(x, self.weight) + self.scaling * lora_part
        return result

class MultiModalFusion(nn.Module):
    """
    Fuses identity, lighting (1D) and spatial (3×H×W) features into a single embedding.
    """
    def __init__(self, id_dim, light_dim, spatial_dim, fuse_dim, num_heads=8):
        super().__init__()
        self.id_proj      = nn.Linear(id_dim,     fuse_dim)
        self.light_proj   = nn.Linear(light_dim,  fuse_dim)
        self.spatial_proj = nn.Conv2d(spatial_dim, fuse_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(fuse_dim, num_heads, batch_first=True)

    def forward(self, identity, lighting, normal_map):
        # identity: [B, id_dim]
        # lighting: [B, light_dim]
        # normal_map: [B, C, H, W]
        id_e     = self.id_proj(identity).unsqueeze(1)         # [B,1,fuse_dim]
        light_e  = self.light_proj(lighting).unsqueeze(1)      # [B,1,fuse_dim]
        spatial  = self.spatial_proj(normal_map)               # [B,fuse_dim,H,W]
        B, F, H, W = spatial.shape
        spat_e   = spatial.flatten(2).transpose(1,2)           # [B, H·W, fuse_dim]
        
        # Concatenate and self-attend
        tokens   = torch.cat([id_e, light_e, spat_e], dim=1)   # [B, 2+H·W, fuse_dim]
        fused, _ = self.attn(tokens, tokens, tokens)           # [B, 2+H·W, fuse_dim]
        # Pool (e.g. mean over sequence) to get [B, fuse_dim]
        return fused.mean(dim=1)