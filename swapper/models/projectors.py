import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionProjector(nn.Module):
    """Base projector class for handling dimension transformations."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ModelProjector(nn.Module):
    """Projects embeddings to different model spaces (UNet, ControlNet, etc.)"""
    def __init__(self, input_dim: int, hidden_dim: int, unet_dim: int, controlnet_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_unet = nn.Linear(hidden_dim, unet_dim)
        self.to_controlnet = nn.Linear(hidden_dim, controlnet_dim)
        self.to_pooled = nn.Linear(hidden_dim, controlnet_dim)  # pooled dim is same as controlnet
    
    def forward(self, x: torch.Tensor, target: str = "unet") -> torch.Tensor:
        x = self.shared(x)
        if target == "unet":
            return self.to_unet(x)
        elif target == "controlnet":
            return self.to_controlnet(x)
        elif target == "pooled":
            return self.to_pooled(x)
        else:
            raise ValueError(f"Unknown target: {target}")

class EmbeddingProjector(nn.Module):
    """Handles both identity and lighting projections."""
    def __init__(self, config: dict, unet_dim: int, controlnet_dim: int):
        super().__init__()
        self.identity_projector = ModelProjector(
            input_dim=128,  # Face embedding dimension
            hidden_dim=config["model"]["hidden_dim"],
            unet_dim=unet_dim,
            controlnet_dim=controlnet_dim
        )
        
        self.lighting_projector = ModelProjector(
            input_dim=config["model"]["lighting_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            unet_dim=unet_dim,
            controlnet_dim=controlnet_dim
        )
    
    def forward(self, identity: torch.Tensor, lighting: torch.Tensor, target: str = "unet") -> torch.Tensor:
        # Project both inputs to the target space
        id_proj = self.identity_projector(identity, target=target)
        light_proj = self.lighting_projector(lighting, target=target)
        
        # Combine and normalize
        combined = (id_proj + light_proj).unsqueeze(1)  # Add sequence dimension
        return combined 