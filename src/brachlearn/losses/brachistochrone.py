import torch
from torch import nn

class BrachistochroneLoss(nn.Module):
    """
    Simplified BRACHISTOCHRONE loss without STFT energy proxy.
    Uses direct feature norms as energy measures.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def energy(self, h: torch.Tensor) -> torch.Tensor:
        """
        Simple energy measure: L2 norm of features.
        h: (B, T) or (B, ...)
        returns (B,) non-negative energies
        """
        # Optimized: compute L2 norm directly without flattening
        return torch.norm(h.view(h.shape[0], -1), dim=1) + self.eps

    def v_of(self, E: torch.Tensor) -> torch.Tensor:
        """Velocity based on energy"""
        return torch.sqrt(2.0 * self.alpha * E + self.eps)

    def forward(self, h_list):
        """
        h_list: list of (B, T) tensors [h0, h1, ..., hT]
        returns L_path, L_mono weighted by beta and gamma
        """
        assert len(h_list) >= 2, "Need at least two stages"
        
        # Ultra-minimal: compute everything in one expression
        h0, h1 = h_list[0], h_list[1]
        
        # Compute energies and losses in minimal operations
        E0 = torch.norm(h0.view(h0.shape[0], -1), dim=1) + self.eps
        E1 = torch.norm(h1.view(h1.shape[0], -1), dim=1) + self.eps
        
        # Single expression for both losses
        v0 = torch.sqrt(2.0 * self.alpha * E0 + self.eps)
        L_path = (torch.abs(E1 - E0) / (v0 + self.eps)).mean()
        L_mono = torch.relu(E1 - E0).mean()
        
        return self.beta * L_path, self.gamma * L_mono
