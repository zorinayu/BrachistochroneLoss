import torch
from torch import nn
from ..utils.stft import stft_power

class BrachistochroneLoss(nn.Module):
    """
    Implements L_path and L_mono.
    v(h) = sqrt(2 * alpha * E(h) + eps), with E(h) as a proxy energy.
    For denoising tasks, E(h) uses high-frequency STFT energy.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1,
                 n_fft: int = 256, hop: int = 128, high_band: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_fft = n_fft
        self.hop = hop
        self.high_band = high_band
        self.eps = eps

    def energy(self, h: torch.Tensor) -> torch.Tensor:
        """
        Proxy E(h): high-frequency band energy from STFT.
        h: (B, T)
        returns (B,) non-negative energies
        """
        P = stft_power(h, n_fft=self.n_fft, hop=self.hop)  # (B, F, frames)
        F = P.shape[1]
        hf = int(F * self.high_band)
        hf_energy = P[:, hf:, :].mean(dim=(1, 2))  # average high-band energy
        return hf_energy + self.eps

    def v_of(self, E: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(2.0 * self.alpha * E + self.eps)

    def forward(self, h_list):
        """
        h_list: list of (B, T) tensors [h0, h1, ..., hT]
        returns L_path, L_mono
        """
        assert len(h_list) >= 2, "Need at least two stages"
        B = h_list[0].shape[0]

        # Energies and speeds at each stage
        E_list = [self.energy(h) for h in h_list]
        v_list = [self.v_of(E) for E in E_list]

        # L_path
        L_path_terms = []
        for k in range(len(h_list) - 1):
            step = (h_list[k+1] - h_list[k]).reshape(B, -1)
            step_norm = torch.norm(step, dim=1)  # (B,)
            L_path_terms.append(step_norm / (v_list[k] + self.eps))
        L_path = torch.stack(L_path_terms, dim=0).mean()

        # L_mono
        mono_terms = []
        for k in range(len(h_list) - 1):
            diff = E_list[k+1] - E_list[k]
            mono_terms.append(torch.relu(diff))
        L_mono = torch.stack(mono_terms, dim=0).mean()

        return self.beta * L_path, self.gamma * L_mono
