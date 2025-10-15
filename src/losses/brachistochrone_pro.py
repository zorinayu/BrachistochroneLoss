import torch
import torch.nn as nn
from .brachistochrone import BrachistochroneLoss

class BrachistochroneAdam(nn.Module):
    """
    Brachistochrone + Adam optimizer combination
    Uses Brachistochrone loss with Adam optimizer for better convergence
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
        # Initialize Brachistochrone loss
        self.brach_loss = BrachistochroneLoss(alpha=alpha, beta=beta, gamma=gamma, eps=eps)
    
    def forward(self, h_list):
        """
        Forward pass using Brachistochrone loss
        This is a wrapper that uses the original Brachistochrone loss
        but is designed to work with Adam optimizer
        
        Args:
            h_list: list of (B, ...) tensors [h0, h1, ..., hT]
            
        Returns:
            L_path: Path loss term
            L_mono: Monotonicity loss term
        """
        return self.brach_loss(h_list)


class BrachistochroneSGD(nn.Module):
    """
    Brachistochrone + SGD optimizer combination
    Uses Brachistochrone loss with SGD optimizer
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
        # Initialize Brachistochrone loss
        self.brach_loss = BrachistochroneLoss(alpha=alpha, beta=beta, gamma=gamma, eps=eps)
    
    def forward(self, h_list):
        """
        Forward pass using Brachistochrone loss
        This is a wrapper that uses the original Brachistochrone loss
        but is designed to work with SGD optimizer
        
        Args:
            h_list: list of (B, ...) tensors [h0, h1, ..., hT]
            
        Returns:
            L_path: Path loss term
            L_mono: Monotonicity loss term
        """
        return self.brach_loss(h_list)


# Keep the old class name for compatibility but make it use BrachistochroneAdam
BrachistochroneLossProSimple = BrachistochroneAdam
