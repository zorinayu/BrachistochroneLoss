import torch

def stft_power(x, n_fft=256, hop=128):
    """
    Compute STFT power spectrogram.
    x: (B, T) input signal
    returns: (B, F, frames) power spectrogram
    """
    # Compute STFT
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop, return_complex=True)
    
    # Convert to power spectrogram
    power = torch.abs(stft) ** 2
    
    return power
