import torch
import numpy as np

def linear_beta_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Linear schedule for variance in diffusion model.
    Args:
        T (int): Total number of diffusion steps.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.
    Returns:
        torch.Tensor: Beta values (variance schedule) for each time step.
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """
    Cosine schedule for variance in diffusion model.
    Args:
        T (int): Total number of diffusion steps.
        s (float): Small constant to prevent division by zero at the beginning.
    Returns:
        torch.Tensor: Beta values (variance schedule) for each time step.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f_t / f_t[0]
    
    # Calculate the beta values from the alphas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)  # Clip values for stability
    
    return betas.float()


