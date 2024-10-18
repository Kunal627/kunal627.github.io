import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 16
theta = 10000.0
timesteps = torch.tensor([0, 1, 2])
timesteps = timesteps.unsqueeze(1)

def sinusoidal_embedding(timesteps, embedding_dim, theta=10000.0):
    """
    Generate sinusoidal embeddings for the given timesteps.

    Args:
        timesteps (torch.Tensor): A tensor of shape (batch_size,) containing integer timestep values.
        embedding_dim (int): The dimension of the embedding.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing the sinusoidal embeddings.
    """
    embed_idx = torch.arange(0, embedding_dim // 2, dtype=torch.float32)
    base = theta ** (2 * embed_idx / embedding_dim)

    input = timesteps / base

    embeddings = torch.cat([torch.sin(input), torch.cos(input)], dim=1)

    return embeddings

output = sinusoidal_embedding(timesteps, embedding_dim, theta)
print(output)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim , embedding_dim)

    def forward(self, t):
        # Pass the timestep embeddings through two linear layers with activation
        t = F.silu(self.linear1(t))
        t = self.linear2(t)
        return t

proj  = TimeEmbedding(embedding_dim)
output = proj(output)
print(output)
print(output.shape) 

#class SinusoidalEmbedding(nn.Module):
#    def __init__(self, embedding_dim):
#        super(SinusoidalEmbedding, self).__init__()
#        self.embedding_dim = embedding_dim
#
#    def forward(self, timesteps):
#        return sinusoidal_embedding(timesteps, self.embedding_dim)
#
#class TimestepEmbeddingFC(nn.Module):
#    def __init__(self, embedding_dim, out_dim):
#        super(TimestepEmbeddingFC, self).__init__()
#        self.fc = nn.Sequential(
#            nn.Linear(embedding_dim, out_dim),
#            nn.ReLU(),
#            nn.Linear(out_dim, out_dim)
#        )
#
#    def forward(self, x):
#        return self.fc(x)
#
#class UNet(nn.Module):
#    def __init__(self, in_channels, out_channels, num_filters=64, embedding_dim=128):
#        super(UNet, self).__init__()
#        self.embedding_dim = embedding_dim
#
#        # Sinusoidal embedding and fully connected transformation
#        self.time_embed = SinusoidalEmbedding(embedding_dim)
#        self.fc_embed = TimestepEmbeddingFC(embedding_dim, num_filters)
#
#        # Encoder
#        self.enc1 = self.conv_block(in_channels, num_filters)
#        self.enc2 = self.conv_block(num_filters, num_filters * 2)
#        self.enc3 = self.conv_block(num_filters * 2, num_filters * 4)
#
#        # Decoder
#        self.dec1 = self.conv_block(num_filters * 4, num_filters * 2)
#        self.dec2 = self.conv_block(num_filters * 2, num_filters)
#        self.final = nn.Conv2d(num_filters, out_channels, kernel_size=1)
#
#    def conv_block(self, in_channels, out_channels):
#        return nn.Sequential(
#            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True)
#        )
#
#    def forward(self, x, t):
#        # Get time embedding
#        time_embed = self.time_embed(t)
#        time_embed = self.fc_embed(time_embed)
#        time_embed = time_embed.view(time_embed.size(0), -1, 1, 1)
#
#        # Encoder
#        enc1 = self.enc1(x + time_embed)
#        enc2 = self.enc2(enc1 + time_embed)
#        enc3 = self.enc3(enc2 + time_embed)
#
#        # Decoder
#        dec1 = self.dec1(enc3 + time_embed)
#        dec2 = self.dec2(dec1 + time_embed)
#        out = self.final(dec2)
#
#        return out
#

#def sinusoidal_embedding(timesteps, embedding_dim, theta = 10000.0):
#    """
#    Compute sinusoidal embedding for the given timesteps
#    """
    #assert embedding_dim % 2 == 0, "Embedding dimension must be even"
    #emb_indices = torch.arange(0, embedding_dim // 2, dtype=torch.float32)

    #base = theta ** (2 * emb_indices / embedding_dim)

    # Calculate sine and cosine embeddings
    #sin_embedding = torch.sin(timesteps / base)  # Shape: (batch_size, embedding_dim // 2)
    #cos_embedding = torch.cos(timesteps / base)  # Shape: (batch_size, embedding_dim // 2)

    # Concatenate sine and cosine to form the final embedding
    #embeddings = torch.cat((sin_embedding, cos_embedding), dim=1)  # Shape: (batch_size, dim)

    #return embeddings





#embedding = sinusoidal_embedding(torch.tensor([0]), 4)
#print(embedding)
#print(embedding.shape)


#import torch
#import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
#import torchvision.transforms as transforms
#
#
## Set up parameters for diffusion
#timesteps = 1000  # Number of diffusion steps
#beta_start = 1e-4  # Small amount of initial noise
#beta_end = 0.02    # Final amount of noise
#
## Create a linear schedule for beta values (variance of noise added at each step)
#betas = torch.linspace(beta_start, beta_end, timesteps)
#
## Calculate alpha values based on betas
#alphas = 1.0 - betas
#alphas_cumprod = torch.cumprod(alphas, dim=0)
#
## Function for sampling Gaussian noise
#def sample_gaussian_noise(shape):
#    return torch.randn(shape)
#
## Forward diffusion sampling function
#def forward_diffusion_sample(x_0, t):
#    """
#    Add noise to the input image x_0 at time step t.
#    
#    Parameters:
#        x_0: Original image (batch of images)
#        t: Time step (0 <= t < timesteps)
#        
#    Returns:
#        Noisy image x_t
#    """
#     # Calculate scaling factors for original image and noise
#    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
#    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
#    noise = sample_gaussian_noise(x_0.shape)
#
#    # Generate noisy image
#    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
#    return x_t, noise
#