import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn1 = nn.GroupNorm(2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.gn2 = nn.GroupNorm(2, out_channels)

    def forward(self, x):
        #identity = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        #out += identity
        return out

class ConvBlock(nn.Nodule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn1 = nn.GroupNorm(2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.gn2 = nn.GroupNorm(2, out_channels)

    def forward(self, x):
        out = F.gelu(self.gn1(self.conv1(x)))
        out = F.gelu(self.gn2(self.conv2(out)))
        return out

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownSampleBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convblk1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.convblk2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.convblk1(out)
        out = self.convblk2(out)
        return out
    
class TimeStepEmbeddingFC(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super(TimeStepEmbeddingFC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn1 = nn.GroupNorm(2, out_channels)

    def forward(self, x):
        out = F.gelu(self.gn1(self.conv1(x)))
        return out