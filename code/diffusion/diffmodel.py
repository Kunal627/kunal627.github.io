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
    base = 1.0/ (theta ** (2 * embed_idx / embedding_dim)).unsqueeze(1)
    embeddings = torch.cat([torch.sin(base), torch.cos(base)], dim=1)

    return timesteps.view(-1,1) * embeddings.view(1,embedding_dim)

class TimeStepEmbedding(nn.Module):
    def __init__(self, embedding_dim=16, out_channels=8, theta = 10000.0):
        super(TimeStepEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, out_channels)
        )

    def forward(self, x):
        x = sinusoidal_embedding(x, self.embedding_dim, self.theta)
        x = self.proj(x)
        x = x[:, :, None, None]
        return x


class ConvBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, embedding_dim=32, pool=False):
        super(DownSampleBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convblk1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.convblk2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.tsembed  = nn.Linear(embedding_dim, out_channels)
        self.pool = pool
        self.ts_encoder = TimeStepEmbedding(embedding_dim=embedding_dim, out_channels=in_channels)

    def forward(self, x, t):
        x = x + self.ts_encoder(t)
        if self.pool:
            x = self.maxpool(x)
        out = self.convblk1(x)
        out = self.convblk2(out)
        
        # add timestep embedding
        #if t is None:
        #    tsembed = F.silu(self.tsembed(t))
        #    tsembed = tsembed[:, :, None, None]
        #    out += tsembed
        return out
    
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.gn1 = nn.GroupNorm(2, out_channels)

    def forward(self, x):
        out = F.gelu(self.gn1(self.conv1(x)))
        return out
    
class DiffModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, embedding_dim=4):
        super(DiffModel, self).__init__()
        
        self.enc1 = DownSampleBlock(in_channels, 64)
        self.enc2 = DownSampleBlock(64, 128, pool=True)
        self.enc3 = DownSampleBlock(128, 256, pool=True)
        self.enc4 = DownSampleBlock(256, 512, pool=True)
        
        self.bottleneck = DownSampleBlock(512, 1024, pool=True)
        
        self.upconv4 = UpSampleBlock(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        
        self.upconv3 = UpSampleBlock(512, 256)
        self.dec3 = ConvBlock(512, 256)
        
        self.upconv2 = UpSampleBlock(256, 128)
        self.dec2 = ConvBlock(256, 128)
        
        self.upconv1 = UpSampleBlock(128, 64)
        self.dec1 = ConvBlock(128, 64)
        
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)


        
        
    def forward(self, x, t):
        e1 = self.enc1(x, t)
        e2 = self.enc2(e1, t)
        e3 = self.enc3(e2, t)
        e4 = self.enc4(e3, t)

        bottleneck = self.bottleneck(e4, t)
               
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat((e4, up4), dim=1)
        d4 = self.dec4(up4)
        
        up3 = self.upconv3(d4)
        up3 = torch.cat((e3, up3), dim=1)
        d3 = self.dec3(up3)
        
        up2 = self.upconv2(d3)
        up2 = torch.cat((e2, up2), dim=1)
        d2 = self.dec2(up2)
        
        up1 = self.upconv1(d2)
        up1 = torch.cat((e1, up1), dim=1)
        d1 = self.dec1(up1)

        return self.final_layer(d1)
    
#moodel = DiffModel()
#img = torch.rand(4, 3, 32, 32)
#y = moodel(img, torch.Tensor([0.1, .5, 0.1,0.5]))
#print(y.shape)