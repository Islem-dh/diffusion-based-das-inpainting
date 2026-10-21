"""Spatial noise embedding used by RAD diffusion model"""

import torch
import torch.nn as nn

class SpatialSinusoidalEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim=dim
        self.theta=theta

    def forward(self,x):
        device = x.device
        half_dim = self.dim //2

        i=torch.arange(half_dim, device=device)
        frequencies= self.theta**(-i/(half_dim-1))
        scaled_x = x*frequencies[None,:,None,None]

        embedding=torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=1)
        return embedding

class SpatialNoiseEmbedding(nn.Module):
    """project simus spatial noise embedding throug 1x1 conv"""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.sinusoidal_emb=SpatialSinusoidalEmb(dim=dim, theta=theta)
        '''self.layers = nn.Sequential(
            nn.Conv2d(dim, noise_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(noise_dim, noise_dim, kernel_size=1)
        )'''

    def forward(self,x):
        x=self.sinusoidal_emb(x)
        #return self.layers(x) 
        return x   