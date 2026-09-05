"""Sinusoidal timestep embeddings used by the diffusion model architectures."""

import torch
import torch.nn as nn

class SinusoidalEmb(nn.Module):
    """Convert diffusion timesteps into sinusoidal embeddings."""

    def __init__(self,dim, theta=10000):
        super().__init__()
        self.dim=dim
        self.theta=theta

    def forward(self,x):
        device=x.device
        half_dim=self.dim //2
        # Create exponentially spaced sinusoidal frequencies.
        i=torch.arange(half_dim,device=device)
        frequencies= self.theta**(-i/(half_dim-1))
        scaled_t=x[:,None]*frequencies[None,:]
        embedding= torch.cat([torch.sin(scaled_t), torch.cos(scaled_t)],dim=-1)
        return embedding

class TimeEmbedding(nn.Module):
    """Project sinusoidal timestep embeddings through an MLP."""

    def __init__(self,dim,time_dim,theta=10000):
        super().__init__()
        self.layers=nn.Sequential(
            SinusoidalEmb(dim=dim, theta=theta),
            nn.Linear(dim,time_dim),
            nn.SiLU(),
            nn.Linear(time_dim,time_dim)
        )
        
    def forward(self,x):
        return self.layers(x)    