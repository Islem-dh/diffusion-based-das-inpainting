"""Soft Gaussian hole around a random bright pixel (MNIST experiments, pre-thesis)."""

import torch

def random_gaussian_mask(x, min_radius, max_radius, min_pixel_value, factor_min, factor_max):
    B,C,H,W = x.shape
    i, j = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W,device=x.device),
                            indexing="ij")
    i= i.unsqueeze(0)
    j=j.unsqueeze(0)
    img=x[:,0]
    pixels=img.flatten(start_dim=1).clone()
    pixels[pixels<min_pixel_value]=0.0
    center_pixel=torch.multinomial(pixels,num_samples=1).squeeze(1)
    center_i=(center_pixel//W)[:,None,None]
    center_j=(center_pixel%W)[:,None,None]
    radius= min_radius + (max_radius - min_radius)* torch.rand(B,device=x.device)
    factor= factor_min + (factor_max - factor_min)* torch.rand(B,device=x.device)
    sigma=(radius*factor)[:,None,None]
    blob= torch.exp(-((i-center_i)**2 +(j-center_j)**2)/(2*sigma**2))
    mask=1.0-blob
    mask=mask.unsqueeze(1)
    return mask