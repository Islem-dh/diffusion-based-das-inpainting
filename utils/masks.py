import torch
import random

def random_circle_mask(x, min_radius, max_radius, min_pixel_value):
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
    #radius= random.uniform(min_radius,max_radius)
    radius= min_radius + (max_radius - min_radius)* torch.rand(B,device=x.device)
    radius=radius[:,None,None]
    circle= torch.sqrt((i-center_i)**2 +(j-center_j)**2)
    missing_region = (circle <= radius).float()
    mask=1.0-missing_region
    mask=mask.unsqueeze(1)
    return mask

    '''masks=[]
    for b in range (B) :
        img = x[b].squeeze(0)
        pixels= img.flatten().clone()
        pixels[pixels< min_pixel_value]=0.0
        center_pixel= torch.multinomial(pixels, num_samples=1).item()
        #center_pixel= W*i+j
        center_i= center_pixel // W
        center_j= center_pixel % W
        
        radius= random.uniform(min_radius, max_radius)
        sigma=radius*random.uniform(factor_min,factor_max)
        #circle= torch.sqrt((i-center_i)**2 +(j-center_j)**2)
        blob= torch.exp(-((i-center_i)**2 +(j-center_j)**2)/(2*sigma**2))
        #missing_region = (circle <= radius).float
        mask=1.0-blob
        masks.append(mask)
    masks= torch.stack(masks,dim=0).unsqueeze(1)
    return masks'''




