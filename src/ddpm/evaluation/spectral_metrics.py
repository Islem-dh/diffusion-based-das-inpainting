"""Spectral evaluation of RAD inpainting on hidden channels."""

import torch

from ddpm.masks.rad_masks import channel_strip
from ddpm.diffusion.rad_inpainting_scaled import inpaint_scaled


def collect_hole_rows(model,scheduler,loader,device,scale=1.0,n_batches=3,seed=0):
    """Hide whole channels, inpaint them and return true (T) and inpainted (G) rows."""

    g=torch.Generator(device=device).manual_seed(seed)
    T=[];G=[]

    for k,batch in enumerate(loader):
        if k>=n_batches: break
        x0=batch.to(device); B,C,H,W=x0.shape
        mask=channel_strip(B,H,W,device,generator=g)
        res,_,_=inpaint_scaled(model,scheduler,x0,mask,device,scale=scale,generator=g)
        rows=(mask[:,0,:,0]==0)
        T.append(x0[:,0][rows].cpu()); G.append(res[:,0][rows].cpu())

    return torch.cat(T),torch.cat(G)


def psd(x):
    """Mean power spectrum along time (last axis)."""
    return (torch.fft.rfft(x,dim=-1).abs()**2).mean(0)


def band_ratios(T,G,sel=None):
    """Power ratio inpainted/true in low, mid and high frequency bands, plus RMS ratio."""

    if sel is None: sel=torch.ones(T.shape[0],dtype=torch.bool)

    pt,pg=psd(T[sel]),psd(G[sel])
    bands={"low 0-45":slice(0,46),"mid 50-140":slice(50,141),"high 150-256":slice(150,257)}

    out={name:round((pg[b].sum()/pt[b].sum()).item(),2) for name,b in bands.items()}
    out["RMS"]=round((G[sel].pow(2).mean().sqrt()/T[sel].pow(2).mean().sqrt()).item(),2)

    return out