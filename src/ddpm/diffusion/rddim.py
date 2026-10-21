"""RAD DDIM sampling utilities for generating DAS samples."""

import torch
from tqdm import tqdm

from ddpm.masks.rad_masks import random_rad_mask


def rddim_sample(model,scheduler,num_samples,num_steps,device,eta=0.0,generator=None,timesteps_shown=None):
    """Generate DAS samples with RDDIM sampling from a trained RAD model."""

    device = torch.device(device)

    masks = random_rad_mask(B=num_samples,H=256,W=512,device=device,generator=generator)

    timesteps = torch.linspace(scheduler.timesteps - 1,0,num_steps,device=device).round().long()
    timesteps = torch.unique_consecutive(timesteps)

    if timesteps_shown is None: timesteps_shown=[]

    saved_steps={}

    model.eval()

    with torch.no_grad():

        x = torch.randn(num_samples,1,256,512,device=device,generator=generator)

        for i,t in enumerate(timesteps):

            t_tensor = torch.full((num_samples,),t.item(),device=device,dtype=torch.long)

            _,_,a_bar_t,b_bar_t,_ = scheduler.get_spatial_params(masks,t_tensor)

            noise_pred = model(x,b_bar_t)

            x0 = (x - torch.sqrt(b_bar_t)*noise_pred) / torch.sqrt(torch.clamp(a_bar_t,min=1e-12))

            if t.item() in timesteps_shown:
                saved_steps[t.item()] = x.detach().cpu().clone()

            if i == len(timesteps)-1:
                x=x0
                saved_steps[0] = x.detach().cpu().clone()
                break

            s = timesteps[i+1]

            s_tensor = torch.full((num_samples,),s.item(),device=device,dtype=torch.long)

            _,_,a_bar_s,b_bar_s,_ = scheduler.get_spatial_params(masks,s_tensor)

            sigma_t = eta * torch.sqrt(torch.clamp((1.0-a_bar_s)/torch.clamp(1.0-a_bar_t,min=1e-12)*(1.0-a_bar_t/torch.clamp(a_bar_s,min=1e-12)),min=0.0))

            direction = torch.sqrt(torch.clamp(b_bar_s-sigma_t**2,min=0.0))*noise_pred

            x = torch.sqrt(a_bar_s)*x0 + direction

            if eta > 0.0:
                noise = torch.randn(x.shape,device=device,generator=generator)
                x = x + sigma_t*noise

    return x,masks,saved_steps


def generate_rddim_samples(model,scheduler,num_samples,batch_size,num_steps,device,eta=0.0,seed=None):
    """Generate many RAD DDIM samples in batches."""

    generated_samples=[]

    generator=None

    if seed is not None: generator=torch.Generator(device=device).manual_seed(seed)

    prog=tqdm(range(0,num_samples,batch_size),desc="RDDIM Samples")

    for start in prog:

        current_size=min(batch_size,num_samples-start)

        batch,_,_=rddim_sample(model=model,scheduler=scheduler,num_samples=current_size,num_steps=num_steps,device=device,eta=eta,generator=generator)

        generated_samples.append(batch.cpu())

        prog.set_postfix(generated=min(start+batch_size,num_samples))

    return torch.cat(generated_samples,dim=0)