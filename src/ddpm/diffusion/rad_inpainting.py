"""RAD inpainting inference."""

import torch


def rad_inpaint(model,scheduler,x0,mask,device,generator=None,timesteps_shown=None):
    """Inpaint masked regions using the RAD Phase-1 reverse process."""

    x0=x0.to(device)
    mask=mask.to(device)

    B=x0.shape[0]

    if timesteps_shown is None: timesteps_shown=[]

    saved_steps={}

    epsilon=torch.randn(x0.shape,device=device,generator=generator)

    t_start=torch.full((B,),scheduler.ph1_steps-1,device=device,dtype=torch.long)

    x,b_bar_start=scheduler.add_noise(x0=x0,epsilon=epsilon,mask=mask,t=t_start)

    x_start=x.detach().cpu().clone()

    model.eval()

    with torch.no_grad():

        for step in reversed(range(scheduler.ph1_steps)):

            t=torch.full((B,),step,device=device,dtype=torch.long)

            _,_,_,b_bar_t,_=scheduler.get_spatial_params(mask,t)

            if step in timesteps_shown:
                saved_steps[step]=x.detach().cpu().clone()

            noise_pred=model(x,b_bar_t)

            x,_=scheduler.reverse_sampling_step(x,noise_pred,mask,t)

    return x,x_start,saved_steps