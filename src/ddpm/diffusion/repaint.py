"""RePaint sampling utilities for masked DAS inpainting with diffusion models."""

import torch
def repaint_schedule(T, jump_length=10, jump_n_sample=3):
    """Create the RePaint reverse/forward timestep schedule with resampling jumps."""

    # Build reverse diffusion schedule and insert forward jumps for resampling.
    start_resampling = 300
    jumps = {
        j: jump_n_sample -1
        for j in range(0, T-jump_length, jump_length)
    } 
    t = T-1
    times = [t]
    while t >=0:
        t-=1
        times.append(t)
        if((jumps.get(t,0) > 0)and t<= start_resampling - jump_length):
            jumps[t] -= 1
            for _ in range(jump_length):
                t +=1
                times.append(t)
    return times

def forward_one_step(x, scheduler, current, next):
    """Move a sample forward in diffusion time from one timestep to a later one."""

    alpha_bar_current = scheduler.alpha_bar[current].to(device=x.device,dtype=x.dtype)

    alpha_bar_next = scheduler.alpha_bar[next].to(device=x.device,dtype=x.dtype)

    # Compute q(x_next | x_current).
    alpha_ratio = alpha_bar_next / alpha_bar_current
    alpha_ratio = torch.clamp(alpha_ratio, 0.0, 1.0)

    noise = torch.randn_like(x)

    return (
        torch.sqrt(alpha_ratio) * x
        + torch.sqrt(1.0 - alpha_ratio) * noise
    )



def sample_repaint(model, scheduler, x0, mask, jump_length=10,jump_n_sample=3, steps_to_save = [999,600,300,100,0]):
    """Inpaint masked DAS data using RePaint resampling and a trained diffusion model."""

    print(x0.mean())

    with torch.no_grad():
        model.eval()
        B=x0.shape[0]
        T=scheduler.betas.shape[0]
        mask=mask.to(x0.device)

        times = repaint_schedule(T=T, jump_length=jump_length, jump_n_sample=jump_n_sample)
        xt=torch.randn_like(x0)
        snapshots = {}
        save_steps = set(steps_to_save)
        if T-1 in save_steps:
            snapshots[T-1] = xt.detach().clone()

        for current, next in zip(times[:-1], times[1:]):
            if next < current : 

                t = torch.full((B,),current,device=x0.device,dtype=torch.long)

                # Diffuse the known region to the current timestep.
                gaussian_noise=torch.randn_like(x0)
                x_known_curr=scheduler.add_noise(x0,gaussian_noise,t)

                # Combine known data with the current unknown-region estimate.
                xt = mask*x_known_curr +(1.0-mask)*xt

                #ddpm prediction
                predicted_noise=model(xt,t)
                xt_unknown,x0_pred = scheduler.reverse_sampling_step(xt,predicted_noise,current)

                if next>=0:
                    
                    t_prev=torch.full((B,),next,device=x0.device,dtype=torch.long)
                    gaussian_noise_next=torch.randn_like(x0)
                    x_known_next = scheduler.add_noise(x0,gaussian_noise_next,t_prev)
                    
                else:
                    x_known_next=x0

                # Merge the known region with the reverse-sampled unknown region.
                xt = mask*x_known_next +(1.0-mask)*xt_unknown

                if next in save_steps:
                    snapshots[next] = xt.detach().clone()
                
            elif next > current:
                # Re-noise the sample during a RePaint forward jump.
                xt = forward_one_step(
                    xt,
                    scheduler,
                    current,
                    next
                )
                
        return xt, snapshots, times