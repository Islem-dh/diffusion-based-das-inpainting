"""RePaint sampling utilities for masked DAS inpainting with diffusion models (with x0 clipping)."""

import torch
def repaint_schedule(T, jump_length=10, jump_n_sample=3, start_resampling=300):
    """Create the RePaint reverse/forward timestep schedule with resampling jumps."""

    # Build reverse diffusion schedule and insert forward jumps for resampling.
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


def ddpm_step_clipped(xt, predicted_noise, scheduler, t, clip_x0=1.2):
    """One DDPM reverse step x_t -> x_{t-1} with the x0 estimate clipped to [-clip_x0, clip_x0]."""

    beta_t = scheduler.betas[t].to(device=xt.device,dtype=xt.dtype)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = scheduler.alpha_bar[t].to(device=xt.device,dtype=xt.dtype)
    if t > 0:
        alpha_bar_prev = scheduler.alpha_bar[t-1].to(device=xt.device,dtype=xt.dtype)
    else:
        alpha_bar_prev = torch.ones((),device=xt.device,dtype=xt.dtype)

    # x0 estimate from the predicted noise, clipped to the data range
    x0_pred = (xt - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    if clip_x0 is not None:
        x0_pred = torch.clamp(x0_pred, -clip_x0, clip_x0)

    # posterior q(x_{t-1} | x_t, x0) (Ho et al., 2020)
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    mean = coef_x0 * x0_pred + coef_xt * xt

    if t > 0:
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return mean + torch.sqrt(variance) * torch.randn_like(xt), x0_pred
    return mean, x0_pred



def sample_repaint(model, scheduler, x0, mask, jump_length=10,jump_n_sample=3, start_resampling=300, clip_x0=1.2, steps_to_save = [999,600,300,100,0]):
    """Inpaint masked DAS data using RePaint resampling and a trained diffusion model."""

    with torch.no_grad():
        model.eval()
        B=x0.shape[0]
        T=scheduler.betas.shape[0]
        mask=mask.to(x0.device)

        times = repaint_schedule(T=T, jump_length=jump_length, jump_n_sample=jump_n_sample, start_resampling=start_resampling)
        xt=torch.randn_like(x0)
        snapshots = {}
        save_steps = set(steps_to_save)
        if T-1 in save_steps:
            snapshots[T-1] = xt.detach().clone()

        for current, next in zip(times[:-1], times[1:]):
            if next < current :

                t = torch.full((B,),current,device=x0.device,dtype=torch.long)

                #ddpm prediction (with x0 clipping)
                predicted_noise=model(xt,t)
                xt_unknown,x0_pred = ddpm_step_clipped(xt,predicted_noise,scheduler,current,clip_x0=clip_x0)

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