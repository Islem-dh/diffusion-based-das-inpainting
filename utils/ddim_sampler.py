import torch

def ddim_sample(model, num_samples, num_steps, device, eta=0.0, beta_min = 1e-4, beta_max=0.02):
    device = torch.device(device)
    betas = torch.linspace(beta_min, beta_max,1000).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    timesteps = torch.linspace(999,0,num_steps).round().long().to(device)

    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 256, 512, device= device)
        for i, t in enumerate(timesteps) :
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_tensor)
            x0 = (x - torch.sqrt(1.0 - alpha_bars[t]) * noise_pred) / torch.sqrt(alpha_bars[t])

            if i < num_steps - 1:
                s = timesteps[i + 1]
                sigma_t = eta * torch.sqrt((1.0 - alpha_bars[s]) / (1.0 - alpha_bars[t])) * torch.sqrt(1.0 - alpha_bars[t] / alpha_bars[s])
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bars[s]) * x0 + torch.sqrt(1.0 - alpha_bars[s] - sigma_t**2) * noise_pred + sigma_t * noise

            else:
                x = x0
    return x