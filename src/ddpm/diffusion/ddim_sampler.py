"""DDIM sampling utilities for generating DAS samples from trained diffusion models."""
import torch
from tqdm import tqdm

def ddim_sample(model, num_samples, num_steps, device, eta=0.0, beta_min = 1e-4, beta_max=0.02):
    """Generate DAS samples with DDIM sampling from a trained noise-prediction model."""
    device = torch.device(device)

    # Recreate the diffusion schedule used during training.
    betas = torch.linspace(beta_min, beta_max,1000).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    # Select a reduced sequence of reverse timesteps for DDIM.
    timesteps = torch.linspace(999,0,num_steps).round().long().to(device)

    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 256, 512, device= device)
        for i, t in enumerate(timesteps) :
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_tensor)
            
            # Predict the clean sample x0 from the current noisy sample xt.
            x0 = (x - torch.sqrt(1.0 - alpha_bars[t]) * noise_pred) / torch.sqrt(alpha_bars[t])

            if i < num_steps - 1:
                s = timesteps[i + 1]
                sigma_t = eta * torch.sqrt((1.0 - alpha_bars[s]) / (1.0 - alpha_bars[t])) * torch.sqrt(1.0 - alpha_bars[t] / alpha_bars[s])
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bars[s]) * x0 + torch.sqrt(1.0 - alpha_bars[s] - sigma_t**2) * noise_pred + sigma_t * noise

            else:
                x = x0
    return x

def generate_ddim_samples(model, num_samples, batch_size, num_steps, device):
    generated_samples = []
    prog = tqdm(range(0, num_samples, batch_size), desc="DDIM Samples")

    for s in prog:

        current_size = min(batch_size, num_samples-s)
        batch = ddim_sample(model = model, num_samples=current_size, num_steps = num_steps, device=device)
        generated_samples.append(batch.cpu())
        prog.set_postfix(generated = min(s + batch_size, num_samples)) 

    return torch.cat(generated_samples, dim=0)    