"""Utilities for visualizing selected DDPM reverse-sampling steps."""

import matplotlib.pyplot as plt
import torch

def plot_sampling_steps(model, scheduler, device, num_images=2, timesteps_shown=(999, 600, 300, 100, 0)):
    """Generate random samples and visualize selected reverse-sampling timesteps."""

    xt = torch.randn(num_images, 1,256,512, device = device)

    num_rows = len(timesteps_shown)
    fig, axes = plt.subplots(num_rows, num_images, figsize = (7*num_images, 3*num_rows))

    model.eval()

    with torch.no_grad():
        for step in reversed(range(scheduler.betas.shape[0])):
            if step in timesteps_shown:
                row= timesteps_shown.index(step)
                for i in range(num_images):
                    axes[row,i].imshow(xt[i,0].detach().cpu(), aspect="auto", cmap="seismic") 
                    axes[row,i].axis("off")
                    axes[row,i].set_title(f"t={step}")
            t=torch.full((num_images,),step, device=device, dtype= torch.long)
            predicted_noise=model(xt,t)
            xt,x0_predicted=scheduler.reverse_sampling_step(xt,predicted_noise,step)
    plt.tight_layout()
    plt.show()

    return xt   