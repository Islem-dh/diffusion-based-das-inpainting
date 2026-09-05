"""Plotting utilities for generated DAS samples."""

import matplotlib.pyplot as plt
from ddpm.utils.normalizer import denormalize

def plot_denormalized_samples(samples):
    denormalized = denormalize(samples)
    num_images = samples.shape[0]

    fig, axes = plt.subplots(1, num_images, figsize = (7*num_images,4))

    if num_images ==1 :
          axes = [axes]
          
    for i in range(num_images):
            axes[i].imshow(denormalized[i,0].detach().cpu(), aspect="auto", cmap="seismic" ) 
            axes[i].axis("off")
            axes[i].set_title("denormalized generated DAS sample")
    plt.tight_layout()
    plt.show() 