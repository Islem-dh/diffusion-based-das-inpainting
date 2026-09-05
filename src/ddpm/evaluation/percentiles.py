"""Percentile comparison utilities for DAS samples."""

import numpy as np
from ddpm.utils.normalizer import denormalize

def compare_percentiles(real_samples, generated_samples):
    percentiles=[50,60,99,99.9,99.99]

    #denomalized
    real = denormalize(real_samples).detach().cpu().numpy()
    generated = denormalize(generated_samples).detach().cpu().numpy()

    real_percentiles = np.percentile(np.abs(real).reshape(-1), percentiles)
    generated_percentiles = np.percentile(np.abs(generated).reshape(-1), percentiles)

    for i in range (len(percentiles)):
        print(
            f"{percentiles[i]}% |" f" real DAS: {real_percentiles[i]:.2f} |" f" generated DAS {generated_percentiles[i]:.2f}"
        )