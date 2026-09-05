"""Utilities for collecting evaluation samples."""

import torch

def collect_samples(loader,num_samples):
    samples = []
    for batch in loader :
        samples.append(batch)
        if sum (x.shape[0] for x in samples) >= num_samples :
            break
    return torch.cat(samples, dim=0)[:num_samples]