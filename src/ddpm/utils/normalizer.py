"""Normalization utilities for DAS amplitudes using an arcsinh transform."""

import torch
import math

#parameters
s=50.0
A_max= 12350.0
denom = math.asinh(A_max/s)

def normalize(x):
    """Normalize DAS amplitudes with the configured arcsinh transform."""
    return torch.arcsinh(x/s) / denom

def denormalize(y):
    """Invert the configured arcsinh normalization."""
    return torch.sinh(y*denom) * s

def normalize_shift(x):
    """Mean-center a sample before applying arcsinh normalization."""
    mu = torch.median(x)
    normalized = torch.arcsinh((x-mu)/s) / denom
    return normalized, mu

def denormalize_shift(y,mu):
    """Invert shifted normalization using the stored sample median."""
    mu = mu[:, None, None, None]
    return torch.sinh(y*denom) *s + mu

def get_config():
    """Return the normalization configuration used by the project."""
    return{
        "type": "arcsinh",
        "s": s,
        "A_max": A_max,
    }