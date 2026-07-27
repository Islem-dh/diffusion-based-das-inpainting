import torch
import math
s=50.0
A_max= 12350.0
denom = math.asinh(A_max/s)

def normalize(x):
    return torch.arcsinh(x/s) / denom

def denormalize(y):
    return torch.sinh(y*denom) * s

def normalize_shift(x):
    mu = torch.median(x)
    normalized = torch.arcsinh((x-mu)/s) / denom
    return normalized, mu

def denormalize_shift(y,mu):
    mu = mu[:, None, None, None]
    return torch.sinh(y*denom) *s + mu

def get_config():
    return{
        "type": "arcsinh",
        "s": s,
        "A_max": A_max,
    }