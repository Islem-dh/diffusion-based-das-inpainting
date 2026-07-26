import torch
import math
s=50.0
A_max= 12350.0
denom = math.asinh(A_max/s)

def normalize(x):
    return torch.arcsinh(x  /s) / denom

def denormalize(y):
    return torch.sinh(y*denom) * s 

def get_config():
    return{
        "type": "arcsinh",
        "s": s,
        "A_max": A_max,
    }