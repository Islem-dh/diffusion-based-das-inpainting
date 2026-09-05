"""Mask-generation utilities for line-shaped traffic gaps in DAS data."""

import torch

def line_mask(h, w, s, e, r, device="cpu"):
    xs, ys = s
    xe, ye = e
    y = torch.arange(h, device=device)
    x = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    l1 = xe-xs
    l2 = ye-ys
    length_sqr = l1**2 + l2**2
    projection = ((xx - xs)*l1 +(yy-ys)*l2) /length_sqr
    projection = torch.clamp(projection, 0, 1)
    closest_x = xs + projection*l1
    closest_y = ys + projection*l2
    distance = torch.sqrt((xx - closest_x)**2 + (yy - closest_y)**2)
    traffic_mask = (distance > r).float()
    return traffic_mask

    