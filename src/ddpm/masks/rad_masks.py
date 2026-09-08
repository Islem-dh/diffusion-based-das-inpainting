import math
import torch

# internal helper functions

def _uniform(low, high, device):
    return low + (high-low) * torch.rand(1, device=device).item()

def _randint(low, high, device):
    return torch.randint(low, high+1, (1,), device=device).item()

def _empty_mask(B, H, W, device):
    return torch.zeros(B, 1, H, W, device=device)

def _fade(t):
    return 6* t**5 - 15 * t**4 +10 * t**3

def _perlin_noise_2d(H, W, res_y, res_x, device):
    angles = 2 * math.pi * torch.rand(res_y + 1, res_x + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)

    gy = y / H * res_y
    gx = x / W * res_x

    iy = torch.floor(gy).long()
    ix = torch.floor(gx).long()

    fy = gy - iy
    fx = gx - ix

    iy = torch.clamp(iy, max=res_y - 1)
    ix = torch.clamp(ix, max=res_x - 1)

    iy_grid, ix_grid = torch.meshgrid(iy,ix,indexing="ij")

    fy_grid, fx_grid = torch.meshgrid(fy,fx,indexing="ij")

    g00 = gradients[iy_grid, ix_grid]
    g10 = gradients[iy_grid + 1, ix_grid]
    g01 = gradients[iy_grid, ix_grid + 1]
    g11 = gradients[iy_grid + 1, ix_grid + 1]

    d00 = torch.stack((fx_grid, fy_grid),dim=-1)

    d10 = torch.stack((fx_grid, fy_grid - 1),dim=-1)

    d01 = torch.stack((fx_grid - 1, fy_grid),dim=-1)

    d11 = torch.stack((fx_grid - 1, fy_grid - 1),dim=-1)

    n00 = torch.sum(g00 * d00, dim=-1)
    n10 = torch.sum(g10 * d10, dim=-1)
    n01 = torch.sum(g01 * d01, dim=-1)
    n11 = torch.sum(g11 * d11, dim=-1)

    u = _fade(fx_grid)
    v = _fade(fy_grid)

    nx0 = n00 * (1 - u) + n01 * u
    nx1 = n10 * (1 - u) + n11 * u

    noise = nx0 * (1 - v) + nx1 * v

    noise = noise - noise.min()

    denominator = noise.max()

    if denominator > 0: noise = noise / denominator

    return noise

#Box mask

def box(B, H, W, device, area_range=(0.20,0.80), aspect_ratio_range=(0.10,10.0)):
    masks = _empty_mask(B, H, W, device)
    total_area = H * W
    for b in range (B):
        area_fract = _uniform(area_range[0], area_range[1], device)
        target_area = area_fract * total_area

        log_min = math.log(aspect_ratio_range[0])
        log_max = math.log(aspect_ratio_range[1])

        aspect_ratio = math.exp(_uniform(log_min, log_max, device))

        box_h = int(round(math.sqrt(target_area / aspect_ratio)))
        box_w = int(round(math.sqrt(target_area * aspect_ratio)))

        box_h = max(1, min(box_h, H))
        box_w = max(1, min(box_w, W))

        y0 = _randint(0, H - box_h, device)
        x0 = _randint(0, W - box_w, device)

        masks[b, 0, y0:y0 + box_h, x0:x0 + box_w] = 1.0

    return masks  

#Channel strip mask

def channel_strip(B, H, W, device, width_range=(0.10, 0.40), count_range=(1,4)):
    masks = _empty_mask(B, H, W, device)
    for b in range(B):
          
        count = _randint(count_range[0], count_range[1], device)
          
        for _ in range(count):
                    
            fract = _uniform(width_range[0], width_range[1], device)
            strip_h = max(1, int(round(fract*H)))
            strip_h = min (strip_h, H)
          
            y0 = _randint(0, H - strip_h, device)
            masks[b, 0, y0:y0 + strip_h, :] = 1.0

    return masks

#Time strip mask

def time_strip(B, H, W, device, width_range=(0.10, 0.40), count_range=(1,4)):
    masks =  _empty_mask(B, H, W, device)
    for b in range(B):
              
        count = _randint(count_range[0], count_range[1], device)
              
        for _ in range(count):
                        
            fract = _uniform(width_range[0], width_range[1], device)
            strip_w = max(1, int(round(fract*W)))
            strip_w = min (strip_w, W)
              
            x0 = _randint(0, W - strip_w, device)
            masks[b, 0,:, x0:x0 + strip_w] = 1.0

    return masks

def oriented_band(B, H, W, device,  angle_range=(-75.0, 75.0), thickness_range=(0.10,0.30), length_range=(0.80, 1.60), count_range=(1,4)):
    masks = _empty_mask(B, H, W, device)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )

    diag = math.sqrt(H**2 + W**2)

    for b in range(B):
        count = _randint(count_range[0], count_range[1], device)
                      
        for _ in range(count):
                                
            angle = math.radians(_uniform(angle_range[0], angle_range[1], device))
            thickness_fract = _uniform(thickness_range[0], thickness_range[1], device)
            length_fract = _uniform(length_range[0], length_range[1], device)

            thickness = max(1.0, thickness_fract*H)
            length = length_fract * diag

            center_y = _uniform(0, H -1, device)
            center_x = _uniform(0, W-1, device)

            dx = xx - center_x
            dy = yy - center_y
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            along = dx * cos_a + dy * sin_a
            perp = -dx *sin_a + dy*cos_a

            band = (torch.abs(along) <= length/2 ) & (torch.abs(perp) <= thickness/2)
            masks[b,0][band] = 1.0

    return masks

#Anisptropic perlin mask 

def anisotropic_perlin(B, H, W, device, channel_range=(1,8), time_range=(1,16), area_range=(0.25, 0.80)):
    masks = _empty_mask(B, H, W, device)

    for b in range(B):

        res_y = _randint(channel_range[0], channel_range[1], device)

        res_x = _randint(time_range[0], time_range[1],device)

        noise = _perlin_noise_2d(H=H, W=W, res_y=res_y, res_x=res_x, device=device)

        area_fraction = _uniform(area_range[0], area_range[1], device)

        threshold = torch.quantile(noise.flatten(),1.0 - area_fraction,
        )

        masks[b, 0] = (noise >= threshold)

    return masks

#Random RAD mask

def random_rad_mask(B, H, W, device, probabilities=None):
    if probabilities is None:
        probabilities = {
            "box": 0.25,
            "channel_strip": 0.20,
            "time_strip": 0.20,
            "oriented_band": 0.25,
            "perlin": 0.10,
        }

    names = list(probabilities.keys())
    probs = torch.tensor([probabilities[name] for name in names], device=device )
    probs = probs / probs.sum()

    choices = torch.multinomial(probs, num_samples=B, replacement=True )

    masks = _empty_mask(B, H, W, device)

    for b, choice in enumerate(choices.tolist()):

        mask_type = names[choice]

        if mask_type == "box":

            mask = box(B=1, H=H, W=W, device=device)

        elif mask_type == "channel_strip":

            mask = channel_strip(B=1, H=H, W=W, device=device)

        elif mask_type == "time_strip":

            mask = time_strip(B=1, H=H, W=W, device=device)

        elif mask_type == "oriented_band":

            mask = oriented_band(B=1, H=H, W=W, device=device)

        elif mask_type == "perlin":

            mask = anisotropic_perlin(B=1, H=H, W=W, device=device)

        masks[b:b + 1] = mask

    return masks