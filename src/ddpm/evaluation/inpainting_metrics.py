"""Evaluation metrics for masked DAS inpainting (RAD and RePaint).

All functions take tensors of shape [B,1,H,W] (channels x time) and a mask of the same shape
with 1 = keep, 0 = hole. They return one value per sample ([B]) unless stated otherwise.

With ground truth (test 1001 + masks):  rms_ratio, zcr_ratio, edge_correlation, edge_jump, band_ratios
Without ground truth (real traffic data): rms_hole_ring, edge_jump, fk_energy, channel_coherence
"""

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _hole(mask):
    """Boolean hole map (True = inpainted pixel)."""
    return mask < 0.5


def _dilate(sel, width):
    """Grow a boolean map by `width` pixels in every direction."""
    return F.max_pool2d(sel.float(), 2*width+1, 1, width) > 0


def _masked_rms(x, sel):
    """RMS of x over the selected pixels, per sample."""
    sel = sel.float()
    return torch.sqrt((x.pow(2)*sel).sum(dim=(1,2,3)) / sel.sum(dim=(1,2,3)).clamp(min=1))


def _masked_corr(a, b, sel):
    """Pearson correlation between a and b over the selected pixels, per sample."""
    sel = sel.float()
    n = sel.sum(dim=(1,2,3)).clamp(min=1)
    ma = ((a*sel).sum(dim=(1,2,3))/n).view(-1,1,1,1)
    mb = ((b*sel).sum(dim=(1,2,3))/n).view(-1,1,1,1)
    da, db = (a-ma)*sel, (b-mb)*sel
    num = (da*db).sum(dim=(1,2,3))
    den = torch.sqrt((da.pow(2).sum(dim=(1,2,3))*db.pow(2).sum(dim=(1,2,3))).clamp(min=1e-12))
    return num/den


def edge_zone(mask, width=10):
    """Hole pixels within `width` pixels of the known region (the hole border)."""
    h = _hole(mask)
    return h & _dilate(~h, width)


def ring_zone(mask, width=16):
    """Known pixels within `width` pixels of the hole (the surroundings of the hole)."""
    h = _hole(mask)
    return ~h & _dilate(h, width)


# ----------------------------------------------------------------------------------------------
# metrics with ground truth
# ----------------------------------------------------------------------------------------------

def rms_ratio(x_true, x_fill, mask):
    """RMS of the fill / RMS of the true signal inside the hole (ideal 1: as loud as the truth)."""
    h = _hole(mask)
    return _masked_rms(x_fill, h)/_masked_rms(x_true, h).clamp(min=1e-12)


def _zero_crossings(x, sel):
    """Number of sign changes along time between neighbouring samples that are both selected."""
    pair = sel[...,1:] & sel[...,:-1]
    change = (torch.sign(x[...,1:]) != torch.sign(x[...,:-1])) & pair
    return change.float().sum(dim=(1,2,3))


def zcr_ratio(x_true, x_fill, mask):
    """Zero-crossing rate of the fill / of the true signal inside the hole (ideal 1: same dominant frequency)."""
    h = _hole(mask)
    return _zero_crossings(x_fill, h)/_zero_crossings(x_true, h).clamp(min=1)


def edge_correlation(x_true, x_fill, mask, width=10):
    """Correlation between fill and truth at the hole border (higher is better)."""
    return _masked_corr(x_true, x_fill, edge_zone(mask, width))


def band_ratios(x_true, x_fill, mask, fs=100.0, bands=((0,9),(9,28),(28,50))):
    """Power of the fill / power of the truth per frequency band [Hz] inside the hole (ideal 1).
    Spectrum along time of the hole-masked signals; both get exactly the same masking. Returns [B, n_bands]."""
    h = _hole(mask).float()
    P_true = torch.fft.rfft(x_true*h, dim=-1).abs().pow(2).sum(dim=(1,2))        # [B, n_freq]
    P_fill = torch.fft.rfft(x_fill*h, dim=-1).abs().pow(2).sum(dim=(1,2))
    f = torch.fft.rfftfreq(x_true.shape[-1], d=1.0/fs).to(x_true.device)
    out = []
    for k,(lo,hi) in enumerate(bands):
        sel = (f >= lo) & ((f <= hi) if k == len(bands)-1 else (f < hi))          # last band includes Nyquist
        out.append(P_fill[:,sel].sum(-1)/P_true[:,sel].sum(-1).clamp(min=1e-12))
    return torch.stack(out, dim=-1)


def known_error(x_true, x_fill, mask):
    """Max. absolute change in the known region (sanity check, should be 0)."""
    k = (~_hole(mask)).float()
    return ((x_fill-x_true).abs()*k).amax(dim=(1,2,3))


# ----------------------------------------------------------------------------------------------
# metrics without ground truth
# ----------------------------------------------------------------------------------------------

def rms_hole_ring(x_fill, mask, width=16):
    """RMS in the hole / RMS in a ring of `width` pixels around it (ideal 1: as loud as the surroundings)."""
    return _masked_rms(x_fill, _hole(mask))/_masked_rms(x_fill, ring_zone(mask, width)).clamp(min=1e-12)


def edge_jump(x_fill, mask, width=16):
    """Mean jump between neighbouring pixels across the hole border / mean jump between known
    neighbours in the ring around the hole (ideal 1: no visible seam at the mask edge).
    Neighbours are taken along channels and along time."""
    h = _hole(mask)
    ring = ring_zone(mask, width)
    border_sum = torch.zeros(x_fill.shape[0], device=x_fill.device); border_n = torch.zeros_like(border_sum)
    ref_sum = torch.zeros_like(border_sum); ref_n = torch.zeros_like(border_sum)
    for dim in (2,3):                                                              # channels, time
        L = x_fill.shape[dim]
        d = (x_fill.narrow(dim,1,L-1) - x_fill.narrow(dim,0,L-1)).abs()
        ha, hb = h.narrow(dim,1,L-1), h.narrow(dim,0,L-1)
        ra, rb = ring.narrow(dim,1,L-1), ring.narrow(dim,0,L-1)
        border = (ha ^ hb).float()                                                 # one pixel hole, one known
        ref = (ra & rb).float()                                                    # both known, near the hole
        border_sum += (d*border).sum(dim=(1,2,3)); border_n += border.sum(dim=(1,2,3))
        ref_sum += (d*ref).sum(dim=(1,2,3)); ref_n += ref.sum(dim=(1,2,3))
    return (border_sum/border_n.clamp(min=1))/(ref_sum/ref_n.clamp(min=1)).clamp(min=1e-12)


def fk_energy(x, fs=100.0, dx=4.0, v_slow=100.0, v_fast=1000.0, f_min=0.5):
    """Energy in the f-k spectrum split by apparent velocity v = |f/k|:
    slow (v < v_slow [m/s], traffic) and fast (v > v_fast [m/s], seismic waves).
    Frequencies below f_min [Hz] are ignored (channel offsets). Returns [B, 2] = (slow, fast).
    Compare before and after removal: slow should drop, fast should stay."""
    X = x[:,0] - x[:,0].mean(dim=(-2,-1), keepdim=True)
    P = torch.fft.fft2(X).abs().pow(2)                                           # [B, H, W]
    k = torch.fft.fftfreq(x.shape[-2], d=dx).to(x.device).abs()                  # wavenumber [1/m] (channels)
    f = torch.fft.fftfreq(x.shape[-1], d=1.0/fs).to(x.device).abs()             # frequency [Hz] (time)
    v = f[None,:]/k[:,None].clamp(min=1e-12)                                      # apparent velocity [m/s]
    valid = (f[None,:] >= f_min).expand_as(v)
    slow = (valid & (v < v_slow)).float(); fast = (valid & (v > v_fast)).float()
    return torch.stack([(P*slow).sum(dim=(1,2)), (P*fast).sum(dim=(1,2))], dim=-1)


def channel_coherence(x, sel, min_len=64):
    """Mean correlation between neighbouring channels along time, using only samples where both
    channels are selected (e.g. sel = hole, or sel = known). Channel pairs with fewer than
    `min_len` common samples are skipped."""
    a, b = x[:,:,1:,:], x[:,:,:-1,:]
    s = (sel[:,:,1:,:] & sel[:,:,:-1,:]).float()
    n = s.sum(dim=-1)                                                              # [B,1,H-1]
    ma = (a*s).sum(dim=-1)/n.clamp(min=1); mb = (b*s).sum(dim=-1)/n.clamp(min=1)
    da, db = (a-ma[...,None])*s, (b-mb[...,None])*s
    r = (da*db).sum(dim=-1)/torch.sqrt((da.pow(2).sum(dim=-1)*db.pow(2).sum(dim=-1)).clamp(min=1e-12))
    ok = (n >= min_len).float()
    return (r*ok).sum(dim=(1,2))/ok.sum(dim=(1,2)).clamp(min=1)


# ----------------------------------------------------------------------------------------------
# collections + summary
# ----------------------------------------------------------------------------------------------

def gt_metrics(x_true, x_fill, mask, fs=100.0, edge_width=10):
    """All metrics with ground truth, per sample, as a dict of tensors [B]."""
    br = band_ratios(x_true, x_fill, mask, fs=fs)
    return {"rms_ratio": rms_ratio(x_true, x_fill, mask),
            "zcr_ratio": zcr_ratio(x_true, x_fill, mask),
            "edge_corr": edge_correlation(x_true, x_fill, mask, edge_width),
            "edge_jump": edge_jump(x_fill, mask),
            "band_low":  br[:,0], "band_mid": br[:,1], "band_high": br[:,2],
            "known_err": known_error(x_true, x_fill, mask)}


def nogt_metrics(x_orig, x_fill, mask, fs=100.0, dx=4.0):
    """All metrics without ground truth, per sample, as a dict of tensors [B].
    x_orig = original data (with traffic), x_fill = inpainted result."""
    fk0, fk1 = fk_energy(x_orig, fs, dx), fk_energy(x_fill, fs, dx)
    h = _hole(mask)
    return {"rms_hole_ring": rms_hole_ring(x_fill, mask),
            "edge_jump":     edge_jump(x_fill, mask),
            "fk_slow_ratio": fk1[:,0]/fk0[:,0].clamp(min=1e-12),     # after / before, should drop
            "fk_fast_ratio": fk1[:,1]/fk0[:,1].clamp(min=1e-12),     # after / before, should stay ~1
            "coh_hole":      channel_coherence(x_fill, h),
            "coh_known":     channel_coherence(x_fill, ~h)}


def summarize(metrics):
    """dict of per-sample tensors -> dict of (mean, std), ignoring NaNs."""
    out = {}
    for k,v in metrics.items():
        v = v.detach().float().cpu(); v = v[~torch.isnan(v)]
        out[k] = (v.mean().item(), v.std().item() if v.numel() > 1 else 0.0)
    return out


def print_table(results, keys=None):
    """results: {method name: summarize(...)} -> one row per method, mean ± std per metric."""
    keys = keys or list(next(iter(results.values())).keys())
    print(f"{'method':22s} | " + " | ".join(f"{k:>15s}" for k in keys))
    for name,s in results.items():
        print(f"{name:22s} | " + " | ".join(f"{s[k][0]:7.3f} ± {s[k][1]:5.3f}" for k in keys))