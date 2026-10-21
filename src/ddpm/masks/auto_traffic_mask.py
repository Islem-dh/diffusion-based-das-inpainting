"""Automatic traffic masks for real DAS recordings (no hand-drawn lines).

Idea: traffic is narrow across channels and long in time, an earthquake is broad across channels
and short in time. A point is marked as traffic if its energy is clearly above the level of the
neighbouring channels at the same time and the structure lasts for several seconds. Detected
pieces are fitted with straight lines (vehicles drive at almost constant speed) and extended to
the image border, so that no unmasked end of a traffic line remains.

Mask convention as everywhere in this project: 1 = keep, 0 = hole.
"""

import numpy as np
import torch
import torch.nn.functional as F


def block_energy(seg, blk=25, device="cpu"):
    x = torch.as_tensor(seg, dtype=torch.float32, device=device)
    H, W = x.shape
    T_blk = W // blk
    E = x[:, :T_blk*blk].pow(2).reshape(H, T_blk, blk).mean(-1).sqrt()
    E = E / E.median(dim=1, keepdim=True).values.clamp(min=1e-12)
    return E[None, None]


def detect_traffic(E, fs=100, blk=25, k_thr=3.0, nb=129, min_len_s=3):

    T_blk = E.shape[-1]

    # typical level of the neighbour channels at the same time (median over +-nb//2 channels)
    Ep = F.pad(E, (0, 0, nb//2, nb//2), mode="reflect")
    bg = Ep.unfold(2, nb, 1).median(dim=-1).values
    det = (E / bg.clamp(min=1e-12) > k_thr).float()

    # persistence: lines shift ~1-2 channels per block -> tolerate +-8 channels, require >= L blocks in a row
    L = int(min_len_s*fs/blk)
    d = F.max_pool2d(det, (17, 1), 1, (8, 0))
    run = F.avg_pool2d(d, (1, L), 1, (0, L//2), count_include_pad=False)[..., :T_blk]
    long = F.max_pool2d((run >= 0.9).float(), (1, L+1), 1, (0, L//2))[..., :T_blk]   # extend back to the segment ends
    return (det*long)[0, 0]


def widen_to_mask(core, W, blk=25, hw=12):
    keep = F.max_pool2d(core[None, None].float(), (2*hw+1, 3), 1, (hw, 1))
    keep = keep.repeat_interleave(blk, dim=-1)
    keep = F.pad(keep, (0, W - keep.shape[-1], 0, 0), mode="replicate")   # rest of the samples (W not divisible by blk)
    return keep[0, 0] > 0


def label_components(b):
    lab = np.zeros(b.shape, np.int32); n = 0; Hb, Tb = b.shape
    for c_, t_ in zip(*np.nonzero(b)):
        if lab[c_, t_]: continue
        n += 1; lab[c_, t_] = n; stack = [(c_, t_)]
        while stack:
            c, t = stack.pop()
            for dc in (-1, 0, 1):
                for dt in (-1, 0, 1):
                    cn, tn = c+dc, t+dt
                    if 0 <= cn < Hb and 0 <= tn < Tb and b[cn, tn] and not lab[cn, tn]:
                        lab[cn, tn] = n; stack.append((cn, tn))
    return lab, n


def extend_lines(core, E, W, fs=100, blk=25, hw=12, min_len_s=3, ext_max_s=10, max_resid=6, max_slope=5):
    H, T_blk = core.shape
    dev = core.device

    # join fragments slightly across channels, then label the connected pieces
    core_c = (F.max_pool2d(core[None, None].float(), (5, 3), 1, (2, 1))[0, 0] > 0).cpu().numpy()
    Eb = E[0, 0].cpu().numpy()
    lab, n = label_components(core_c)

    cs_all, ts_all = np.nonzero(lab); ids = lab[cs_all, ts_all]
    order = np.argsort(ids); cs_all, ts_all, ids = cs_all[order], ts_all[order], ids[order]
    splits = np.flatnonzero(np.diff(ids)) + 1

    band = torch.zeros(H, W, dtype=torch.bool, device=dev)
    cc = torch.arange(H, device=dev)[:, None].float()
    n_ext = 0
    for cs_i, ts_i in zip(np.split(cs_all, splits), np.split(ts_all, splits)):
        if len(ts_i) < 2 or (ts_i.max()-ts_i.min())*blk/fs < min_len_s: continue   
        a, b = np.polyfit(ts_i, cs_i, 1, w=np.sqrt(Eb[cs_i, ts_i]))             
        if np.std(cs_i - (a*ts_i+b)) > max_resid or abs(a) > max_slope: continue  
        for g0, g1 in [(0, ts_i.min()), (ts_i.max(), T_blk-1)]:                    
            if g1 - g0 <= 0 or (g1-g0)*blk/fs > ext_max_s: continue
            s0 = 0 if g0 == 0 else int(g0*blk)
            s1 = W if g1 == T_blk-1 else int((g1+1)*blk)
            tt = torch.arange(s0, s1, device=dev).float()
            c_line = a*((tt - blk/2)/blk) + b                                  
            band[:, s0:s1] |= (torch.abs(cc - c_line[None, :]) <= hw)
            n_ext += 1
    return band, n, n_ext


def auto_traffic_mask(seg, fs=100, blk=25, k_thr=3.0, nb=129, hw=12, min_len_s=3,
                      extend=True, ext_max_s=10, max_resid=6, max_slope=5, device="cpu", return_info=False):
    H, W = seg.shape
    E = block_energy(seg, blk, device)
    core = detect_traffic(E, fs, blk, k_thr, nb, min_len_s)
    hole = widen_to_mask(core, W, blk, hw)
    hole_old = hole.clone()

    n_comp = n_ext = 0
    if extend:
        band, n_comp, n_ext = extend_lines(core, E, W, fs, blk, hw, min_len_s, ext_max_s, max_resid, max_slope)
        hole |= band

    mask = (~hole).float()[None, None].cpu()
    if not return_info:
        return mask
    info = {"energy": E[0, 0].cpu(), "core": core.cpu(),
            "mask_no_ext": (~hole_old).float()[None, None].cpu(),
            "n_components": n_comp, "n_extensions": n_ext,
            "hole_share": (mask == 0).float().mean().item(),
            "hole_share_no_ext": (hole_old).float().mean().item()}
    return mask, info