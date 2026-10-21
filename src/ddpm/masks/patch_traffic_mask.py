import numpy as np
import torch


def chunk_profiles(raw, n_chunks=8, smooth=5, despike=5):
    H, W = raw.shape
    L = W // n_chunks
    e = np.sqrt((raw[:, :L*n_chunks]**2).reshape(H, n_chunks, L).mean(-1)).T
    e = np.log10(np.maximum(e, 1e-12))
    if despike > 1:
        ep = np.pad(e, ((0, 0), (despike//2, despike//2)), mode="edge")
        e = np.median(np.lib.stride_tricks.sliding_window_view(ep, despike, axis=1), axis=-1)
    if smooth > 1:
        k = np.ones(smooth)/smooth
        e = np.stack([np.convolve(np.pad(p, smooth//2, mode="edge"), k, mode="valid") for p in e])
    return e


def low_baseline(p, win=65, q=30):
    H = len(p); h = win//2
    pp = np.pad(p, h, mode="reflect")
    return np.array([np.percentile(pp[c:c+win], q) for c in range(H)])


def find_bands(p, base, k_thr=1.7, min_width=4, max_width=60):
    above = (p - base) > np.log10(k_thr)
    bands = []; c = 0; H = len(p)
    while c < H:
        if above[c]:
            c0 = c
            while c < H and above[c]: c += 1
            if min_width <= c - c0 <= max_width:
                seg = p[c0:c] - base[c0:c]
                w = 10**seg
                centre = (np.arange(c0, c)*w).sum()/w.sum()
                bands.append((centre, (c - c0)/2, seg.max()))
        else:
            c += 1
    return bands


def link_tracks(bands_per_chunk, max_shift=12, max_gap=4):
    tracks = []
    for ci, bands in enumerate(bands_per_chunk):
        used = set()
        for tr in tracks:
            last_ci, last_c, _ = tr[-1]
            gap = ci - last_ci
            if gap < 1 or gap > max_gap+1: continue
            best, best_d = None, max_shift*gap+1
            for bi, (c, hw, _) in enumerate(bands):
                d = abs(c - last_c)
                if bi not in used and d < best_d: best, best_d = bi, d
            if best is not None:
                used.add(best); c, hw, _ = bands[best]; tr.append((ci, c, hw))
        for bi, (c, hw, _) in enumerate(bands):
            if bi not in used: tracks.append([(ci, c, hw)])
    return tracks


def patch_traffic_mask(raw, n_chunks=8, k_thr=1.7, base_win=65, base_q=30, min_width=4, max_width=60,
                       min_chunks=4, min_span=6, max_start=1, max_shift=12, max_gap=4, margin=6, max_half_width=40,
                       return_info=False):
    raw = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
    raw = raw.astype(np.float64)
    H, W = raw.shape
    L = W // n_chunks

    prof = chunk_profiles(raw, n_chunks)
    bands = [find_bands(p, low_baseline(p, base_win, base_q), k_thr, min_width, max_width) for p in prof]
    tracks = [tr for tr in link_tracks(bands, max_shift, max_gap)
              if len(tr) >= min_chunks and tr[-1][0]-tr[0][0]+1 >= min_span and tr[0][0] <= max_start]

    hole = np.zeros((H, W), dtype=bool)
    cc = np.arange(H)[:, None]; tt = np.arange(W)[None, :]
    lines = []
    for tr in tracks:
        t_c = np.array([(ci + 0.5)*L for ci, _, _ in tr]); c_c = np.array([c for _, c, _ in tr])
        a, b = np.polyfit(t_c, c_c, 1) if len(tr) > 1 else (0.0, c_c[0])
        hw = min(max(h for _, _, h in tr) + margin, max_half_width)
        hole |= np.abs(cc - (a*tt + b)) <= hw
        lines.append((a, b, hw))

    mask = torch.from_numpy(~hole).float()[None]
    if not return_info:
        return mask
    return mask, {"n_tracks": len(tracks), "lines": lines, "hole_share": hole.mean(), "profiles": prof}