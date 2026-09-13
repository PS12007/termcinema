"""numba-compiled per-cell fitting kernels (optional accelerator).

The NumPy implementations in ``text.py`` are exact and portable but
allocate big intermediate arrays; these loops do the same math per cell
with no allocations and run 10-40x faster. ``HAVE_NUMBA`` is False when
numba isn't installed and callers fall back to NumPy.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

WR, WG, WB = 0.55, 1.0, 0.35


@njit(cache=True, fastmath=True, nogil=True)
def fit_masks(px, gh, gw, masks, cps, out_cps, out_fg, out_bg):  # pragma: no cover - compiled
    """Exhaustive best two-color fit over `masks` (K, n) of 0/1."""
    rows = px.shape[0] // gh
    cols = px.shape[1] // gw
    n = gh * gw
    K = masks.shape[0]
    pr = np.empty(n, np.float32)
    pg = np.empty(n, np.float32)
    pb = np.empty(n, np.float32)
    for cy in range(rows):
        for cx in range(cols):
            tr = 0.0
            tg = 0.0
            tb = 0.0
            for i in range(n):
                yy = cy * gh + i // gw
                xx = cx * gw + i % gw
                pr[i] = px[yy, xx, 0]
                pg[i] = px[yy, xx, 1]
                pb[i] = px[yy, xx, 2]
                tr += pr[i]
                tg += pg[i]
                tb += pb[i]
            best = 0
            best_score = -1.0
            for k in range(K):
                sr = 0.0
                sg = 0.0
                sb = 0.0
                non = 0
                for i in range(n):
                    if masks[k, i]:
                        sr += pr[i]
                        sg += pg[i]
                        sb += pb[i]
                        non += 1
                noff = n - non
                a = (sr * WR) ** 2 + (sg * WG) ** 2 + (sb * WB) ** 2
                orr = (tr - sr) * WR
                og = (tg - sg) * WG
                ob = (tb - sb) * WB
                score = 0.0
                if non > 0:
                    score += a / non
                if noff > 0:
                    score += (orr * orr + og * og + ob * ob) / noff
                if score > best_score + 1e-3:
                    best_score = score
                    best = k
            sr = 0.0
            sg = 0.0
            sb = 0.0
            non = 0
            for i in range(n):
                if masks[best, i]:
                    sr += pr[i]
                    sg += pg[i]
                    sb += pb[i]
                    non += 1
            noff = n - non
            if non > 0:
                fr, fg_, fb = sr / non, sg / non, sb / non
            else:
                fr, fg_, fb = tr / n, tg / n, tb / n
            if noff > 0:
                br, bg_, bb = (tr - sr) / noff, (tg - sg) / noff, (tb - sb) / noff
            else:
                br, bg_, bb = fr, fg_, fb
            cp = cps[best]
            if abs(fr - br) < 1.0 and abs(fg_ - bg_) < 1.0 and abs(fb - bb) < 1.0:
                cp = 32
                fr, fg_, fb = br, bg_, bb
            out_cps[cy, cx] = cp
            out_fg[cy, cx, 0] = min(max(fr + 0.5, 0.0), 255.0)
            out_fg[cy, cx, 1] = min(max(fg_ + 0.5, 0.0), 255.0)
            out_fg[cy, cx, 2] = min(max(fb + 0.5, 0.0), 255.0)
            out_bg[cy, cx, 0] = min(max(br + 0.5, 0.0), 255.0)
            out_bg[cy, cx, 1] = min(max(bg_ + 0.5, 0.0), 255.0)
            out_bg[cy, cx, 2] = min(max(bb + 0.5, 0.0), 255.0)


@njit(cache=True, fastmath=True, nogil=True)
def two_means(px, lut, braille, flat_thresh, out_cps, out_fg, out_bg):  # pragma: no cover - compiled
    """2x4 cells: split pixels into two clusters, pick the exact glyph."""
    rows = px.shape[0] // 4
    cols = px.shape[1] // 2
    pr = np.empty(8, np.float32)
    pg = np.empty(8, np.float32)
    pb = np.empty(8, np.float32)
    on = np.zeros(8, np.bool_)
    for cy in range(rows):
        for cx in range(cols):
            mr = 0.0
            mg = 0.0
            mb = 0.0
            for i in range(8):
                yy = cy * 4 + i // 2
                xx = cx * 2 + i % 2
                pr[i] = px[yy, xx, 0]
                pg[i] = px[yy, xx, 1]
                pb[i] = px[yy, xx, 2]
                mr += pr[i]
                mg += pg[i]
                mb += pb[i]
            mr /= 8.0
            mg /= 8.0
            mb /= 8.0
            # farthest pixel from the mean defines the split axis
            far = 0
            fd = -1.0
            for i in range(8):
                d = ((pr[i] - mr) * WR) ** 2 + ((pg[i] - mg) * WG) ** 2 + ((pb[i] - mb) * WB) ** 2
                if d > fd:
                    fd = d
                    far = i
            ar = (pr[far] - mr) * WR
            ag = (pg[far] - mg) * WG
            ab = (pb[far] - mb) * WB
            for i in range(8):
                on[i] = ((pr[i] - mr) * WR * ar + (pg[i] - mg) * WG * ag + (pb[i] - mb) * WB * ab) > 0
            c1r = c1g = c1b = c0r = c0g = c0b = 0.0
            n1 = 0
            for it in range(3):
                c1r = c1g = c1b = c0r = c0g = c0b = 0.0
                n1 = 0
                for i in range(8):
                    if on[i]:
                        c1r += pr[i]
                        c1g += pg[i]
                        c1b += pb[i]
                        n1 += 1
                    else:
                        c0r += pr[i]
                        c0g += pg[i]
                        c0b += pb[i]
                n0 = 8 - n1
                if n1 > 0:
                    c1r /= n1
                    c1g /= n1
                    c1b /= n1
                if n0 > 0:
                    c0r /= n0
                    c0g /= n0
                    c0b /= n0
                if it == 2 or n1 == 0 or n0 == 0:
                    break
                for i in range(8):
                    d1 = ((pr[i] - c1r) * WR) ** 2 + ((pg[i] - c1g) * WG) ** 2 + ((pb[i] - c1b) * WB) ** 2
                    d0 = ((pr[i] - c0r) * WR) ** 2 + ((pg[i] - c0g) * WG) ** 2 + ((pb[i] - c0b) * WB) ** 2
                    on[i] = d1 < d0
            n0 = 8 - n1
            if n1 == 0:
                c1r, c1g, c1b = c0r, c0g, c0b
            if n0 == 0:
                c0r, c0g, c0b = c1r, c1g, c1b
            # orient: "on" is the brighter cluster
            if (c1r * WR + c1g * WG + c1b * WB) < (c0r * WR + c0g * WG + c0b * WB):
                for i in range(8):
                    on[i] = not on[i]
                c1r, c0r = c0r, c1r
                c1g, c0g = c0g, c1g
                c1b, c0b = c0b, c1b
                n1 = 8 - n1
            bits = 0
            for i in range(8):
                if on[i]:
                    bits |= 1 << i
            contrast = max(abs(c1r - c0r), max(abs(c1g - c0g), abs(c1b - c0b)))
            if braille:
                if contrast < flat_thresh:
                    bits = 0
                    c1r, c1g, c1b = mr, mg, mb
                    c0r, c0g, c0b = mr, mg, mb
                else:
                    c1r = c1r * 1.12 + 6
                    c1g = c1g * 1.12 + 6
                    c1b = c1b * 1.12 + 6
                    c0r *= 0.85
                    c0g *= 0.85
                    c0b *= 0.85
            else:
                if n1 > 4:
                    bits = 255 - bits
                    c1r, c0r = c0r, c1r
                    c1g, c0g = c0g, c1g
                    c1b, c0b = c0b, c1b
                if contrast < flat_thresh:
                    bits = 0
                    c1r, c1g, c1b = c0r, c0g, c0b
            out_cps[cy, cx] = lut[bits]
            out_fg[cy, cx, 0] = min(max(c1r + 0.5, 0.0), 255.0)
            out_fg[cy, cx, 1] = min(max(c1g + 0.5, 0.0), 255.0)
            out_fg[cy, cx, 2] = min(max(c1b + 0.5, 0.0), 255.0)
            out_bg[cy, cx, 0] = min(max(c0r + 0.5, 0.0), 255.0)
            out_bg[cy, cx, 1] = min(max(c0g + 0.5, 0.0), 255.0)
            out_bg[cy, cx, 2] = min(max(c0b + 0.5, 0.0), 255.0)


def warmup() -> None:
    """Compile kernels ahead of first use (runs in a background thread)."""
    if not HAVE_NUMBA:
        return
    px = np.zeros((8, 4, 3), np.uint8)
    cps = np.zeros((2, 2), np.int32)
    fg = np.zeros((2, 2, 3), np.uint8)
    bg = np.zeros((2, 2, 3), np.uint8)
    masks = np.zeros((2, 8), np.uint8)
    fit_masks(px, 4, 2, masks, np.array([32, 32], np.int32), cps, fg, bg)
    two_means(px, np.full(256, 32, np.int32), False, 3.0, cps, fg, bg)
