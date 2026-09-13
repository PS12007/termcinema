"""Text-cell renderers.

Every renderer receives an RGB image whose size is an exact multiple of
its sub-pixel grid (``gw`` x ``gh`` pixels per terminal cell) and returns a
:class:`CellFrame`.

The block renderers treat each cell as a tiny two-color image: choose a
glyph shape (which sub-pixels are "foreground") and two colors so the cell
reproduces the source pixels with the least squared error. For small
glyph sets this is solved exactly by scoring every candidate shape at
once; for the 256-pattern octant/braille sets a per-cell 2-means split
finds the partition directly. This is the same idea behind chafa's
symbol mode, and it's the difference between blurry mush and crisp edges.
"""
from __future__ import annotations

import numpy as np

from . import glyphs, kernels
from .encode import CellFrame

# Perceptual channel weights used when *choosing* shapes (colors
# themselves are always computed from unweighted pixels).
_W = np.array([0.55, 1.0, 0.35], np.float32)


def _cells(px: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """(rows*gh, cols*gw, 3) -> (rows, cols, gh*gw, 3) float32."""
    h, w, _ = px.shape
    rows, cols = h // gh, w // gw
    v = px[: rows * gh, : cols * gw].reshape(rows, gh, cols, gw, 3).transpose(0, 2, 1, 3, 4)
    return v.reshape(rows, cols, gh * gw, 3).astype(np.float32)


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


class TextRenderer:
    kind = "cells"
    name = "base"
    gw = 1
    gh = 1
    description = ""
    needs_unicode = True

    def render(self, px: np.ndarray) -> CellFrame:  # pragma: no cover - interface
        raise NotImplementedError


class HalfBlock(TextRenderer):
    name = "halfblock"
    gw, gh = 1, 2
    description = "▀ half blocks: 1x2 px per cell, universally supported"

    def render(self, px):
        rows, cols = px.shape[0] // 2, px.shape[1]
        top = px[0:rows * 2:2]
        bot = px[1:rows * 2:2]
        cps = np.full((rows, cols), 0x2580, np.int32)
        same = (top == bot).all(-1)
        cps[same] = 0x20
        return CellFrame(cps, np.ascontiguousarray(top), np.ascontiguousarray(bot))


class _ExhaustiveBlocks(TextRenderer):
    """Best two-color fit over a small fixed glyph set, solved exactly."""

    masks: tuple = ()

    def __init__(self):
        n = self.gw * self.gh
        self._mask_ids = np.array([m for m, _ in self.masks], np.int64)
        self._cps = np.array([c for _, c in self.masks], np.int32)
        self._M = glyphs.mask_matrix(self._mask_ids, n)       # (K, n)
        self._n_on = self._M.sum(1)
        self._n_off = n - self._n_on

    def render(self, px):
        if kernels.HAVE_NUMBA:
            rows, cols = px.shape[0] // self.gh, px.shape[1] // self.gw
            cps = np.empty((rows, cols), np.int32)
            fg = np.empty((rows, cols, 3), np.uint8)
            bg = np.empty((rows, cols, 3), np.uint8)
            kernels.fit_masks(np.ascontiguousarray(px), self.gh, self.gw, self._M.astype(np.uint8),
                              self._cps, cps, fg, bg)
            return CellFrame(cps, fg, bg)
        return self.render_numpy(px)

    def render_numpy(self, px):
        c = _cells(px, self.gh, self.gw)                      # (R, C, n, 3)
        rows, cols, n, _ = c.shape
        P = c.reshape(-1, n, 3)
        Pw = P * _W
        S = Pw.sum(1)                                         # (N, 3)
        S_on = (Pw.transpose(0, 2, 1) @ self._M.T).transpose(0, 2, 1)  # (N, K, 3)
        S_off = S[:, None, :] - S_on
        score = (S_on ** 2).sum(-1) / np.maximum(self._n_on, 1) + (S_off ** 2).sum(-1) / np.maximum(self._n_off, 1)
        # Flat cells: every mask scores the same; argmax picks index 0 (space).
        best = score.argmax(1)
        ar = np.arange(best.size)
        on = S_on[ar, best] / _W
        off = S_off[ar, best] / _W
        n_on = self._n_on[best][:, None]
        n_off = self._n_off[best][:, None]
        mean = (P.sum(1))[:, :] / n
        fg = np.where(n_on > 0, on / np.maximum(n_on, 1), mean)
        bg = np.where(n_off > 0, off / np.maximum(n_off, 1), mean)
        cps = self._cps[best]
        # Zero-contrast cells collapse to a plain background cell.
        flat = np.abs(fg - bg).max(1) < 1.0
        cps = np.where(flat, 0x20, cps)
        fg = np.where(flat[:, None], bg, fg)
        return CellFrame(cps.reshape(rows, cols), _u8(fg).reshape(rows, cols, 3), _u8(bg).reshape(rows, cols, 3))


class Blocks(_ExhaustiveBlocks):
    name = "blocks"
    gw, gh = 2, 4
    description = "Best-fit block elements (quadrants, halves, quarters): sharp and very compatible"
    masks = glyphs.BLOCK_SET


class Quadrant(_ExhaustiveBlocks):
    name = "quadrant"
    gw, gh = 2, 2
    description = "2x2 quadrant blocks with two-color fitting"
    masks = tuple((m, glyphs.OCTANT_CODEPOINTS[_q]) for m, _q in (
        (0b0000, 0), (0b0011, 0b00001111), (0b0101, 0b01010101), (0b0001, 0b00000101),
        (0b0010, 0b00001010), (0b0100, 0b01010000), (0b1000, 0b10100000), (0b1001, 0b10100101),
    ))


class Sextant(_ExhaustiveBlocks):
    name = "sextant"
    gw, gh = 2, 3
    description = "2x3 sextant mosaic (Unicode 13): more detail, needs a recent font/terminal"
    # One representative per complement pair: masks without the top-left bit.
    masks = tuple((m, glyphs.SEXTANT_CODEPOINTS[m]) for m in range(64) if not (m & 1))


def _two_means(P: np.ndarray, iters: int = 2):
    """Split each cell's pixels into two clusters. P: (N, n, 3).
    Returns boolean (N, n) membership of the brighter cluster."""
    Pw = P * _W
    mean = Pw.mean(1, keepdims=True)
    d = Pw - mean
    dist = (d ** 2).sum(-1)
    far = dist.argmax(1)
    axis = d[np.arange(P.shape[0]), far]                     # (N, 3)
    on = (d * axis[:, None, :]).sum(-1) > 0
    for _ in range(iters):
        onf = on[..., None].astype(np.float32)
        n_on = onf.sum(1)
        n_off = P.shape[1] - n_on
        c1 = (Pw * onf).sum(1) / np.maximum(n_on, 1)
        c0 = (Pw * (1 - onf)).sum(1) / np.maximum(n_off, 1)
        d1 = ((Pw - c1[:, None]) ** 2).sum(-1)
        d0 = ((Pw - c0[:, None]) ** 2).sum(-1)
        on = d1 < d0
    # Orient: "on" = brighter cluster.
    luma = Pw.sum(-1)
    onf = on.astype(np.float32)
    n_on = onf.sum(1)
    b1 = (luma * onf).sum(1) / np.maximum(n_on, 1)
    b0 = (luma * (1 - onf)).sum(1) / np.maximum(P.shape[1] - n_on, 1)
    flip = (b1 < b0) & (n_on > 0) & (n_on < P.shape[1])
    on[flip] = ~on[flip]
    return on


def _cluster_colors(P, on):
    onf = on[..., None].astype(np.float32)
    n = P.shape[1]
    n_on = onf.sum(1)
    n_off = n - n_on
    mean = P.mean(1)
    fg = np.where(n_on > 0, (P * onf).sum(1) / np.maximum(n_on, 1), mean)
    bg = np.where(n_off > 0, (P * (1 - onf)).sum(1) / np.maximum(n_off, 1), mean)
    return fg, bg


def _two_means_numba(px, lut, braille, flat):
    rows, cols = px.shape[0] // 4, px.shape[1] // 2
    cps = np.empty((rows, cols), np.int32)
    fg = np.empty((rows, cols, 3), np.uint8)
    bg = np.empty((rows, cols, 3), np.uint8)
    kernels.two_means(np.ascontiguousarray(px), lut, braille, flat, cps, fg, bg)
    return CellFrame(cps, fg, bg)


class Octant(TextRenderer):
    name = "octant"
    gw, gh = 2, 4
    description = "2x4 octant mosaic (Unicode 16): maximum block detail, newest terminals only"

    def render(self, px):
        if kernels.HAVE_NUMBA:
            return _two_means_numba(px, glyphs.OCTANT_LUT, False, 3.0)
        return self.render_numpy(px)

    def render_numpy(self, px):
        c = _cells(px, 4, 2)
        rows, cols, n, _ = c.shape
        P = c.reshape(-1, n, 3)
        on = _two_means(P)
        bits = (on.astype(np.int32) << np.arange(8, dtype=np.int32)).sum(1)
        fg, bg = _cluster_colors(P, on)
        # Prefer the glyph with fewer lit sub-pixels (swap colors for complements).
        pop = on.sum(1)
        inv = pop > 4
        bits = np.where(inv, 255 - bits, bits)
        fg, bg = np.where(inv[:, None], bg, fg), np.where(inv[:, None], fg, bg)
        flat = np.abs(fg - bg).max(1) < 3.0
        bits = np.where(flat, 0, bits)
        fg = np.where(flat[:, None], bg, fg)
        cps = glyphs.OCTANT_LUT[bits]
        return CellFrame(cps.reshape(rows, cols), _u8(fg).reshape(rows, cols, 3), _u8(bg).reshape(rows, cols, 3))


class Braille(TextRenderer):
    name = "braille"
    gw, gh = 2, 4
    description = "Braille dot matrix: 2x4 dots per cell, a fine halftone look"

    def render(self, px):
        if kernels.HAVE_NUMBA:
            return _two_means_numba(px, glyphs.BRAILLE_LUT, True, 14.0)
        return self.render_numpy(px)

    def render_numpy(self, px):
        c = _cells(px, 4, 2)
        rows, cols, n, _ = c.shape
        P = c.reshape(-1, n, 3)
        on = _two_means(P)
        fg, bg = _cluster_colors(P, on)
        # Dots are thin, so push the dot color a little brighter and the
        # background a little darker to keep perceived contrast.
        contrast = np.abs(fg - bg).max(1)
        flat = contrast < 14.0
        mean = P.mean(1)
        fg = np.where(flat[:, None], mean, fg * 1.12 + 6)
        bg = np.where(flat[:, None], mean, bg * 0.85)
        bits = (on.astype(np.int32) << np.arange(8, dtype=np.int32)).sum(1)
        bits = np.where(flat, 0, bits)
        cps = glyphs.BRAILLE_LUT[bits]
        return CellFrame(cps.reshape(rows, cols), _u8(fg).reshape(rows, cols, 3), _u8(bg).reshape(rows, cols, 3))


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------

_ASCII_CHARS = "".join(chr(c) for c in range(32, 127) if chr(c) not in "`")


def _char_coverage(chars: str, gh: int, gw: int) -> np.ndarray:
    """Rasterize printable ASCII with OpenCV's Hershey font and measure how
    much of each sub-cell every glyph covers. (K, gh*gw) in [0, 1]."""
    import cv2

    cw, ch = 12 * gw, 12 * gh
    out = np.zeros((len(chars), gh * gw), np.float32)
    for i, s in enumerate(chars):
        img = np.zeros((ch, cw), np.uint8)
        scale = ch / 26.0
        (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = (cw - tw) // 2
        y = int(ch * 0.78)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 2, cv2.LINE_AA)
        cov = cv2.resize(img.astype(np.float32) / 255.0, (gw, gh), interpolation=cv2.INTER_AREA)
        out[i] = cov.reshape(-1)
    return out


class Ascii(TextRenderer):
    name = "ascii"
    gw, gh = 2, 4
    description = "Shape-matched colored ASCII: picks the character whose shape fits each cell"
    needs_unicode = False
    LEVELS = 4

    def __init__(self):
        self._lut = None
        self._cov = None

    SHAPES = "|/\\-_=^v<>()[]{}LJT7Y'`,.;:!+xXoO\"~"
    RAMP = " .:-=+*#%@"

    def _build(self):
        n = self.gh * self.gw
        L = self.LEVELS
        idx = np.arange(L ** n)
        pats = ((idx[:, None] // (L ** np.arange(n))[None, :]) % L).astype(np.float32) / (L - 1)

        # Structured cells: match the *shape* of the light/dark pattern
        # (mean-centered cosine similarity), brightness is carried by color.
        cov = _char_coverage(self.SHAPES, self.gh, self.gw)
        cc = cov - cov.mean(1, keepdims=True)
        cc /= np.linalg.norm(cc, axis=1, keepdims=True) + 1e-6
        shape_pick = np.empty(L ** n, np.int32)
        for s in range(0, pats.shape[0], 8192):
            p = pats[s:s + 8192]
            pc = p - p.mean(1, keepdims=True)
            shape_pick[s:s + 8192] = (pc @ cc.T).argmax(1)
        shape_cps = np.array([ord(ch) for ch in self.SHAPES], np.int32)[shape_pick]

        # Flat cells: classic density ramp by mean brightness.
        ramp_idx = np.clip((pats.mean(1) * (len(self.RAMP) - 1)).round().astype(np.int64), 0, len(self.RAMP) - 1)
        ramp_cps = np.array([ord(ch) for ch in self.RAMP], np.int32)[ramp_idx]

        contrast = pats.max(1) - pats.min(1)
        self._lut = np.where(contrast >= 0.66, shape_cps, ramp_cps).astype(np.int32)
        all_chars = self.SHAPES + self.RAMP
        allcov = _char_coverage(all_chars, self.gh, self.gw).mean(1)
        allcov /= max(allcov.max(), 1e-6)
        self._density = np.full(128, 1.0, np.float32)
        for ch, d in zip(all_chars, allcov):
            self._density[ord(ch)] = d

    def render(self, px):
        if self._lut is None:
            self._build()
        c = _cells(px, self.gh, self.gw)
        rows, cols, n, _ = c.shape
        P = c.reshape(-1, n, 3)
        luma = (0.5 * (P @ np.array([0.2126, 0.7152, 0.0722], np.float32)) + 0.5 * P.max(-1)) / 255.0
        lvl = np.clip(luma * (self.LEVELS - 1) + 0.5, 0, self.LEVELS - 1).astype(np.int64)
        code = (lvl * (self.LEVELS ** np.arange(n))[None, :]).sum(1)
        cps = self._lut[code]
        # Cells too dark to show any glyph still get a faint dot so shapes
        # read against black.
        mean = P.mean(1)
        mean_l = luma.mean(1)
        cps = np.where((cps == 0x20) & (mean_l > 0.06), ord("."), cps)
        dens = self._density[np.minimum(cps, 127)][:, None]
        fg = mean / np.clip(dens * 1.6, 0.4, 1.0)
        bg = np.zeros_like(mean)
        return CellFrame(cps.reshape(rows, cols).astype(np.int32), _u8(fg).reshape(rows, cols, 3),
                         _u8(bg).reshape(rows, cols, 3))


class AsciiRamp(TextRenderer):
    name = "ascii-ramp"
    gw, gh = 1, 2
    description = "Classic luminance-ramp ASCII art in color"
    needs_unicode = False
    RAMP = " .'`^,:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

    def __init__(self):
        self._cps = np.array([ord(ch) for ch in self.RAMP], np.int32)

    def render(self, px):
        c = _cells(px, 2, 1)
        rows, cols, n, _ = c.shape
        mean = c.mean(2).reshape(-1, 3)
        luma = (mean @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
        idx = np.clip((luma * (len(self.RAMP) - 1)).round().astype(np.int64), 0, len(self.RAMP) - 1)
        cps = self._cps[idx]
        fg = mean / np.clip(luma[:, None] * 1.4, 0.45, 1.0)
        bg = np.zeros_like(mean)
        return CellFrame(cps.reshape(rows, cols), _u8(fg).reshape(rows, cols, 3), _u8(bg).reshape(rows, cols, 3))
