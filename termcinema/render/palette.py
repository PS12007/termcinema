"""Palette quantization for 256- and 16-color terminals.

Nearest-color search runs once per palette over a 32x32x32 RGB grid in
CIE L*a*b* space (a real perceptual metric, not Euclidean RGB), and
per-frame quantization is a single table gather. Optional ordered
dithering trades a little texture for much less banding.
"""
from __future__ import annotations

import numpy as np

_BASE16 = np.array([
    (12, 12, 12), (197, 15, 31), (19, 161, 14), (193, 156, 0),
    (0, 55, 218), (136, 23, 152), (58, 150, 221), (204, 204, 204),
    (118, 118, 118), (231, 72, 86), (22, 198, 12), (249, 241, 165),
    (59, 120, 255), (180, 0, 158), (97, 214, 214), (242, 242, 242),
], dtype=np.uint8)  # Windows Terminal "Campbell" -- close to most defaults


def xterm256() -> np.ndarray:
    pal = np.zeros((256, 3), np.uint8)
    pal[:16] = _BASE16
    steps = np.array([0, 95, 135, 175, 215, 255], np.uint8)
    r, g, b = np.meshgrid(steps, steps, steps, indexing="ij")
    pal[16:232] = np.stack([r, g, b], -1).reshape(-1, 3)
    gray = 8 + 10 * np.arange(24)
    pal[232:] = gray[:, None]
    return pal


PALETTE_256 = xterm256()
PALETTE_16 = _BASE16


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def build_lut(palette: np.ndarray, restrict: "slice | None" = None) -> np.ndarray:
    levels = (np.arange(32, dtype=np.float32) * 255.0 / 31.0).astype(np.uint8)
    r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
    grid = srgb_to_lab(np.stack([r, g, b], -1).reshape(-1, 3))
    candidates = np.arange(len(palette))
    if restrict is not None:
        candidates = candidates[restrict]
    pal_lab = srgb_to_lab(palette[candidates])
    best = np.empty(grid.shape[0], np.int32)
    for s in range(0, grid.shape[0], 4096):
        d = ((grid[s:s + 4096, None, :] - pal_lab[None]) ** 2).sum(-1)
        best[s:s + 4096] = candidates[d.argmin(1)]
    return best.reshape(32, 32, 32).astype(np.uint8)


_LUTS: dict = {}


def lut(n: int) -> np.ndarray:
    if n not in _LUTS:
        if n == 256:
            # Skip 0-15: their actual colors depend on the user's theme.
            _LUTS[n] = build_lut(PALETTE_256, restrict=slice(16, None))
        else:
            _LUTS[n] = build_lut(PALETTE_16)
    return _LUTS[n]


_BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0) - 0.47


def quantize(rgb: np.ndarray, n: int, dither: bool = False) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) uint8 palette indices."""
    if dither:
        h, w = rgb.shape[:2]
        amp = 40.0 if n == 256 else 90.0
        tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w, None] * amp
        rgb = np.clip(rgb.astype(np.float32) + tile, 0, 255).astype(np.uint8)
    q = rgb >> 3
    return lut(n)[q[..., 0], q[..., 1], q[..., 2]]
