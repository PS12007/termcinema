"""Color theme transforms and ANSI color-code quantization.

Everything here is vectorized with NumPy so a full HD-ish frame can be
transformed in well under a millisecond, and the 256/16-color paths are
backed by precomputed lookup tables so no per-pixel Python math is needed
in the hot path.
"""
from __future__ import annotations

import numpy as np
from enum import Enum


class Theme(str, Enum):
    NORMAL = "normal"
    GRAYSCALE = "grayscale"
    MONOCHROME = "monochrome"
    SEPIA = "sepia"
    GREEN = "green"       # retro green phosphor terminal
    AMBER = "amber"       # amber CRT
    MATRIX = "matrix"
    DRACULA = "dracula"
    NORD = "nord"
    GRUVBOX = "gruvbox"
    CATPPUCCIN = "catppuccin"
    TOKYO_NIGHT = "tokyonight"


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Duotone themes: (shadow_color, highlight_color) in RGB, luminance-mapped.
_DUOTONES = {
    Theme.SEPIA: (np.array([40, 26, 13], np.float32), np.array([255, 230, 180], np.float32)),
    Theme.GREEN: (np.array([0, 10, 0], np.float32), np.array([70, 255, 90], np.float32)),
    Theme.AMBER: (np.array([20, 8, 0], np.float32), np.array([255, 176, 0], np.float32)),
    Theme.MATRIX: (np.array([0, 5, 0], np.float32), np.array([60, 255, 70], np.float32)),
    Theme.DRACULA: (np.array([40, 42, 54], np.float32), np.array([189, 147, 249], np.float32)),
    Theme.NORD: (np.array([46, 52, 64], np.float32), np.array([136, 192, 208], np.float32)),
    Theme.GRUVBOX: (np.array([40, 40, 40], np.float32), np.array([250, 189, 47], np.float32)),
    Theme.CATPPUCCIN: (np.array([30, 30, 46], np.float32), np.array([245, 194, 231], np.float32)),
    Theme.TOKYO_NIGHT: (np.array([26, 27, 38], np.float32), np.array([122, 162, 247], np.float32)),
}


def apply_theme(frame_rgb: np.ndarray, theme: Theme) -> np.ndarray:
    """Apply a color theme to an (H, W, 3) uint8 RGB frame. Returns uint8."""
    if theme == Theme.NORMAL:
        return frame_rgb

    luma = (frame_rgb.astype(np.float32) @ _LUMA) / 255.0  # (H, W) in [0,1]

    if theme == Theme.GRAYSCALE:
        out = np.repeat(luma[..., None], 3, axis=2) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    if theme == Theme.MONOCHROME:
        # Hard black/white threshold, useful for pure-ASCII "print" look.
        bw = (luma > 0.5).astype(np.float32) * 255.0
        out = np.repeat(bw[..., None], 3, axis=2)
        return out.astype(np.uint8)

    shadow, highlight = _DUOTONES[theme]
    t = luma[..., None]
    out = shadow * (1 - t) + highlight * t
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ANSI-256 palette quantization (precomputed lookup table for speed).
# ---------------------------------------------------------------------------

def _build_256_palette() -> np.ndarray:
    """Return the 256-color xterm palette as an (256, 3) uint8 array."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    # 0-15: standard + bright (approximate xterm defaults)
    base16 = [
        (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
        (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
        (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ]
    palette[:16] = base16
    # 16-231: 6x6x6 color cube
    steps = [0, 95, 135, 175, 215, 255]
    idx = 16
    for r in steps:
        for g in steps:
            for b in steps:
                palette[idx] = (r, g, b)
                idx += 1
    # 232-255: grayscale ramp
    for i in range(24):
        v = 8 + i * 10
        palette[232 + i] = (v, v, v)
    return palette


PALETTE_256 = _build_256_palette()
PALETTE_16 = PALETTE_256[:16]


def _nearest_index_table(palette: np.ndarray) -> np.ndarray:
    """Precompute a (32, 32, 32) LUT mapping quantized RGB -> palette index.

    Distance is the "redmean" weighted approximation of perceptual color
    difference (used by e.g. ImageMagick's color reduction) rather than
    plain Euclidean RGB distance: it weights the R/B channels by how
    much red is in the color, which tracks human color perception much
    more closely than treating R/G/B as equally important, and costs
    nothing extra since the table is only built once at startup.

    We quantize each channel to 5 bits (32 levels) which is plenty of
    precision for terminal color anyway, and keeps the table small
    (32^3 = 32768 entries) and cache-friendly.
    """
    levels = np.arange(32, dtype=np.float32) * (255.0 / 31.0)
    r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)  # (32768, 3)
    pal_f = palette.astype(np.float32)
    best = np.zeros(grid.shape[0], dtype=np.int32)
    chunk = 4096
    for start in range(0, grid.shape[0], chunk):
        block = grid[start:start + chunk]
        dr = block[:, None, 0] - pal_f[None, :, 0]
        dg = block[:, None, 1] - pal_f[None, :, 1]
        db = block[:, None, 2] - pal_f[None, :, 2]
        rmean = (block[:, None, 0] + pal_f[None, :, 0]) / 2.0
        # Redmean perceptual distance approximation.
        d = ((2 + rmean / 256.0) * dr * dr
             + 4.0 * dg * dg
             + (2 + (255.0 - rmean) / 256.0) * db * db)
        best[start:start + chunk] = d.argmin(axis=1)
    return best.reshape(32, 32, 32).astype(np.uint8)


_LUT256 = None
_LUT16 = None


def _get_lut(bits256: bool) -> np.ndarray:
    global _LUT256, _LUT16
    if bits256:
        if _LUT256 is None:
            _LUT256 = _nearest_index_table(PALETTE_256)
        return _LUT256
    if _LUT16 is None:
        _LUT16 = _nearest_index_table(PALETTE_16)
    return _LUT16


def quantize(frame_rgb: np.ndarray, bits256: bool, dither: bool = False) -> np.ndarray:
    """Map an (H, W, 3) uint8 frame to palette indices via the LUT.

    With `dither=True`, applies a 4x4 Bayer ordered dither before
    quantizing so smooth gradients (skies, skin tones) don't band as
    hard when reduced to 256 or 16 colors -- trades a little noise for
    noticeably better perceived color accuracy on a limited palette.
    """
    frame = frame_rgb
    if dither:
        # Half a quantization step in each direction, tiled across the frame.
        step = 255.0 / 31.0
        tile = np.tile(_BAYER4X4, (frame.shape[0] // 4 + 1, frame.shape[1] // 4 + 1))
        tile = tile[:frame.shape[0], :frame.shape[1]]
        noise = (tile[..., None] * step)
        frame = np.clip(frame_rgb.astype(np.float32) + noise, 0, 255)

    lut = _get_lut(bits256)
    q = (frame.astype(np.uint16) * 31 // 255).astype(np.uint8)
    return lut[q[..., 0], q[..., 1], q[..., 2]]


_BAYER4X4 = (np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32) / 16.0) - 0.5
