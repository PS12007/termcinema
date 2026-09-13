"""Quarter-block renderer.

Each terminal cell covers a 2x2 pixel block. We pick the Unicode block
element whose "on" quadrants match a per-pixel brightness threshold,
and color it with a two-tone (fg = avg of lit quadrants, bg = avg of
unlit quadrants) blend. This gives 4x the spatial resolution of plain
one-pixel-per-cell rendering, sitting between half-block and braille in
the resolution/color trade-off (more color fidelity than braille, more
shape detail than half-block).
"""
from __future__ import annotations

import numpy as np

from ..terminal import ColorSupport
from . import base

# Index = TL*8 + TR*4 + BL*2 + BR*1
_QUAD_TABLE = np.array(list(
    " ▗▖▄▝▐▞▟▘▚▌▙▀▜▛█"
))

_BAYER2 = np.array([[0.0, 0.5], [0.75, 0.25]], dtype=np.float32) - 0.5


def render(
    frame_rgb: np.ndarray,
    support: ColorSupport,
    threshold: float = 0.5,
    dither: bool = True,
    color_dither: bool = False,
) -> str:
    h, w, _ = frame_rgb.shape
    h -= h % 2
    w -= w % 2
    frame_rgb = frame_rgb[:h, :w]

    luma = (frame_rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
    if dither:
        tile = np.tile(_BAYER2, (h // 2 + 1, w // 2 + 1))[:h, :w]
        luma = luma + tile * 0.3

    on = luma > threshold
    rows, cols = h // 2, w // 2
    on_b = on.reshape(rows, 2, cols, 2).transpose(0, 2, 1, 3)  # (rows, cols, 2, 2)
    tl, tr, bl, br = on_b[..., 0, 0], on_b[..., 0, 1], on_b[..., 1, 0], on_b[..., 1, 1]
    bits = tl.astype(np.uint8) * 8 + tr.astype(np.uint8) * 4 + bl.astype(np.uint8) * 2 + br.astype(np.uint8)
    glyphs = _QUAD_TABLE[bits]

    color_b = frame_rgb.reshape(rows, 2, cols, 2, 3).transpose(0, 2, 1, 3, 4).astype(np.float32)
    color_b = color_b.reshape(rows, cols, 4, 3)
    on_flat = on_b.reshape(rows, cols, 4)

    on_count = on_flat.sum(axis=2, keepdims=True).astype(np.float32)
    off_count = (4 - on_flat.sum(axis=2)).astype(np.float32)[..., None]
    block_mean = color_b.mean(axis=2)

    on_sum = np.where(on_flat[..., None], color_b, 0).sum(axis=2)
    off_sum = np.where(~on_flat[..., None], color_b, 0).sum(axis=2)

    fg = np.where(on_count > 0, on_sum / np.maximum(on_count, 1), block_mean).astype(np.uint8)
    bg = np.where(off_count > 0, off_sum / np.maximum(off_count, 1), block_mean).astype(np.uint8)

    fg_use = fg if support != ColorSupport.MONO else None
    bg_use = bg if support != ColorSupport.MONO else None
    return base.assemble_frame(glyphs, fg_use, bg_use, support, color_dither=color_dither)
