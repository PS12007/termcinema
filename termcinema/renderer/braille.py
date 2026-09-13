"""Braille renderer.

Each Unicode Braille character (U+2800 block) encodes an independent
2 columns x 4 rows dot matrix, giving 8x the spatial resolution of a
plain one-pixel-per-cell renderer. We threshold luminance per pixel
(with an optional 4x4 Bayer ordered dither to avoid harsh banding) to
decide which dots are "on", and color each cell using the average RGB
of the block so the shape stays sharp while still carrying color.
"""
from __future__ import annotations

import numpy as np

from ..terminal import ColorSupport
from . import base

_BRAILLE_BASE = 0x2800

# Bit weight for each (row, col) position inside a 4-row x 2-col cell.
_BIT_WEIGHTS = np.array([
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
], dtype=np.uint16)

_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32) / 16.0 - 0.5  # centered around 0


def render(
    frame_rgb: np.ndarray,
    support: ColorSupport,
    threshold: float = 0.5,
    dither: bool = True,
    color_dither: bool = False,
) -> str:
    h, w, _ = frame_rgb.shape
    h -= h % 4
    w -= w % 2
    frame_rgb = frame_rgb[:h, :w]

    luma = (frame_rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0

    if dither:
        tile = np.tile(_BAYER4, (h // 4 + 1, w // 2 + 1))[:h, :w]
        luma = luma + tile * 0.35

    on = luma > threshold  # (H, W) bool

    rows, cols = h // 4, w // 2
    on_blocks = on.reshape(rows, 4, cols, 2).transpose(0, 2, 1, 3)  # (rows, cols, 4, 2)
    weights = _BIT_WEIGHTS  # (4, 2)
    codes = (on_blocks.astype(np.uint16) * weights).sum(axis=(2, 3)) + _BRAILLE_BASE
    glyphs = np.vectorize(chr)(codes)

    color_blocks = frame_rgb.reshape(rows, 4, cols, 2, 3).transpose(0, 2, 1, 3, 4)
    avg_color = color_blocks.reshape(rows, cols, 8, 3).mean(axis=2).astype(np.uint8)

    fg = avg_color if support != ColorSupport.MONO else None
    return base.assemble_frame(glyphs, fg, None, support, color_dither=color_dither)
