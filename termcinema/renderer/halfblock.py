"""Half-block renderer.

Each terminal cell shows the upper-half-block glyph '▀' with the
foreground color set to the top source pixel and the background color
set to the bottom source pixel. Because a cell can display two
independently-colored pixels, this doubles effective vertical
resolution versus one-pixel-per-char rendering, and at TrueColor depth
looks close to a real (blocky) image. This is the recommended default
mode.
"""
from __future__ import annotations

import numpy as np

from ..terminal import ColorSupport
from . import base

GLYPH = "\u2580"  # ▀ upper half block


def render(frame_rgb: np.ndarray, support: ColorSupport, color_dither: bool = False, **_ignored) -> str:
    h, w, _ = frame_rgb.shape
    if h % 2:
        frame_rgb = frame_rgb[: h - 1]
        h -= 1
    top = frame_rgb[0::2]
    bottom = frame_rgb[1::2]
    glyphs = np.full((h // 2, w), GLYPH)
    return base.assemble_frame(glyphs, top, bottom, support, color_dither=color_dither)
