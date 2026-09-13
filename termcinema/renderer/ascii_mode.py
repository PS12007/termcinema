"""Luminance-ramp renderer: classic ASCII art, extended block shading,
and dense Unicode ramps, all sharing one implementation.

Optionally does edge-aware glyph selection: pixels that sit on a strong
gradient get a directional character (|, /, -, \\) instead of a plain
density character, which noticeably sharpens outlines like ChatGPT's
"intelligent character selection" idea, at a modest extra cost (one
Sobel pass, fully vectorized).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..terminal import ColorSupport
from .. import scaler
from . import base

RAMPS = {
    "classic": " .:-=+*#%@",
    "extended": " ░▒▓█",
    "unicode": " ⠁⠃⠇⠏⠟⠿⣟⣿",
    "minimal": " .oO@",
}

_EDGE_CHARS = {
    # angle bucket (0=|, 1=/, 2=-, 3=\)
    0: "|",
    1: "/",
    2: "-",
    3: "\\",
}


def _luma(frame_rgb: np.ndarray) -> np.ndarray:
    return (frame_rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32))


def render(
    frame_rgb: np.ndarray,
    support: ColorSupport,
    ramp: str = "classic",
    colorize: bool = True,
    edge_aware: bool = True,
    edge_threshold: float = 60.0,
    color_dither: bool = False,
    use_gpu: bool = True,
) -> str:
    charset = RAMPS.get(ramp, RAMPS["classic"])
    luma = _luma(frame_rgb)  # (H, W) 0..255
    n = len(charset)
    idx = np.clip((luma / 255.0 * (n - 1)).round().astype(np.int32), 0, n - 1)
    char_arr = np.array(list(charset))
    glyphs = char_arr[idx]

    if edge_aware:
        gray = luma.astype(np.uint8)
        gpu_result = scaler.sobel_gpu(gray) if use_gpu else None
        if gpu_result is not None:
            gx, gy = gpu_result
        else:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        strong = mag > edge_threshold
        if np.any(strong):
            ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0  # 0..180
            # bucket into 4 directions: |, /, -, \
            bucket = np.digitize(ang, [22.5, 67.5, 112.5, 157.5]) % 4
            edge_glyphs = np.array([_EDGE_CHARS[i] for i in range(4)])[bucket]
            glyphs = np.where(strong, edge_glyphs, glyphs)

    fg = frame_rgb if colorize and support != ColorSupport.MONO else None
    return base.assemble_frame(glyphs, fg, None, support, color_dither=color_dither)
