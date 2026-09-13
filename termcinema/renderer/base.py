"""Shared machinery for turning a grid of (glyph, fg, bg) into an ANSI
terminal frame as fast as possible.

Three optimizations stack here, roughly in order of impact:

1. Per-row run-length encoding: consecutive cells with the same color
   only pay for one escape code, no matter how wide the run is. Flat
   regions (skies, walls, letterboxing) are nearly free.
2. Run boundaries are found with a single vectorized `np.diff` per row
   instead of `itertools.groupby` walking Python tuples one at a time.
3. 256-color and 16-color escape fragments are precomputed once (only
   256/16 possible values exist) and looked up, never re-formatted.
   TrueColor has too many possible values to precompute, so those are
   memoized in a capped LRU-ish cache instead -- real footage reuses
   a much smaller set of actual colors than the 16M possible ones.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..terminal import ColorSupport
from .. import colorspace

RESET = "\x1b[0m"

# 16-color SGR bases.
_FG16_BASE = [30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97]
_BG16_BASE = [40, 41, 42, 43, 44, 45, 46, 47, 100, 101, 102, 103, 104, 105, 106, 107]

# Precomputed once: every possible fg/bg escape fragment for 256- and
# 16-color modes. Lookup is O(1) instead of formatting a new string
# for every run of every frame.
_FG256_CODES = tuple(f"38;5;{i}" for i in range(256))
_BG256_CODES = tuple(f"48;5;{i}" for i in range(256))
_FG16_CODES = tuple(str(c) for c in _FG16_BASE)
_BG16_CODES = tuple(str(c) for c in _BG16_BASE)

# TrueColor has 16.7M possible values -- can't precompute, so we memoize
# what actually gets used. Real video reuses a much smaller working set
# of colors per scene than the full RGB space, so this pays off. Capped
# so a long-running player doesn't leak memory across a whole movie.
_TC_FG_CACHE: dict[int, str] = {}
_TC_BG_CACHE: dict[int, str] = {}
_TC_CACHE_CAP = 200_000


def _tc_fg(key: int) -> str:
    c = _TC_FG_CACHE.get(key)
    if c is None:
        if len(_TC_FG_CACHE) >= _TC_CACHE_CAP:
            _TC_FG_CACHE.clear()
        c = f"38;2;{(key >> 16) & 255};{(key >> 8) & 255};{key & 255}"
        _TC_FG_CACHE[key] = c
    return c


def _tc_bg(key: int) -> str:
    c = _TC_BG_CACHE.get(key)
    if c is None:
        if len(_TC_BG_CACHE) >= _TC_CACHE_CAP:
            _TC_BG_CACHE.clear()
        c = f"48;2;{(key >> 16) & 255};{(key >> 8) & 255};{key & 255}"
        _TC_BG_CACHE[key] = c
    return c


def _color_keys(rgb: np.ndarray, support: ColorSupport, color_dither: bool) -> np.ndarray:
    """Return an (H, W) int32 array of color-key ids for one plane."""
    if support == ColorSupport.TRUECOLOR:
        return ((rgb[..., 0].astype(np.int32) << 16)
                 | (rgb[..., 1].astype(np.int32) << 8)
                 | rgb[..., 2].astype(np.int32))
    if support == ColorSupport.ANSI256:
        return colorspace.quantize(rgb, bits256=True, dither=color_dither).astype(np.int32)
    if support == ColorSupport.ANSI16:
        return colorspace.quantize(rgb, bits256=False, dither=color_dither).astype(np.int32)
    return np.zeros(rgb.shape[:2], dtype=np.int32)  # MONO


def _fg_code(key: int, support: ColorSupport) -> str:
    if support == ColorSupport.TRUECOLOR:
        return _tc_fg(key)
    if support == ColorSupport.ANSI256:
        return _FG256_CODES[key]
    return _FG16_CODES[key]


def _bg_code(key: int, support: ColorSupport) -> str:
    if support == ColorSupport.TRUECOLOR:
        return _tc_bg(key)
    if support == ColorSupport.ANSI256:
        return _BG256_CODES[key]
    return _BG16_CODES[key]


def _row_runs(keys: np.ndarray) -> np.ndarray:
    """Given a 1D array of per-cell keys, return the start index of each
    run of equal consecutive values (always includes index 0)."""
    if keys.shape[0] <= 1:
        return np.array([0])
    changed = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    if changed.size == 0:
        return np.array([0])
    return np.concatenate(([0], changed))


def assemble_frame(
    glyphs: np.ndarray,
    fg_rgb: Optional[np.ndarray],
    bg_rgb: Optional[np.ndarray],
    support: ColorSupport,
    color_dither: bool = False,
) -> str:
    """glyphs: (H, W) array of single-character strings.
    fg_rgb / bg_rgb: (H, W, 3) uint8 or None.
    """
    h, w = glyphs.shape

    if support == ColorSupport.MONO or (fg_rgb is None and bg_rgb is None):
        return "\n".join("".join(row.tolist()) for row in glyphs)

    fg_keys = _color_keys(fg_rgb, support, color_dither) if fg_rgb is not None else None
    bg_keys = _color_keys(bg_rgb, support, color_dither) if bg_rgb is not None else None

    # Combine fg/bg into one key stream per row so a run only continues
    # while *both* stay constant.
    if fg_keys is not None and bg_keys is not None:
        combined = (fg_keys.astype(np.int64) << 32) | (bg_keys.astype(np.int64) & 0xFFFFFFFF)
    elif fg_keys is not None:
        combined = fg_keys.astype(np.int64)
    else:
        combined = bg_keys.astype(np.int64)

    lines = []
    for y in range(h):
        row_glyphs = glyphs[y]
        row_combined = combined[y]
        starts = _row_runs(row_combined)
        ends = np.append(starts[1:], w)

        parts = []
        row_fg = fg_keys[y] if fg_keys is not None else None
        row_bg = bg_keys[y] if bg_keys is not None else None
        for s, e in zip(starts.tolist(), ends.tolist()):
            codes = []
            if row_fg is not None:
                codes.append(_fg_code(int(row_fg[s]), support))
            if row_bg is not None:
                codes.append(_bg_code(int(row_bg[s]), support))
            chars = "".join(row_glyphs[s:e].tolist())
            parts.append(f"\x1b[{';'.join(codes)}m{chars}" if codes else chars)
        lines.append("".join(parts) + RESET)

    return "\n".join(lines)
