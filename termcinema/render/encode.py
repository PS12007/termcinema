"""Vectorized cell-grid -> ANSI byte encoder with on-screen diffing.

A frame is a grid of cells: a Unicode codepoint plus foreground and
background colors. Turning ~20k cells into escape codes with a Python
loop costs tens of milliseconds, so this module never loops over cells.
Instead every cell reserves a fixed-width slot of bytes

    [cursor move][SGR: fg ; bg][UTF-8 glyph]

filled in with NumPy arithmetic, along with a boolean "keep" mask that
drops the parts a cell doesn't need (leading zeros, repeated colors,
cursor moves between adjacent cells). One boolean gather then yields the
final byte string.

The encoder also remembers what is currently on screen, so cells that
didn't change since the last frame are skipped entirely. Short unchanged
gaps are still re-sent because jumping over them costs more bytes than
repainting them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..term.caps import ColorDepth
from . import palette

CONTINUATION = 0  # codepoint marking the right half of a wide character


@dataclass
class CellFrame:
    cps: np.ndarray                    # (H, W) int32 codepoints
    fg: np.ndarray                     # (H, W, 3) uint8
    bg: Optional[np.ndarray]           # (H, W, 3) uint8 or None (terminal default bg)

    @property
    def shape(self):
        return self.cps.shape

    def copy(self) -> "CellFrame":
        return CellFrame(self.cps.copy(), self.fg.copy(), None if self.bg is None else self.bg.copy())


def _digits(v: np.ndarray, n: int):
    """Decimal digits of non-negative ints -> (N, n) ascii bytes and keep mask
    that strips leading zeros (the last digit is always kept)."""
    v = v.astype(np.int32)
    pows = 10 ** np.arange(n - 1, -1, -1, dtype=np.int32)
    d = (v[:, None] // pows[None, :]) % 10
    keep = v[:, None] >= pows[None, :]
    keep[:, -1] = True
    return (d + 48).astype(np.uint8), keep


def _lit(s: bytes, n: int):
    return np.broadcast_to(np.frombuffer(s, np.uint8), (n, len(s)))


def _utf8(cps: np.ndarray):
    n = cps.shape[0]
    out = np.zeros((n, 4), np.uint8)
    keep = np.zeros((n, 4), bool)
    c = cps.astype(np.int64)
    one = c < 0x80
    two = (c >= 0x80) & (c < 0x800)
    three = (c >= 0x800) & (c < 0x10000)
    four = c >= 0x10000
    out[:, 0] = np.select(
        [one, two, three, four],
        [c, 0xC0 | (c >> 6), 0xE0 | (c >> 12), 0xF0 | (c >> 18)],
    ).astype(np.uint8)
    out[:, 1] = np.select(
        [two, three, four],
        [0x80 | (c & 0x3F), 0x80 | ((c >> 6) & 0x3F), 0x80 | ((c >> 12) & 0x3F)],
    ).astype(np.uint8)
    out[:, 2] = np.select([three, four], [0x80 | (c & 0x3F), 0x80 | ((c >> 6) & 0x3F)]).astype(np.uint8)
    out[:, 3] = (0x80 | (c & 0x3F)).astype(np.uint8)
    keep[:, 0] = True
    keep[:, 1] = ~one
    keep[:, 2] = three | four
    keep[:, 3] = four
    return out, keep


def _table(strings):
    width = max(len(s) for s in strings)
    tab = np.zeros((len(strings), width), np.uint8)
    lens = np.zeros(len(strings), np.int32)
    for i, s in enumerate(strings):
        b = s.encode()
        tab[i, :len(b)] = np.frombuffer(b, np.uint8)
        lens[i] = len(b)
    return tab, lens


_FG256 = _table([f"38;5;{i}" for i in range(256)])
_BG256 = _table([f"48;5;{i}" for i in range(256)])
_FG16 = _table([str(c) for c in (30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97)])
_BG16 = _table([str(c) for c in (40, 41, 42, 43, 44, 45, 46, 47, 100, 101, 102, 103, 104, 105, 106, 107)])


def _gather(table, idx):
    tab, lens = table
    b = tab[idx]
    keep = np.arange(tab.shape[1])[None, :] < lens[idx][:, None]
    return b, keep


def _truecolor_param(prefix: bytes, rgb: np.ndarray):
    n = rgb.shape[0]
    parts_b = [_lit(prefix, n)]
    parts_k = [np.ones((n, len(prefix)), bool)]
    for ch in range(3):
        d, k = _digits(rgb[:, ch], 3)
        parts_b.append(d)
        parts_k.append(k)
        if ch < 2:
            parts_b.append(_lit(b";", n))
            parts_k.append(np.ones((n, 1), bool))
    return np.concatenate(parts_b, 1), np.concatenate(parts_k, 1)


def color_keys(rgb: np.ndarray, depth: ColorDepth, dither: bool = False) -> np.ndarray:
    """Per-cell comparable color id: packed 24-bit RGB or palette index."""
    if depth == ColorDepth.TRUECOLOR:
        return (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    if depth == ColorDepth.ANSI256:
        return palette.quantize(rgb, 256, dither).astype(np.int32)
    if depth == ColorDepth.ANSI16:
        return palette.quantize(rgb, 16, dither).astype(np.int32)
    return np.zeros(rgb.shape[:2], np.int32)


class CellEncoder:
    """Stateful encoder: knows what's on screen and only sends changes."""

    GAP = 5  # repaint unchanged runs shorter than this instead of jumping

    def __init__(self):
        self.invalidate()

    def invalidate(self) -> None:
        self._cps = None
        self._fg = None
        self._bg = None
        self._geom = None

    def invalidate_region(self, y0: int, y1: int, x0: int = 0, x1: Optional[int] = None) -> None:
        """Forget what's on screen in a region (screen coordinates)."""
        if self._cps is None or self._geom is None:
            return
        oy, ox, depth, has_bg = self._geom
        h, w = self._cps.shape
        ry0, ry1 = max(y0 - oy, 0), min(y1 - oy, h)
        rx0 = max(x0 - ox, 0)
        rx1 = w if x1 is None else min(x1 - ox, w)
        if ry0 < ry1 and rx0 < rx1:
            self._cps[ry0:ry1, rx0:rx1] = -1

    def encode(self, frame: CellFrame, oy: int, ox: int, depth: ColorDepth,
               tolerance: int = 0, dither: bool = False, full: bool = False) -> bytes:
        cps = frame.cps
        h, w = cps.shape
        has_bg = frame.bg is not None
        if depth == ColorDepth.MONO:
            has_bg = False

        fgk = color_keys(frame.fg, depth, dither) if depth != ColorDepth.MONO else None
        bgk = color_keys(frame.bg, depth, dither) if has_bg else None

        geom = (oy, ox, depth, has_bg)
        if full or self._cps is None or self._geom != geom or self._cps.shape != (h, w):
            changed = np.ones((h, w), bool)
            self._cps = np.full((h, w), -1, np.int32)
            self._fg = np.zeros((h, w, 3), np.uint8)
            self._bg = np.zeros((h, w, 3), np.uint8)
            self._fgk = np.zeros((h, w), np.int32)
            self._bgk = np.zeros((h, w), np.int32)
            self._geom = geom
        else:
            changed = cps != self._cps
            if depth == ColorDepth.TRUECOLOR and tolerance > 0:
                changed |= np.abs(frame.fg.astype(np.int16) - self._fg).max(-1) > tolerance
                if has_bg:
                    changed |= np.abs(frame.bg.astype(np.int16) - self._bg).max(-1) > tolerance
            else:
                if fgk is not None:
                    changed |= fgk != self._fgk
                if has_bg:
                    changed |= bgk != self._bgk

        emit = self._close_gaps(changed)
        emit &= cps != CONTINUATION
        if not emit.any():
            return b""

        # Update on-screen model for everything we're about to draw.
        self._cps[emit] = cps[emit]
        self._fg[emit] = frame.fg[emit]
        if fgk is not None:
            self._fgk[emit] = fgk[emit]
        if has_bg:
            self._bg[emit] = frame.bg[emit]
            self._bgk[emit] = bgk[emit]
        # Continuation cells ride along with their wide lead character.
        cont = (cps == CONTINUATION) & changed
        self._cps[cont] = CONTINUATION

        k = np.flatnonzero(emit)
        n = k.size
        ys, xs = np.divmod(k, w)

        # --- cursor positioning -----------------------------------------
        need_cup = np.ones(n, bool)
        need_cup[1:] = (k[1:] != k[:-1] + 1) | (xs[1:] == 0)
        cup_b = [_lit(b"\x1b[", n)]
        cup_k = [np.ones((n, 2), bool)]
        d, kk = _digits(ys + oy + 1, 4)
        cup_b.append(d); cup_k.append(kk)
        cup_b.append(_lit(b";", n)); cup_k.append(np.ones((n, 1), bool))
        d, kk = _digits(xs + ox + 1, 4)
        cup_b.append(d); cup_k.append(kk)
        cup_b.append(_lit(b"H", n)); cup_k.append(np.ones((n, 1), bool))
        cup_bytes = np.concatenate(cup_b, 1)
        cup_keep = np.concatenate(cup_k, 1) & need_cup[:, None]

        parts_b = [cup_bytes]
        parts_k = [cup_keep]

        # --- colors ------------------------------------------------------
        if fgk is not None:
            fk = fgk.reshape(-1)[k]
            need_fg = np.ones(n, bool)
            need_fg[1:] = fk[1:] != fk[:-1]
            if has_bg:
                bk = bgk.reshape(-1)[k]
                need_bg = np.ones(n, bool)
                need_bg[1:] = bk[1:] != bk[:-1]
            else:
                need_bg = np.zeros(n, bool)
            need_sgr = need_fg | need_bg

            parts_b.append(_lit(b"\x1b[", n)); parts_k.append(np.repeat(need_sgr[:, None], 2, 1))
            if depth == ColorDepth.TRUECOLOR:
                fb, fkeep = _truecolor_param(b"38;2;", frame.fg.reshape(-1, 3)[k])
            elif depth == ColorDepth.ANSI256:
                fb, fkeep = _gather(_FG256, fk)
            else:
                fb, fkeep = _gather(_FG16, fk)
            parts_b.append(fb); parts_k.append(fkeep & need_fg[:, None])
            if has_bg:
                parts_b.append(_lit(b";", n)); parts_k.append((need_fg & need_bg)[:, None])
                if depth == ColorDepth.TRUECOLOR:
                    bb, bkeep = _truecolor_param(b"48;2;", frame.bg.reshape(-1, 3)[k])
                elif depth == ColorDepth.ANSI256:
                    bb, bkeep = _gather(_BG256, bk)
                else:
                    bb, bkeep = _gather(_BG16, bk)
                parts_b.append(bb); parts_k.append(bkeep & need_bg[:, None])
            parts_b.append(_lit(b"m", n)); parts_k.append(need_sgr[:, None])

        # --- glyphs ------------------------------------------------------
        gb, gk = _utf8(cps.reshape(-1)[k])
        parts_b.append(gb); parts_k.append(gk)

        buf = np.concatenate(parts_b, 1)
        keep = np.concatenate(parts_k, 1)
        prefix = b"\x1b[0m"
        return prefix + buf[keep].tobytes() + b"\x1b[0m"

    def _close_gaps(self, changed: np.ndarray) -> np.ndarray:
        h, w = changed.shape
        if w < 3 or not changed.any():
            return changed.copy()
        idx = np.broadcast_to(np.arange(w), (h, w))
        left = np.where(changed, idx, -10**6)
        left = np.maximum.accumulate(left, axis=1)
        right = np.where(changed, idx, 10**6)
        right = np.minimum.accumulate(right[:, ::-1], axis=1)[:, ::-1]
        return changed | ((right - left) <= self.GAP)
