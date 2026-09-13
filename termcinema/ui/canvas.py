"""A cell canvas for drawing the on-screen display over video.

The OSD is composed *into* the cell grid rather than printed on top, so
panels can be translucent: cells under a panel keep their glyph and get
their colors blended toward the panel color, letting the video show
through. Text cells replace the glyph but keep a blended background.
"""
from __future__ import annotations

import unicodedata

import numpy as np

from ..render.encode import CONTINUATION, CellFrame

TRANSPARENT = -1

# Palette
WHITE = (240, 240, 245)
DIM = (150, 150, 165)
FAINT = (95, 95, 110)
ACCENT = (255, 70, 140)
ACCENT2 = (80, 200, 255)
GOOD = (90, 230, 140)
WARN = (255, 190, 70)
PANEL = (12, 12, 20)


def char_width(ch: str) -> int:
    if not ch or ch == "‍":
        return 0
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def text_width(s: str) -> int:
    return sum(char_width(c) for c in s)


def truncate(s: str, width: int, ellipsis: str = "…") -> str:
    if width <= 0:
        return ""
    if text_width(s) <= width:
        return s
    out = []
    w = 0
    for c in s:
        cw = char_width(c)
        if w + cw > width - 1:
            break
        out.append(c)
        w += cw
    return "".join(out) + ellipsis


class Canvas:
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.cps = np.full((rows, cols), TRANSPARENT, np.int32)
        self.fg = np.zeros((rows, cols, 3), np.uint8)
        self.bg = np.zeros((rows, cols, 3), np.uint8)
        self.alpha = np.zeros((rows, cols), np.float32)   # panel opacity
        self.dirty = False

    # -- primitives --------------------------------------------------------
    def shade(self, y: int, x: int, h: int, w: int, color=PANEL, alpha: float = 0.72) -> None:
        y0, y1 = max(y, 0), min(y + h, self.rows)
        x0, x1 = max(x, 0), min(x + w, self.cols)
        if y0 >= y1 or x0 >= x1:
            return
        self.bg[y0:y1, x0:x1] = color
        self.alpha[y0:y1, x0:x1] = np.maximum(self.alpha[y0:y1, x0:x1], alpha)
        self.dirty = True

    def text(self, y: int, x: int, s: str, fg=WHITE, max_width: int | None = None, bg=None, alpha: float = 0.0) -> int:
        """Draw a string; returns the column after the last character."""
        if y < 0 or y >= self.rows:
            return x
        limit = self.cols if max_width is None else min(self.cols, x + max_width)
        col = x
        for ch in s:
            cw = char_width(ch)
            if cw == 0:
                continue
            if col + cw > limit:
                break
            if col >= 0:
                self.cps[y, col] = ord(ch)
                self.fg[y, col] = fg
                if bg is not None:
                    self.bg[y, col] = bg
                    self.alpha[y, col] = max(self.alpha[y, col], alpha)
                if cw == 2 and col + 1 < self.cols:
                    self.cps[y, col + 1] = CONTINUATION
                    self.fg[y, col + 1] = fg
                    if bg is not None:
                        self.bg[y, col + 1] = bg
                        self.alpha[y, col + 1] = max(self.alpha[y, col + 1], alpha)
            col += cw
        self.dirty = True
        return col

    def gradient_text(self, y: int, x: int, s: str, c0, c1) -> int:
        n = max(len(s) - 1, 1)
        col = x
        for i, ch in enumerate(s):
            t = i / n
            color = tuple(int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))
            col = self.text(y, col, ch, color)
        return col

    def box(self, y: int, x: int, h: int, w: int, title: str = "", border=FAINT, alpha: float = 0.82,
            title_color=ACCENT) -> None:
        self.shade(y, x, h, w, PANEL, alpha)
        if w < 2 or h < 2:
            return
        self.text(y, x, "╭" + "─" * (w - 2) + "╮", border)
        for r in range(1, h - 1):
            self.text(y + r, x, "│", border)
            self.text(y + r, x + w - 1, "│", border)
        self.text(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", border)
        if title:
            t = f" {truncate(title, w - 6)} "
            self.text(y, x + 2, t, title_color)

    # -- composition ---------------------------------------------------------
    def compose(self, frame: CellFrame) -> CellFrame:
        """Blend this canvas over a full-screen CellFrame (in place)."""
        if not self.dirty:
            return frame
        a = self.alpha[..., None]
        has = a[..., 0] > 0
        if frame.bg is None:
            frame.bg = np.zeros_like(frame.fg)
        if has.any():
            bgf = frame.bg.astype(np.float32)
            fgf = frame.fg.astype(np.float32)
            panel = self.bg.astype(np.float32)
            new_bg = bgf * (1 - a) + panel * a
            new_fg = fgf * (1 - a) + panel * a
            frame.bg = np.where(has[..., None], new_bg, bgf).astype(np.uint8)
            frame.fg = np.where(has[..., None], new_fg, fgf).astype(np.uint8)
        txt = self.cps != TRANSPARENT
        if txt.any():
            frame.cps = np.where(txt, self.cps, frame.cps)
            frame.fg = np.where(txt[..., None], self.fg, frame.fg)
        return frame


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN
        seconds = 0
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_time(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    if s.endswith("%"):
        raise ValueError("percent")
    parts = s.split(":")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total
