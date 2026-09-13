"""A tiny terminal interpreter for tests: understands exactly the subset of
escape sequences the cell encoder emits (CUP, SGR colors/reset, UTF-8 text)
and ignores private modes. Lets tests assert what a real terminal would
display after a sequence of encoded frames."""
from __future__ import annotations

import re

import numpy as np

_SEQ = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")

_BASE16 = [(0, 0, 0)] * 16


class Screen:
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.cps = np.full((rows, cols), 32, np.int32)
        self.fg = np.zeros((rows, cols, 3), np.int32)
        self.bg = np.zeros((rows, cols, 3), np.int32)
        self.fg_idx = np.full((rows, cols), -1, np.int32)
        self.bg_idx = np.full((rows, cols), -1, np.int32)
        self.y = self.x = 0
        self.cur_fg = (255, 255, 255)
        self.cur_bg = (0, 0, 0)
        self.cur_fg_idx = -1
        self.cur_bg_idx = -1

    def feed(self, data: bytes) -> None:
        s = data.decode("utf-8")
        i = 0
        while i < len(s):
            if s[i] == "\x1b":
                m = _SEQ.match(s, i)
                assert m, f"unexpected escape at {i}: {s[i:i + 12]!r}"
                self._csi(m.group(1), m.group(2))
                i = m.end()
                continue
            ch = s[i]
            assert self.x < self.cols, "wrote past the right edge"
            self.cps[self.y, self.x] = ord(ch)
            self.fg[self.y, self.x] = self.cur_fg
            self.bg[self.y, self.x] = self.cur_bg
            self.fg_idx[self.y, self.x] = self.cur_fg_idx
            self.bg_idx[self.y, self.x] = self.cur_bg_idx
            self.x += 1
            i += 1

    def _csi(self, params: str, final: str) -> None:
        if params.startswith("?"):
            return
        if final == "H":
            parts = [int(p) if p else 1 for p in params.split(";")] if params else [1, 1]
            self.y, self.x = parts[0] - 1, parts[1] - 1
        elif final == "J":
            self.cps[:] = 32
        elif final == "m":
            vals = [int(p) if p else 0 for p in params.split(";")] if params else [0]
            j = 0
            while j < len(vals):
                v = vals[j]
                if v == 0:
                    self.cur_fg, self.cur_bg = (255, 255, 255), (0, 0, 0)
                    self.cur_fg_idx = self.cur_bg_idx = -1
                elif v in (38, 48) and vals[j + 1] == 2:
                    rgb = tuple(vals[j + 2:j + 5])
                    if v == 38:
                        self.cur_fg = rgb
                    else:
                        self.cur_bg = rgb
                    j += 4
                elif v in (38, 48) and vals[j + 1] == 5:
                    if v == 38:
                        self.cur_fg_idx = vals[j + 2]
                    else:
                        self.cur_bg_idx = vals[j + 2]
                    j += 2
                elif 30 <= v <= 37 or 90 <= v <= 97:
                    self.cur_fg_idx = v
                elif 40 <= v <= 47 or 100 <= v <= 107:
                    self.cur_bg_idx = v
                j += 1
