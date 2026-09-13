"""True-pixel renderers using terminal graphics protocols.

* ``kitty``  -- kitty graphics protocol (kitty, Ghostty, WezTerm, Konsole):
                zlib-compressed raw RGB, scaled by the terminal into cells.
* ``iterm``  -- iTerm2 inline images (iTerm2, WezTerm, VS Code with
                ``terminal.integrated.enableImages``): JPEG frames.
* ``sixel``  -- DEC sixel (Windows Terminal 1.22+, foot, xterm, mlterm,
                VS Code, WezTerm): 252-color palette with ordered dither and
                a run-length-encoded band writer, JIT-compiled with numba
                when available.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np

try:  # pragma: no cover - optional accelerator
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


class PixelRenderer:
    kind = "pixels"
    name = "pixels"
    description = ""
    needs_cell_size = False
    max_width = 960

    def pixel_size(self, cols: int, rows: int, cell_w: float, cell_h: float) -> tuple[int, int]:
        aspect = (cell_h / cell_w) if cell_w > 0 and cell_h > 0 else 2.0
        cw = cell_w if cell_w > 0 else 10.0
        w = int(min(cols * cw, self.max_width))
        h = int(round(w * rows * aspect / max(cols, 1)))
        return max(w, 2), max(h, 2)

    def render(self, rgb: np.ndarray, row: int, col: int, cols: int, rows: int) -> bytes:
        raise NotImplementedError

    def clear(self) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# kitty
# ---------------------------------------------------------------------------

class Kitty(PixelRenderer):
    name = "kitty"
    description = "Kitty graphics protocol: real pixels (kitty, Ghostty, WezTerm)"
    max_width = 1280
    IMAGE_ID = 7355608

    def render(self, rgb, row, col, cols, rows):
        h, w = rgb.shape[:2]
        payload = base64.standard_b64encode(zlib.compress(np.ascontiguousarray(rgb).tobytes(), 1))
        chunks = [payload[i:i + 4096] for i in range(0, len(payload), 4096)] or [b""]
        out = [f"\x1b[{row + 1};{col + 1}H".encode()]
        head = f"a=T,f=24,o=z,s={w},v={h},c={cols},r={rows},i={self.IMAGE_ID},p=1,q=2,C=1,z=-1".encode()
        for n, chunk in enumerate(chunks):
            more = b"1" if n < len(chunks) - 1 else b"0"
            if n == 0:
                out.append(b"\x1b_G" + head + b",m=" + more + b";" + chunk + b"\x1b\\")
            else:
                out.append(b"\x1b_Gm=" + more + b";" + chunk + b"\x1b\\")
        return b"".join(out)

    def clear(self):
        return f"\x1b_Ga=d,d=I,i={self.IMAGE_ID},q=2\x1b\\".encode()


# ---------------------------------------------------------------------------
# iTerm2
# ---------------------------------------------------------------------------

class ITerm(PixelRenderer):
    name = "iterm"
    description = "iTerm2 inline images: real pixels as JPEG (iTerm2, WezTerm, VS Code)"
    max_width = 1280
    quality = 82

    def render(self, rgb, row, col, cols, rows):
        import cv2

        ok, jpg = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return b""
        data = jpg.tobytes()
        b64 = base64.standard_b64encode(data)
        head = (f"\x1b[{row + 1};{col + 1}H\x1b]1337;File=inline=1;size={len(data)};width={cols};"
                f"height={rows};preserveAspectRatio=0;doNotMoveCursor=1:").encode()
        return head + b64 + b"\x07"


# ---------------------------------------------------------------------------
# sixel
# ---------------------------------------------------------------------------

_R_LEVELS, _G_LEVELS, _B_LEVELS = 6, 7, 6
_NCOLORS = _R_LEVELS * _G_LEVELS * _B_LEVELS  # 252


def _sixel_palette() -> bytes:
    parts = []
    for r in range(_R_LEVELS):
        for g in range(_G_LEVELS):
            for b in range(_B_LEVELS):
                i = (r * _G_LEVELS + g) * _B_LEVELS + b
                parts.append(f"#{i};2;{round(r * 100 / (_R_LEVELS - 1))};"
                             f"{round(g * 100 / (_G_LEVELS - 1))};{round(b * 100 / (_B_LEVELS - 1))}")
    return "".join(parts).encode()


_PALETTE = _sixel_palette()
_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], np.float32) / 64.0


@njit(cache=True)
def _quantize_nb(rgb, bayer):  # pragma: no cover - compiled
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.empty((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            t = bayer[y & 7, x & 7]
            r = int(rgb[y, x, 0] * 5 / 255.0 + t)
            g = int(rgb[y, x, 1] * 6 / 255.0 + t)
            b = int(rgb[y, x, 2] * 5 / 255.0 + t)
            if r > 5:
                r = 5
            if g > 6:
                g = 6
            if b > 5:
                b = 5
            out[y, x] = (r * 7 + g) * 6 + b
    return out


def _quantize_np(rgb, bayer):
    h, w = rgb.shape[:2]
    t = np.tile(bayer, (h // 8 + 1, w // 8 + 1))[:h, :w]
    f = rgb.astype(np.float32) / 255.0
    r = np.minimum((f[..., 0] * 5 + t).astype(np.int32), 5)
    g = np.minimum((f[..., 1] * 6 + t).astype(np.int32), 6)
    b = np.minimum((f[..., 2] * 5 + t).astype(np.int32), 5)
    return ((r * 7 + g) * 6 + b).astype(np.uint8)


@njit(cache=True)
def _put_int(buf, pos, v):  # pragma: no cover - compiled
    if v >= 1000:
        buf[pos] = 48 + (v // 1000) % 10
        pos += 1
    if v >= 100:
        buf[pos] = 48 + (v // 100) % 10
        pos += 1
    if v >= 10:
        buf[pos] = 48 + (v // 10) % 10
        pos += 1
    buf[pos] = 48 + v % 10
    return pos + 1


@njit(cache=True)
def _sixel_body_nb(idx, ncolors, buf):  # pragma: no cover - compiled
    h, w = idx.shape
    bits = np.zeros((ncolors, w), np.uint8)
    used = np.zeros(ncolors, np.uint8)
    order = np.empty(ncolors, np.int32)
    lo = np.zeros(ncolors, np.int32)
    hi = np.zeros(ncolors, np.int32)
    pos = 0
    for band in range(0, h, 6):
        nused = 0
        bh = min(6, h - band)
        for x in range(w):
            for r in range(bh):
                c = idx[band + r, x]
                if used[c] == 0:
                    used[c] = 1
                    order[nused] = c
                    nused += 1
                    lo[c] = x
                bits[c, x] |= 1 << r
                hi[c] = x
        for k in range(nused):
            c = order[k]
            if k > 0:
                buf[pos] = 36  # '$' carriage return within band
                pos += 1
            buf[pos] = 35  # '#'
            pos = _put_int(buf, pos + 1, c)
            x = 0
            end = hi[c] + 1
            # leading empty run
            if lo[c] > 0:
                n = lo[c]
                if n > 3:
                    buf[pos] = 33
                    pos = _put_int(buf, pos + 1, n)
                    buf[pos] = 63
                    pos += 1
                else:
                    for _ in range(n):
                        buf[pos] = 63
                        pos += 1
                x = lo[c]
            while x < end:
                v = bits[c, x]
                run = 1
                while x + run < end and bits[c, x + run] == v:
                    run += 1
                ch = 63 + v
                if run > 3:
                    buf[pos] = 33
                    pos = _put_int(buf, pos + 1, run)
                    buf[pos] = ch
                    pos += 1
                else:
                    for _ in range(run):
                        buf[pos] = ch
                        pos += 1
                x += run
            for xx in range(lo[c], end):
                bits[c, xx] = 0
            used[c] = 0
        buf[pos] = 45  # '-' next band
        pos += 1
    return pos


def _sixel_body_np(idx: np.ndarray) -> bytes:
    import re

    h, w = idx.shape
    weights = (1 << np.arange(6, dtype=np.uint8))[:, None]
    out = []
    rle = re.compile(rb"(.)\1{3,}")
    for band in range(0, h, 6):
        blk = idx[band:band + 6]
        colors = np.unique(blk)
        lines = []
        for c in colors.tolist():
            v = ((blk == c) * weights[: blk.shape[0]]).sum(0).astype(np.uint8) + 63
            row = v.tobytes().rstrip(b"?")
            row = rle.sub(lambda m: b"!%d%c" % (len(m.group(0)), m.group(1)[0]), row)
            lines.append(b"#%d" % c + row)
        out.append(b"$".join(lines) + b"-")
    return b"".join(out)


class Sixel(PixelRenderer):
    name = "sixel"
    description = "DEC sixel graphics: real pixels (Windows Terminal, foot, xterm, VS Code)"
    needs_cell_size = True
    max_width = 800

    def __init__(self):
        self._buf = None

    def pixel_size(self, cols, rows, cell_w, cell_h):
        cw = cell_w if cell_w > 0 else 10.0
        ch = cell_h if cell_h > 0 else 20.0
        w = int(cols * cw)
        h = int(rows * ch)
        if w > self.max_width:
            h = int(h * self.max_width / w)
            w = self.max_width
        h -= h % 6
        return max(w, 6), max(h, 6)

    def render(self, rgb, row, col, cols, rows):
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        if HAVE_NUMBA:
            idx = _quantize_nb(rgb, _BAYER8)
            need = (h // 6 + 1) * (w * 80 + _NCOLORS * 16) + 64
            if self._buf is None or self._buf.size < need:
                self._buf = np.empty(need, np.uint8)
            n = _sixel_body_nb(idx, _NCOLORS, self._buf)
            body = self._buf[:n].tobytes()
        else:
            body = _sixel_body_np(_quantize_np(rgb, _BAYER8))
        head = f"\x1b[{row + 1};{col + 1}H\x1bP0;1;0q\"1;1;{w};{h}".encode()
        return head + _PALETTE + body + b"\x1b\\"
