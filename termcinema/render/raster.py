"""Rasterize a CellFrame back into an RGB image.

Powers the screenshot key and ``--snapshot``: you get a PNG of exactly
what the terminal shows, independent of the font you happen to use.
Block, sextant and octant glyphs are drawn from their exact coverage
masks, braille as dots, and everything else with OpenCV's vector font.
"""
from __future__ import annotations

import numpy as np

from . import glyphs
from .encode import CONTINUATION, CellFrame

CELL_W = 8
CELL_H = 16


def _block_mask(cp: int, cw: int, chh: int):
    if 0x2800 <= cp <= 0x28FF:
        bits = cp - 0x2800
        order = ((0, 0, 0x01), (1, 0, 0x02), (2, 0, 0x04), (0, 1, 0x08), (1, 1, 0x10), (2, 1, 0x20), (3, 0, 0x40), (3, 1, 0x80))
        img = np.zeros((chh, cw), np.float32)
        import cv2

        r = max(1, cw // 5)
        for row, col, bit in order:
            if bits & bit:
                cy = int((row + 0.5) * chh / 4)
                cx = int((col + 0.5) * cw / 2)
                cv2.circle(img, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
        return img
    cov = glyphs.coverage_for_codepoint(cp)
    if cov is None:
        return None
    gh, gw, mask = cov
    bits = ((mask >> np.arange(gh * gw)) & 1).reshape(gh, gw).astype(np.float32)
    return np.kron(bits, np.ones((chh // gh + 1, cw // gw + 1), np.float32))[:chh, :cw]


# Line-drawing glyphs: (left, right, up, down) arm weights, 1 = light, 2 = heavy.
_LINES = {
    0x2500: (1, 1, 0, 0), 0x2501: (2, 2, 0, 0), 0x2502: (0, 0, 1, 1), 0x2503: (0, 0, 2, 2),
    0x253C: (1, 1, 1, 1), 0x254B: (2, 2, 2, 2), 0x256D: (0, 1, 0, 1), 0x256E: (1, 0, 0, 1),
    0x2570: (0, 1, 1, 0), 0x256F: (1, 0, 1, 0), 0x250C: (0, 1, 0, 1), 0x2510: (1, 0, 0, 1),
    0x2514: (0, 1, 1, 0), 0x2518: (1, 0, 1, 0),
}


def _symbol_mask(cp: int, cw: int, chh: int):
    import cv2

    img = np.zeros((chh, cw), np.float32)
    cx, cy = cw // 2, chh // 2
    if cp in _LINES:
        left, right, up, down = _LINES[cp]
        for weight, p0, p1 in ((left, (0, cy), (cx, cy)), (right, (cx, cy), (cw - 1, cy)),
                               (up, (cx, 0), (cx, cy)), (down, (cx, cy), (cx, chh - 1))):
            if weight:
                cv2.line(img, p0, p1, 1.0, weight)
        return img
    if cp in (0x25CF, 0x2022):          # ● •
        cv2.circle(img, (cx, cy), max(2, cw // 3), 1.0, -1, cv2.LINE_AA)
        return img
    if cp in (0x25B6, 0x25BA):          # ▶
        pts = np.array([[1, chh // 4], [cw - 1, cy], [1, chh * 3 // 4]], np.int32)
        cv2.fillPoly(img, [pts], 1.0, cv2.LINE_AA)
        return img
    if cp == 0x23F8:                    # ⏸
        img[chh // 4: chh * 3 // 4, 1:cw // 2 - 1] = 1
        img[chh // 4: chh * 3 // 4, cw // 2 + 1:cw - 1] = 1
        return img
    arrows = {0x2190: ((cw - 1, cy), (1, cy)), 0x2192: ((1, cy), (cw - 1, cy)),
              0x2191: ((cx, chh * 3 // 4), (cx, chh // 4)), 0x2193: ((cx, chh // 4), (cx, chh * 3 // 4))}
    if cp in arrows:
        cv2.arrowedLine(img, arrows[cp][0], arrows[cp][1], 1.0, 1, cv2.LINE_AA, tipLength=0.45)
        return img
    if cp in (0x2039, 0x203A):          # ‹ ›
        d = 1 if cp == 0x203A else -1
        pts = np.array([[cx - 2 * d, cy - 3], [cx + 2 * d, cy], [cx - 2 * d, cy + 3]], np.int32)
        cv2.polylines(img, [pts], False, 1.0, 1, cv2.LINE_AA)
        return img
    if cp in (0x00B7, 0x2027):          # ·
        cv2.circle(img, (cx, cy), 1, 1.0, -1)
        return img
    if cp == 0x232B:                    # ⌫
        pts = np.array([[1, cy], [cw // 3, cy - 3], [cw - 1, cy - 3], [cw - 1, cy + 3], [cw // 3, cy + 3]], np.int32)
        cv2.polylines(img, [pts], True, 1.0, 1, cv2.LINE_AA)
        return img
    if cp == 0x2315:                    # ⌕
        cv2.circle(img, (cx - 1, cy - 1), max(2, cw // 3), 1.0, 1, cv2.LINE_AA)
        cv2.line(img, (cx + 1, cy + 2), (cw - 1, chh * 3 // 4), 1.0, 1, cv2.LINE_AA)
        return img
    if cp == 0x2026:                    # …
        for x in (cw // 6, cw // 2, cw * 5 // 6):
            cv2.circle(img, (x, chh * 3 // 4), 1, 1.0, -1)
        return img
    return None


def _text_mask(cp: int, cw: int, chh: int):
    import cv2

    img = np.zeros((chh, cw), np.uint8)
    try:
        s = chr(cp)
    except ValueError:
        return img.astype(np.float32)
    if not s.isprintable() or ord(s) > 126:
        s = "?" if ord(s) > 126 and not s.isspace() else " "
    scale = chh / 30.0
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, s, ((cw - tw) // 2, int(chh * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def rasterize(frame: CellFrame, cell_w: int = CELL_W, cell_h: int = CELL_H) -> np.ndarray:
    cps = frame.cps
    rows, cols = cps.shape
    uniq, inv = np.unique(cps, return_inverse=True)
    atlas = np.zeros((len(uniq), cell_h, cell_w), np.float32)
    for i, cp in enumerate(uniq.tolist()):
        if cp in (CONTINUATION, 0x20):
            continue
        m = _block_mask(cp, cell_w, cell_h)
        if m is None:
            m = _symbol_mask(cp, cell_w, cell_h)
        atlas[i] = m if m is not None else _text_mask(cp, cell_w, cell_h)
    cov = atlas[inv.reshape(rows, cols)]                        # (R, C, h, w)
    fg = frame.fg.astype(np.float32)[:, :, None, None, :]
    bg = (frame.bg if frame.bg is not None else np.zeros_like(frame.fg)).astype(np.float32)[:, :, None, None, :]
    img = bg + (fg - bg) * cov[..., None]
    img = img.transpose(0, 2, 1, 3, 4).reshape(rows * cell_h, cols * cell_w, 3)
    return np.clip(img, 0, 255).astype(np.uint8)
