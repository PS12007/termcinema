"""Where the picture goes on screen, and which part of the source it shows."""
from __future__ import annotations

from dataclasses import dataclass

FIT_MODES = ("fit", "fill", "stretch")


@dataclass(frozen=True)
class Layout:
    cols: int
    rows: int
    x0: int
    y0: int
    w: int          # cells
    h: int          # cells
    crop: tuple     # (x, y, w, h) as fractions of the source frame


def compute(cols: int, rows: int, src_aspect: float, cell_aspect: float, fit: str = "fit",
            zoom: float = 1.0, reserve: int = 0, pan: tuple = (0.0, 0.0)) -> Layout:
    avail_w = max(cols, 1)
    avail_h = max(rows - reserve, 1)
    src_aspect = src_aspect if src_aspect > 0 else 16 / 9
    cell_aspect = cell_aspect if cell_aspect > 0 else 2.0
    cells_aspect = src_aspect * cell_aspect   # width/height of the picture measured in cells

    cx, cy, cw, ch = 0.0, 0.0, 1.0, 1.0
    if fit == "stretch":
        w, h = avail_w, avail_h
    elif fit == "fill":
        w, h = avail_w, avail_h
        area_aspect = w / h
        if area_aspect > cells_aspect:      # area wider than picture: crop top/bottom
            ch = cells_aspect / area_aspect
            cy = (1 - ch) / 2
        else:
            cw = area_aspect / cells_aspect
            cx = (1 - cw) / 2
    else:
        w = avail_w
        h = int(round(w / cells_aspect))
        if h > avail_h:
            h = avail_h
            w = int(round(h * cells_aspect))
        w, h = max(1, min(w, avail_w)), max(1, min(h, avail_h))

    if zoom > 1.0001:
        nw, nh = cw / zoom, ch / zoom
        px, py = pan
        cx = cx + (cw - nw) / 2 + px * (cw - nw) / 2
        cy = cy + (ch - nh) / 2 + py * (ch - nh) / 2
        cw, ch = nw, nh

    x0 = (avail_w - w) // 2
    y0 = (avail_h - h) // 2
    return Layout(cols, rows, x0, y0, w, h, (cx, cy, cw, ch))
