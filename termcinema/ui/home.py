"""Home screen and search-results picker.

Both are drawn with the same cell pipeline as video, so thumbnails are
rendered live with the block renderer right next to the result list.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..render import text as text_renderers
from ..render.encode import CellEncoder, CellFrame
from ..term import control
from ..term.keys import Key, Mouse
from .canvas import ACCENT, ACCENT2, DIM, FAINT, GOOD, PANEL, WARN, WHITE, Canvas, fmt_time, text_width, truncate

DEMO_ENTRIES = [
    ("demo:mandelbrot", "Mandelbrot deep zoom"),
    ("demo:life", "Game of Life"),
    ("demo:gradients", "Flowing gradients"),
    ("demo:synth", "Synth + audio visualizer"),
    ("demo:zoneplate", "Zone plate (moiré test)"),
    ("webcam", "Your webcam"),
    ("screen", "Your desktop (recursion!)"),
]


def _logo_image(width: int, height: int, t: float, bg=None) -> np.ndarray:
    """Procedural logo: text rasterized, filled with a drifting gradient."""
    mask = np.zeros((height, width), np.uint8)
    label = "TERMCINEMA"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.0
    thick = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    scale = min(width * 0.94 / tw, height * 0.78 / th)
    thick = max(1, int(scale * 2.2))
    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(mask, label, ((width - tw) // 2, (height + th) // 2), font, scale, 255, thick, cv2.LINE_AA)
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    phase = (xs * 0.9 + ys * 0.35 - t * 0.12) % 1.0
    tri = np.abs(phase * 2 - 1)
    c0 = np.array(ACCENT, np.float32)
    c1 = np.array(ACCENT2, np.float32)
    grad = c0 * tri[..., None] + c1 * (1 - tri[..., None])
    m = (mask.astype(np.float32) / 255.0)[..., None]
    shine = np.clip(1.0 - np.abs(((xs - (t * 0.35 % 1.6) + 0.3)) * 6), 0, 1)[..., None] * 0.6
    img = grad * m + 255 * shine * m
    if bg is not None:
        img = img + np.asarray(bg, np.float32) * (1 - m)
    return np.clip(img, 0, 255).astype(np.uint8)


class _Screen:
    def __init__(self, out: control.Output, depth):
        self.out = out
        self.depth = depth
        self.encoder = CellEncoder()
        self.blocks = text_renderers.Blocks()
        self.size = (0, 0)

    def begin(self):
        cols, rows = control.terminal_size()
        if (cols, rows) != self.size:
            self.size = (cols, rows)
            self.out.write("\x1b[0m\x1b[2J")
            self.encoder.invalidate()
        frame = CellFrame(np.full((rows, cols), 0x20, np.int32), np.zeros((rows, cols, 3), np.uint8),
                          np.zeros((rows, cols, 3), np.uint8))
        # subtle vertical background gradient
        g = np.linspace(14, 4, rows, dtype=np.float32)[:, None, None]
        frame.bg[:] = (g * np.array([1.0, 0.8, 1.6], np.float32)).astype(np.uint8)
        frame.fg[:] = frame.bg
        return frame, Canvas(rows, cols)

    def paste_image(self, frame: CellFrame, img: np.ndarray, y: int, x: int, w: int, h: int) -> None:
        if w < 2 or h < 1:
            return
        px = cv2.resize(img, (w * 2, h * 4), interpolation=cv2.INTER_AREA)
        cf = self.blocks.render(px)
        rows, cols = frame.cps.shape
        hh, ww = min(h, rows - y), min(w, cols - x)
        if hh <= 0 or ww <= 0:
            return
        frame.cps[y:y + hh, x:x + ww] = cf.cps[:hh, :ww]
        frame.fg[y:y + hh, x:x + ww] = cf.fg[:hh, :ww]
        frame.bg[y:y + hh, x:x + ww] = cf.bg[:hh, :ww]

    def flush(self, frame: CellFrame, canvas: Canvas) -> None:
        canvas.compose(frame)
        data = self.encoder.encode(frame, 0, 0, self.depth)
        if data:
            self.out.write(control.SYNC_BEGIN.encode() + data + control.SYNC_END.encode())


class Home:
    """Returns ("play", [sources]) / ("search", query) / ("quit", None)."""

    def __init__(self, out, reader, caps, history: list, status_line: str = ""):
        self.screen = _Screen(out, caps.color)
        self.reader = reader
        self.caps = caps
        self.history = history[:12]
        self.query = ""
        self.focus = 0          # 0 search box, 1 recent, 2 demos
        self.sel = [0, 0, 0]
        self.status_line = status_line
        self.message = ""

    def run(self):
        while True:
            for ev in self.reader.drain():
                result = self._handle(ev)
                if result is not None:
                    return result
            self._draw()
            ev = self.reader.get(timeout=1 / 20)
            if ev is not None:
                result = self._handle(ev)
                if result is not None:
                    return result

    def _lists(self):
        return [None, [(h.get("source", ""), h.get("title", "")) for h in self.history], DEMO_ENTRIES]

    def _handle(self, ev):
        if isinstance(ev, Mouse):
            return None
        if not isinstance(ev, Key):
            return None
        k = ev.name
        lists = self._lists()
        if k in ("esc", "ctrl+c"):
            if self.focus == 0 and self.query:
                self.query = ""
                return None
            return ("quit", None)
        if k == "tab":
            self.focus = (self.focus + 1) % 3
            if self.focus == 1 and not lists[1]:
                self.focus = 2
            return None
        if k == "shift+tab":
            self.focus = (self.focus - 1) % 3
            return None
        if k == "down":
            if self.focus == 0:
                self.focus = 1 if lists[1] else 2
            else:
                self.sel[self.focus] = min(self.sel[self.focus] + 1, len(lists[self.focus]) - 1)
            return None
        if k == "up":
            if self.focus in (1, 2):
                if self.sel[self.focus] == 0:
                    self.focus = 0
                else:
                    self.sel[self.focus] -= 1
            return None
        if self.focus in (1, 2):
            if k in ("left", "right"):
                self.focus = 1 if k == "left" and lists[1] else 2
                return None
            if k == "enter":
                items = lists[self.focus]
                if items:
                    return ("play", [items[self.sel[self.focus]][0]])
            if k == "q":
                return ("quit", None)
            if len(k) == 1 or k == "space":
                self.focus = 0
            else:
                return None
        # search box
        if k == "enter":
            q = self.query.strip().strip('"')
            if not q:
                return None
            from ..media.source import is_url, looks_like_path
            import os

            if is_url(q) or os.path.exists(q) or q.lower().startswith(("demo", "webcam", "screen")):
                return ("play", [q])
            if looks_like_path(q):
                self.message = f"Not found: {q}"
                return None
            return ("search", q)
        if k == "backspace":
            self.query = self.query[:-1]
        elif k == "ctrl+u":
            self.query = ""
        elif k == "ctrl+w":
            self.query = self.query.rstrip().rsplit(" ", 1)[0] if " " in self.query.strip() else ""
        elif k == "space":
            self.query += " "
        elif len(k) == 1:
            self.query += k
        self.message = ""
        return None

    def _draw(self):
        frame, cv = self.screen.begin()
        rows, cols = frame.cps.shape
        t = time.monotonic()
        logo_h = max(3, min(7, rows // 5))
        logo_w = min(cols - 4, logo_h * 11)
        lx = (cols - logo_w) // 2
        bg_rows = frame.bg[1:1 + logo_h, lx].astype(np.float32)                 # (logo_h, 3)
        bg_img = np.repeat(np.repeat(bg_rows[:, None, :], 4, axis=0), logo_w * 2, axis=1)
        self.screen.paste_image(frame, _logo_image(logo_w * 2, logo_h * 4, t, bg_img), 1, lx, logo_w, logo_h)
        y = 1 + logo_h
        tag = "watch anything in your terminal"
        cv.text(y, (cols - len(tag)) // 2, tag, DIM)
        y += 2

        # search box
        bw = min(cols - 4, 72)
        bx = (cols - bw) // 2
        focused = self.focus == 0
        cv.box(y, bx, 3, bw, "", border=ACCENT if focused else FAINT, alpha=0.9)
        placeholder = "Search YouTube, or paste a URL / file path"
        shown = self.query if self.query else placeholder
        caret = "█" if focused and int(t * 2) % 2 == 0 else " "
        inner = bw - 7
        if self.query and text_width(shown) > inner - 1:
            shown = "…" + shown[-(inner - 2):]
        cv.text(y + 1, bx + 2, "⌕", ACCENT2)
        x = cv.text(y + 1, bx + 4, truncate(shown, inner), WHITE if self.query else FAINT)
        if focused:
            cv.text(y + 1, x if self.query else bx + 4, caret, ACCENT)
        y += 4
        if self.message:
            cv.text(y - 1, bx + 2, self.message, WARN)

        # columns
        lists = self._lists()
        col_w = (bw - 2) // 2
        heads = [(1, "RECENT", bx), (2, "DEMOS & DEVICES", bx + col_w + 2)]
        max_items = max(rows - y - 3, 1)
        for idx, head, x0 in heads:
            items = lists[idx]
            cv.text(y, x0, head, ACCENT2 if self.focus == idx else DIM)
            if not items:
                cv.text(y + 1, x0, "nothing yet", FAINT)
                continue
            start = max(0, min(self.sel[idx] - max_items // 2, len(items) - max_items))
            for row, i in enumerate(range(start, min(start + max_items, len(items)))):
                src, label = items[i]
                sel = self.focus == idx and self.sel[idx] == i
                yy = y + 1 + row
                if sel:
                    cv.shade(yy, x0 - 1, 1, col_w + 1, (70, 20, 55), 0.9)
                name = label or src
                cv.text(yy, x0, ("› " if sel else "  ") + truncate(name, col_w - 3), WHITE if sel else DIM)

        hint = "type to search  ·  Tab/↑↓ navigate  ·  Enter play  ·  Esc quit"
        cv.text(rows - 1, 1, truncate(hint, cols - 2), FAINT)
        if self.status_line:
            s = truncate(self.status_line, max(cols - text_width(hint) - 6, 0))
            cv.text(rows - 1, cols - 1 - text_width(s), s, FAINT)
        self.screen.flush(frame, cv)


class Picker:
    """Search results with live thumbnails. Returns list of URLs or None."""

    def __init__(self, out, reader, caps, query: str, results: list):
        self.screen = _Screen(out, caps.color)
        self.reader = reader
        self.query = query
        self.results = results
        self.sel = 0
        self.marked: list = []
        self.thumbs: dict = {}
        self._fetching: set = set()

    def _thumb(self, r) -> Optional[np.ndarray]:
        if r.id in self.thumbs:
            return self.thumbs[r.id]
        if r.thumbnail and r.id not in self._fetching:
            self._fetching.add(r.id)

            def work(res=r):
                from ..media import ytdl

                try:
                    data = ytdl.fetch_bytes(res.thumbnail, timeout=8)
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    self.thumbs[res.id] = None if img is None else img[:, :, ::-1]
                except Exception:
                    self.thumbs[res.id] = None
            threading.Thread(target=work, daemon=True).start()
        return None

    def run(self):
        # warm up the first few thumbnails in parallel
        for r in self.results[:6]:
            self._thumb(r)
        while True:
            for ev in self.reader.drain():
                res = self._handle(ev)
                if res is not None:
                    return res or None
            self._draw()
            ev = self.reader.get(timeout=1 / 15)
            if ev is not None:
                res = self._handle(ev)
                if res is not None:
                    return res or None

    def _handle(self, ev):
        if not isinstance(ev, Key):
            return None
        k = ev.name
        n = len(self.results)
        if k in ("esc", "q", "ctrl+c"):
            return []
        if k in ("down", "j"):
            self.sel = min(self.sel + 1, n - 1)
        elif k in ("up", "k"):
            self.sel = max(self.sel - 1, 0)
        elif k == "pgdn":
            self.sel = min(self.sel + 8, n - 1)
        elif k == "pgup":
            self.sel = max(self.sel - 8, 0)
        elif k == "space":
            u = self.results[self.sel].url
            if u in self.marked:
                self.marked.remove(u)
            else:
                self.marked.append(u)
        elif k == "enter":
            if self.marked:
                return list(self.marked)
            return [self.results[self.sel].url]
        elif k == "a":
            return [r.url for r in self.results]
        return None

    def _draw(self):
        frame, cv = self.screen.begin()
        rows, cols = frame.cps.shape
        wide = cols >= 90
        list_w = int(cols * 0.56) if wide else cols - 2
        cv.text(0, 1, "TermCinema", ACCENT)
        cv.text(0, 12, "›  " + truncate(self.query, cols - 20), WHITE)
        cv.text(0, cols - 1 - len(f"{len(self.results)} results"), f"{len(self.results)} results", DIM)
        top = 2
        visible = max(rows - top - 2, 1)
        start = max(0, min(self.sel - visible // 2, len(self.results) - visible))
        for row, i in enumerate(range(start, min(start + visible, len(self.results)))):
            r = self.results[i]
            y = top + row
            sel = i == self.sel
            if sel:
                cv.shade(y, 0, 1, list_w + 1, (70, 20, 55), 0.92)
            mark = "●" if r.url in self.marked else " "
            cv.text(y, 1, mark, GOOD)
            cv.text(y, 3, f"{i + 1:>2}", ACCENT if sel else FAINT)
            dur = "LIVE" if r.is_live else (fmt_time(r.duration) if r.duration else "")
            meta_w = 8
            title_w = list_w - 7 - meta_w
            cv.text(y, 6, truncate(r.title, title_w), WHITE if sel else (200, 200, 210))
            cv.text(y, list_w - len(dur), dur, (255, 90, 90) if r.is_live else DIM)
        if wide and self.results:
            r = self.results[self.sel]
            px = list_w + 3
            pw = cols - px - 2
            ph = max(3, int(pw * 9 / 16 / 2.0))
            ph = min(ph, rows - 10)
            img = self._thumb(r)
            if img is not None:
                self.screen.paste_image(frame, img, top, px, pw, ph)
            else:
                cv.shade(top, px, ph, pw, PANEL, 0.9)
                cv.text(top + ph // 2, px + max(pw // 2 - 5, 0), "loading…", FAINT)
            y = top + ph + 1
            words = r.title.split()
            line = ""
            lines = []
            for w in words:
                if text_width(line + " " + w) > pw:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            for ln in lines[:3]:
                cv.text(y, px, truncate(ln, pw), WHITE)
                y += 1
            y += 1
            if r.channel:
                cv.text(y, px, truncate(r.channel, pw), ACCENT2)
                y += 1
            meta = []
            if r.duration:
                meta.append(fmt_time(r.duration))
            if r.views:
                meta.append(_views(r.views))
            if r.is_live:
                meta.append("● LIVE")
            cv.text(y, px, "  ·  ".join(meta), DIM)
        hint = "↑↓ select  ·  Enter play  ·  Space queue  ·  a play all  ·  Esc back"
        cv.text(rows - 1, 1, truncate(hint, cols - 2), FAINT)
        if self.marked:
            s = f"{len(self.marked)} queued"
            cv.text(rows - 1, cols - 1 - len(s), s, GOOD)
        self.screen.flush(frame, cv)


def _views(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit} views".replace(".0", "")
    return f"{n} views"


def loading(out, reader, caps, label: str, work):
    """Run `work()` in a thread while showing a spinner. Returns (ok, result|error)."""
    screen = _Screen(out, caps.color)
    box: dict = {}

    def runner():
        try:
            box["result"] = work()
        except Exception as e:  # noqa: BLE001
            box["error"] = e
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    while th.is_alive():
        for ev in reader.drain():
            if isinstance(ev, Key) and ev.name in ("esc", "ctrl+c", "q"):
                return False, None
        frame, cv = screen.begin()
        rows, cols = frame.cps.shape
        from . import osd

        osd.spinner(cv, label)
        screen.flush(frame, cv)
        time.sleep(1 / 20)
    if "error" in box:
        return False, box["error"]
    return True, box.get("result")
