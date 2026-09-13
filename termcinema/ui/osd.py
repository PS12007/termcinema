"""On-screen display: progress bar, info line, toasts, help, stats,
settings menu, subtitles and the buffering spinner."""
from __future__ import annotations

import time

from .canvas import (ACCENT, ACCENT2, DIM, FAINT, GOOD, PANEL, WARN, WHITE, Canvas, fmt_time, text_width,
                     truncate)

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

HELP = [
    ("Playback", [
        ("Space / k", "play / pause"),
        ("← → / j l", "seek 5s / 10s"),
        ("PgUp PgDn", "seek 1 min"),
        ("0 … 9", "jump to 0% … 90%"),
        (", .", "frame step (paused)"),
        ("[ ]  ⌫", "speed - / + / reset"),
        ("↑ ↓  m", "volume / mute"),
        ("n p", "next / previous"),
        ("L", "loop"),
    ]),
    ("Picture", [
        ("v V", "render mode"),
        ("c", "color depth"),
        ("e E", "effect"),
        ("f", "fit / fill / stretch"),
        ("z Z", "zoom in / out"),
        ("d", "dithering"),
        ("a", "visualizer style"),
        ("r", "reset picture"),
    ]),
    ("Interface", [
        ("Tab / Enter", "settings menu"),
        ("t", "subtitles"),
        ("i", "stats for nerds"),
        ("o", "OSD auto/always/off"),
        ("s", "screenshot"),
        ("? h", "this help"),
        ("q / Esc", "quit"),
        ("mouse", "click bar to seek"),
    ]),
]


def _blend(c0, c1, t):
    return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))


def progress_bar(cv: Canvas, y: int, x: int, width: int, frac: float, buffered: float = 0.0,
                 chapters=(), live: bool = False) -> None:
    if width < 4:
        return
    if live:
        cv.text(y, x, "━" * width, _blend(ACCENT, (60, 0, 20), 0.5))
        return
    frac = min(max(frac, 0.0), 1.0)
    filled = frac * width
    knob = min(int(filled), width - 1)
    buf_end = min(width, int((frac + max(buffered, 0.0)) * width))
    for i in range(width):
        if i < knob:
            color = _blend(ACCENT, ACCENT2, i / max(width - 1, 1))
            ch = "━"
        elif i == knob:
            color = WHITE
            ch = "●"
        elif i < buf_end:
            color = (110, 110, 125)
            ch = "━"
        else:
            color = (60, 60, 72)
            ch = "─"
        cv.text(y, x + i, ch, color)
    for t in chapters:
        cx = x + int(t * width)
        if x < cx < x + width - 1 and cx != x + knob:
            cv.text(y, cx, "╋" if cx < x + knob else "┼", (200, 200, 210))


def bottom_bar(cv: Canvas, p) -> int:
    """Draw the two-line control bar. Returns the row where it starts."""
    rows, cols = cv.rows, cv.cols
    y = rows - 2
    cv.shade(rows - 3, 0, 1, cols, PANEL, 0.35)
    cv.shade(y, 0, 2, cols, PANEL, 0.78)
    m = p.media
    pos = p.position()
    dur = m.duration if m else 0.0
    live = bool(m and (m.is_live or dur <= 0))
    left = fmt_time(pos)
    right = "● LIVE" if live else fmt_time(dur)
    cv.text(y, 1, left, WHITE)
    rx = cols - 1 - text_width(right)
    cv.text(y, rx, right, (255, 80, 80) if live else DIM)
    bar_x = 2 + text_width(left)
    bar_w = rx - 1 - bar_x
    chapters = [c[0] / dur for c in (m.chapters if m else []) if dur > 0]
    buffered = (p.buffered_seconds() / dur) if dur > 0 else 0.0
    progress_bar(cv, y, bar_x, bar_w, (pos / dur) if dur > 0 else 0.0, buffered, chapters, live)
    p.bar_geometry = (y, bar_x, bar_w)

    # Info line
    y2 = rows - 1
    icon = "⏸" if p.paused else "▶"
    x = cv.text(y2, 1, icon + " ", ACCENT if not p.paused else WARN)
    chips = []
    chips.append((p.mode, ACCENT2))
    chips.append((p.depth.label if p.mode not in ("kitty", "iterm", "sixel") else "pixels", DIM))
    if p.effect != "none":
        chips.append((p.effect, GOOD))
    if p.fit != "fit" or p.zoom > 1.001:
        chips.append((f"{p.fit}{' x%.1f' % p.zoom if p.zoom > 1.001 else ''}", DIM))
    if abs(p.speed - 1.0) > 1e-3:
        chips.append((f"{p.speed:g}x", WARN))
    if p.has_audio():
        chips.append(("muted" if p.muted else f"vol {int(round(p.volume * 100))}%", DIM if not p.muted else WARN))
    if p.subs_on and p.track is not None:
        chips.append(("CC", WHITE))
    if p.loop:
        chips.append(("loop", GOOD))
    if len(p.items) > 1:
        chips.append((f"{p.index + 1}/{len(p.items)}", DIM))
    right_text = "  ".join(c for c, _ in chips)
    rw = text_width(right_text)
    title = m.title if m else ""
    if m and m.uploader:
        title = f"{title}  ·  {m.uploader}"
    chapter = p.current_chapter()
    if chapter:
        title = f"{title}  ›  {chapter}"
    cv.text(y2, x, truncate(title, max(cols - x - rw - 4, 0)), WHITE)
    cx = cols - 1 - rw
    for i, (c, color) in enumerate(chips):
        cx = cv.text(y2, cx, c, color)
        if i < len(chips) - 1:
            cx = cv.text(y2, cx, "  ", DIM)
    return rows - 3


def toast(cv: Canvas, text: str) -> None:
    w = min(text_width(text) + 4, cv.cols)
    x = cv.cols - w - 1
    cv.shade(1, x, 1, w, PANEL, 0.85)
    cv.text(1, x, "▌", ACCENT)
    cv.text(1, x + 2, truncate(text, w - 3), WHITE)


def spinner(cv: Canvas, label: str) -> None:
    frame = SPINNER[int(time.monotonic() * 12) % len(SPINNER)]
    s = f"{frame}  {label}"
    w = min(text_width(s) + 6, cv.cols)
    x = (cv.cols - w) // 2
    y = cv.rows // 2
    cv.shade(y - 1, x, 3, w, PANEL, 0.8)
    cv.text(y, x + 3, truncate(s, w - 4), WHITE)


def subtitles(cv: Canvas, text: str, bottom: int) -> None:
    lines = [ln for ln in text.split("\n") if ln.strip()][-3:]
    y = bottom - len(lines)
    for ln in lines:
        ln = truncate(ln.strip(), cv.cols - 4)
        w = text_width(ln) + 2
        x = (cv.cols - w) // 2
        cv.shade(y, x, 1, w, (0, 0, 0), 0.62)
        cv.text(y, x + 1, ln, (255, 255, 240))
        y += 1


def help_panel(cv: Canvas) -> None:
    col_w = 34
    ncols = 3 if cv.cols >= col_w * 3 + 4 else (2 if cv.cols >= col_w * 2 + 4 else 1)
    groups = HELP
    if ncols == 1:
        cols_content = [[g] for g in groups]
    elif ncols == 2:
        cols_content = [[groups[0], groups[2]], [groups[1]]]
    else:
        cols_content = [[g] for g in groups]
    heights = [sum(len(g[1]) + 2 for g in c) for c in cols_content]
    if ncols == 1:
        cols_content = [groups]
        heights = [sum(len(g[1]) + 2 for g in groups)]
    h = min(max(heights) + 3, cv.rows)
    w = min(col_w * len(cols_content) + 4, cv.cols)
    x0 = (cv.cols - w) // 2
    y0 = max((cv.rows - h) // 2, 0)
    cv.box(y0, x0, h, w, "TermCinema  ·  keys")
    for ci, content in enumerate(cols_content):
        y = y0 + 2
        x = x0 + 2 + ci * col_w
        for name, keys in content:
            if y >= y0 + h - 1:
                break
            cv.text(y, x, name.upper(), ACCENT2)
            y += 1
            for k, desc in keys:
                if y >= y0 + h - 1:
                    break
                cv.text(y, x + 1, k, WARN)
                cv.text(y, x + 14, truncate(desc, col_w - 15), WHITE)
                y += 1
            y += 1


def stats_panel(cv: Canvas, lines: list) -> None:
    w = min(max(text_width(a) + text_width(b) for a, b in lines) + 17, cv.cols - 2)
    h = len(lines) + 2
    cv.box(0, 1, h, w, "stats for nerds", title_color=ACCENT2)
    for i, (k, v) in enumerate(lines):
        cv.text(1 + i, 3, k, DIM)
        cv.text(1 + i, 15, truncate(v, w - 17), WHITE)


def menu_panel(cv: Canvas, items: list, selected: int) -> None:
    """items: [(label, value_text)]"""
    w = min(58, cv.cols - 2)
    h = min(len(items) + 4, cv.rows)
    x0 = (cv.cols - w) // 2
    y0 = max((cv.rows - h) // 2, 0)
    cv.box(y0, x0, h, w, "settings")
    visible = h - 4
    start = max(0, min(selected - visible // 2, len(items) - visible))
    for row, i in enumerate(range(start, min(start + visible, len(items)))):
        label, value = items[i]
        y = y0 + 2 + row
        sel = i == selected
        if sel:
            cv.shade(y, x0 + 1, 1, w - 2, (70, 20, 55), 0.9)
        cv.text(y, x0 + 3, ("› " if sel else "  ") + label, WHITE if sel else DIM)
        if value is not None:
            v = f"‹ {value} ›" if sel else value
            vx = x0 + w - 3 - text_width(v)
            cv.text(y, vx, v, ACCENT if sel else WHITE)
    cv.text(y0 + h - 1, x0 + 2, " ↑↓ select  ←→ change  Enter apply  Esc close ", FAINT)


def title_card(cv: Canvas, title: str, subtitle: str = "") -> None:
    y = cv.rows // 2 - 1
    t = truncate(title, cv.cols - 8)
    x = (cv.cols - text_width(t)) // 2
    cv.gradient_text(y, x, t, ACCENT, ACCENT2)
    if subtitle:
        s = truncate(subtitle, cv.cols - 8)
        cv.text(y + 2, (cv.cols - text_width(s)) // 2, s, DIM)
