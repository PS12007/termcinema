from __future__ import annotations

from . import utils


def status_bar(
    *,
    title: str,
    width: int,
    playing: bool,
    pos: float,
    duration: float,
    fps: float,
    target_fps: float,
    mode: str,
    color_mode: str,
    volume: int,
    muted: bool,
    speed: float,
    dropped: int,
) -> str:
    state = "▶" if playing else "⏸"
    pos_s = utils.fmt_time(pos)
    dur_s = utils.fmt_time(duration) if duration else "--:--"
    vol = "muted" if muted else f"{volume}%"

    left = f" {state} {pos_s}/{dur_s}  {title}"
    right = f"{fps:5.1f}/{target_fps:.0f}fps  {mode}  {color_mode}  vol:{vol}  {speed:.2f}x  drop:{dropped} "

    bar_width = max(width - len(left) - len(right), 1)
    pct = 0.0 if duration <= 0 else utils.clamp(pos / duration, 0, 1)
    filled = int(bar_width * pct)
    bar = "─" * filled + "●" + "─" * max(bar_width - filled - 1, 0)

    line = (left + bar + right)[:width]
    return "\x1b[7m" + line.ljust(width) + "\x1b[0m"
