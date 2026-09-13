"""Terminal capability detection.

Two layers:

1. Environment heuristics (instant, always available): COLORTERM, TERM,
   TERM_PROGRAM, WT_SESSION, KITTY_WINDOW_ID ...
2. Active probing (only on a real TTY): we send a kitty-graphics query,
   XTWINOPS cell/window size requests, an XTSMGRAPHICS geometry query and
   finally DA1. Every terminal answers DA1, so once its reply arrives we
   know all other answers that are ever coming have already arrived.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum

from . import control


class ColorDepth(IntEnum):
    MONO = 0
    ANSI16 = 1
    ANSI256 = 2
    TRUECOLOR = 3

    @property
    def label(self) -> str:
        return {0: "mono", 1: "16", 2: "256", 3: "truecolor"}[int(self)]

    @classmethod
    def parse(cls, s: str) -> "ColorDepth | None":
        return {
            "mono": cls.MONO, "none": cls.MONO, "16": cls.ANSI16, "256": cls.ANSI256,
            "truecolor": cls.TRUECOLOR, "24bit": cls.TRUECOLOR, "true": cls.TRUECOLOR,
        }.get(s.lower())


@dataclass
class Caps:
    cols: int = 120
    rows: int = 36
    color: ColorDepth = ColorDepth.TRUECOLOR
    terminal: str = "unknown"
    sixel: bool = False
    kitty: bool = False
    iterm: bool = False
    cell_w: float = 0.0     # pixels, 0 = unknown
    cell_h: float = 0.0
    probed: bool = False
    notes: list = field(default_factory=list)

    @property
    def cell_aspect(self) -> float:
        """Height / width of one character cell."""
        if self.cell_w > 0 and self.cell_h > 0:
            return self.cell_h / self.cell_w
        return 2.0

    @property
    def best_pixel_mode(self) -> "str | None":
        if self.kitty:
            return "kitty"
        if self.iterm:
            return "iterm"
        if self.sixel:
            return "sixel"
        return None


def _env(name: str) -> str:
    return os.environ.get(name, "")


def detect_terminal_name() -> str:
    tp = _env("TERM_PROGRAM").lower()
    if _env("KITTY_WINDOW_ID") or _env("TERM") == "xterm-kitty":
        return "kitty"
    if tp == "ghostty" or _env("GHOSTTY_RESOURCES_DIR"):
        return "ghostty"
    if tp == "wezterm" or _env("WEZTERM_EXECUTABLE"):
        return "wezterm"
    if tp == "iterm.app" or _env("ITERM_SESSION_ID"):
        return "iterm2"
    if tp == "vscode":
        return "vscode"
    if _env("WT_SESSION"):
        return "windows-terminal"
    if _env("KONSOLE_VERSION"):
        return "konsole"
    if _env("TERM").startswith("foot"):
        return "foot"
    if tp == "apple_terminal":
        return "apple-terminal"
    if _env("TERM") == "alacritty" or _env("ALACRITTY_SOCKET"):
        return "alacritty"
    if _env("TMUX"):
        return "tmux"
    if control.IS_WINDOWS:
        return "windows-console"
    return _env("TERM") or "unknown"


def detect_color() -> ColorDepth:
    if _env("NO_COLOR"):
        return ColorDepth.MONO
    forced = ColorDepth.parse(_env("TERMCINEMA_COLOR")) if _env("TERMCINEMA_COLOR") else None
    if forced is not None:
        return forced
    ct = _env("COLORTERM").lower()
    if ct in ("truecolor", "24bit"):
        return ColorDepth.TRUECOLOR
    name = detect_terminal_name()
    if name in ("kitty", "ghostty", "wezterm", "iterm2", "vscode", "windows-terminal", "konsole",
                "foot", "alacritty"):
        return ColorDepth.TRUECOLOR
    if name == "apple-terminal":
        return ColorDepth.ANSI256
    term = _env("TERM")
    if control.IS_WINDOWS:
        # Every console host since Windows 10 1703 renders 24-bit color once
        # VT processing is on -- defaulting lower is what made colors wrong.
        return ColorDepth.TRUECOLOR
    if "truecolor" in term or "24bit" in term or "direct" in term:
        return ColorDepth.TRUECOLOR
    if "256" in term:
        return ColorDepth.ANSI256
    if term in ("linux", "vt100", "vt220", "ansi", "dumb"):
        return ColorDepth.ANSI16 if term != "dumb" else ColorDepth.MONO
    return ColorDepth.ANSI256 if term else ColorDepth.TRUECOLOR


def detect_env() -> Caps:
    cols, rows = control.terminal_size()
    name = detect_terminal_name()
    caps = Caps(cols=cols, rows=rows, color=detect_color(), terminal=name)
    if name in ("kitty", "ghostty"):
        caps.kitty = True
    elif name == "wezterm":
        caps.kitty = caps.iterm = caps.sixel = True
    elif name == "iterm2":
        caps.iterm = True
    elif name in ("foot",):
        caps.sixel = True
    elif name == "konsole":
        caps.kitty = True
    return caps


_DA1 = re.compile(r"\x1b\[\?([\d;]*)c")
_CELL = re.compile(r"\x1b\[6;(\d+);(\d+)t")
_WIN = re.compile(r"\x1b\[4;(\d+);(\d+)t")
_GEOM = re.compile(r"\x1b\[\?2;0;(\d+);(\d+)S")
_KITTY_OK = re.compile(r"\x1b_Gi=31;OK")


def probe(caps: Caps, reader=None, timeout: float = 0.6) -> Caps:
    """Actively query the terminal. Needs raw mode and a running InputReader
    (or creates a temporary one). Safe to call on terminals that ignore
    some or all of the queries."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return caps
    from .keys import InputReader, Reply

    own_reader = reader is None
    if own_reader:
        reader = InputReader().start()
    out = control.Output()
    query = (
        "\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\"  # kitty graphics
        "\x1b[16t"                                    # cell size in px
        "\x1b[14t"                                    # text area size in px
        "\x1b[?2;1;0S"                                # XTSMGRAPHICS sixel geometry
        "\x1b[c"                                      # DA1 (sentinel)
    )
    replies = []
    requeue = []
    try:
        out.write(query)
        deadline = time.monotonic() + timeout
        got_da1 = False
        while time.monotonic() < deadline and not got_da1:
            ev = reader.get(timeout=0.02)
            if ev is None:
                continue
            if isinstance(ev, Reply):
                replies.append(ev.seq)
                if _DA1.search(ev.seq):
                    got_da1 = True
            else:
                requeue.append(ev)
    finally:
        for ev in requeue:
            reader.events.put(ev)
        if own_reader:
            reader.stop()

    blob = "".join(replies)
    caps = replace(caps, probed=bool(replies))
    m = _DA1.search(blob)
    if m and "4" in m.group(1).split(";"):
        caps.sixel = True
    if _KITTY_OK.search(blob):
        caps.kitty = True
    m = _CELL.search(blob)
    if m:
        caps.cell_h, caps.cell_w = float(m.group(1)), float(m.group(2))
    else:
        m = _WIN.search(blob) or None
        g = _GEOM.search(blob)
        if m:
            h, w = float(m.group(1)), float(m.group(2))
        elif g:
            w, h = float(g.group(1)), float(g.group(2))
        else:
            w = h = 0.0
        if w > 0 and h > 0 and caps.cols and caps.rows:
            caps.cell_w, caps.cell_h = w / caps.cols, h / caps.rows
    if caps.terminal == "vscode" and caps.sixel:
        # xterm.js' image addon implements the iTerm2 protocol too, and JPEG
        # frames are far smaller than sixel for video.
        caps.iterm = True
    if caps.terminal == "windows-terminal" and caps.sixel and not caps.cell_w:
        caps.cell_w, caps.cell_h = 10.0, 20.0  # WT maps sixels onto a 10x20 virtual cell
    return caps


def detect(active: bool = True, reader=None) -> Caps:
    control.enable_vt()
    caps = detect_env()
    if active:
        try:
            caps = probe(caps, reader=reader)
        except Exception as e:  # pragma: no cover - never let probing crash playback
            caps.notes.append(f"probe failed: {e}")
    return caps
