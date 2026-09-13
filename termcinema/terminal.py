"""Terminal capability detection and low-level control.

Detects color depth, size, and Unicode support so the renderer can pick
the best available mode automatically.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import IntEnum


class ColorSupport(IntEnum):
    MONO = 0
    ANSI16 = 1
    ANSI256 = 2
    TRUECOLOR = 3


@dataclass(frozen=True)
class TerminalCaps:
    columns: int
    rows: int
    color: ColorSupport
    unicode_ok: bool
    # Real terminal character cells are taller than wide. This ratio
    # (height / width of a single glyph cell, in "pixels") is used to
    # avoid stretched/squashed output. ~2.0 is a good default for most
    # monospace terminal fonts.
    font_aspect: float = 2.0


def _detect_color_support() -> ColorSupport:
    if os.environ.get("TERMCINEMA_FORCE_MONO"):
        return ColorSupport.MONO

    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return ColorSupport.TRUECOLOR

    term = os.environ.get("TERM", "")
    if "256color" in term:
        return ColorSupport.ANSI256
    if term in ("xterm", "screen", "vt100", "linux", "ansi"):
        return ColorSupport.ANSI16

    # Windows Terminal / modern ConPTY supports truecolor.
    if os.name == "nt" and os.environ.get("WT_SESSION"):
        return ColorSupport.TRUECOLOR

    # Fall back conservatively but don't cripple things that clearly
    # support at least 256 colors (most modern emulators do).
    if term:
        return ColorSupport.ANSI256
    return ColorSupport.ANSI16


def _detect_unicode_support() -> bool:
    if os.environ.get("TERMCINEMA_FORCE_ASCII"):
        return False
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in enc:
        return True
    # LANG/LC_ALL commonly declare UTF-8 on Linux/macOS even when
    # stdout.encoding is ambiguous (e.g. when piped).
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        val = os.environ.get(var, "").lower()
        if "utf" in val:
            return True
    return os.name != "nt"


def detect_caps() -> TerminalCaps:
    size = shutil.get_terminal_size(fallback=(120, 40))
    return TerminalCaps(
        columns=size.columns,
        rows=max(size.lines - 1, 1),  # reserve a line for the status bar
        color=_detect_color_support(),
        unicode_ok=_detect_unicode_support(),
    )


HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR_SCREEN = "\x1b[2J"
HOME = "\x1b[H"
ALT_SCREEN_ON = "\x1b[?1049h"
ALT_SCREEN_OFF = "\x1b[?1049l"
RESET = "\x1b[0m"


def move_home() -> str:
    return HOME
