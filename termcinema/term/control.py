"""Low-level terminal control: escape sequences, Windows console setup,
raw mode, and a binary output writer.

Most "colors look wrong" bugs in terminal video players come from this
layer rather than from the color math: Windows consoles that were never
switched into VT mode, text written through a non-UTF-8 codec, frames
torn in half by partial writes, or auto-wrap scrolling the whole screen
when the last column is filled. Everything here exists to rule those out.
"""
from __future__ import annotations

import os
import sys

IS_WINDOWS = os.name == "nt"

CSI = "\x1b["
ESC = "\x1b"

HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ALT_SCREEN_ON = "\x1b[?1049h"
ALT_SCREEN_OFF = "\x1b[?1049l"
AUTOWRAP_OFF = "\x1b[?7l"
AUTOWRAP_ON = "\x1b[?7h"
CLEAR = "\x1b[2J"
HOME = "\x1b[H"
RESET = "\x1b[0m"
SYNC_BEGIN = "\x1b[?2026h"
SYNC_END = "\x1b[?2026l"
# Button-event mouse tracking + SGR extended coordinates.
MOUSE_ON = "\x1b[?1002h\x1b[?1006h"
MOUSE_OFF = "\x1b[?1002l\x1b[?1006l"
# Kitty keyboard / bracketed paste are left alone on purpose.


def cup(row: int, col: int) -> str:
    """Cursor position, 0-based arguments."""
    return f"\x1b[{row + 1};{col + 1}H"


def fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


# ---------------------------------------------------------------------------
# Windows console setup
# ---------------------------------------------------------------------------

_STD_INPUT_HANDLE = -10
_STD_OUTPUT_HANDLE = -11
_ENABLE_PROCESSED_INPUT = 0x0001
_ENABLE_LINE_INPUT = 0x0002
_ENABLE_ECHO_INPUT = 0x0004
_ENABLE_WINDOW_INPUT = 0x0008
_ENABLE_MOUSE_INPUT = 0x0010
_ENABLE_QUICK_EDIT_MODE = 0x0040
_ENABLE_EXTENDED_FLAGS = 0x0080
_ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
_ENABLE_PROCESSED_OUTPUT = 0x0001
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


class _WinConsole:
    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ctypes = ctypes
        self.wintypes = wintypes
        self.hin = self.k32.GetStdHandle(_STD_INPUT_HANDLE)
        self.hout = self.k32.GetStdHandle(_STD_OUTPUT_HANDLE)
        self.saved_in = self._get(self.hin)
        self.saved_out = self._get(self.hout)
        self.saved_cp = self.k32.GetConsoleOutputCP()
        self.saved_in_cp = self.k32.GetConsoleCP()

    def _get(self, h):
        mode = self.wintypes.DWORD()
        if self.k32.GetConsoleMode(h, self.ctypes.byref(mode)):
            return mode.value
        return None

    def enable_vt_output(self) -> bool:
        if self.saved_out is None:
            return False
        ok = self.k32.SetConsoleMode(
            self.hout, self.saved_out | _ENABLE_PROCESSED_OUTPUT | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
        )
        self.k32.SetConsoleOutputCP(65001)
        return bool(ok)

    def raw_input(self) -> bool:
        if self.saved_in is None:
            return False
        self.k32.SetConsoleCP(65001)
        mode = (_ENABLE_VIRTUAL_TERMINAL_INPUT | _ENABLE_WINDOW_INPUT | _ENABLE_EXTENDED_FLAGS)
        return bool(self.k32.SetConsoleMode(self.hin, mode))

    def restore(self) -> None:
        if self.saved_in is not None:
            self.k32.SetConsoleMode(self.hin, self.saved_in)
        if self.saved_out is not None:
            self.k32.SetConsoleMode(self.hout, self.saved_out)
        if self.saved_cp:
            self.k32.SetConsoleOutputCP(self.saved_cp)
        if self.saved_in_cp:
            self.k32.SetConsoleCP(self.saved_in_cp)


_win: "_WinConsole | None" = None


def _win_console() -> "_WinConsole | None":
    global _win
    if IS_WINDOWS and _win is None:
        try:
            _win = _WinConsole()
        except Exception:
            _win = None
    return _win


def enable_vt() -> bool:
    """Make sure escape sequences are interpreted (Windows needs an
    explicit opt-in per console) and output is UTF-8."""
    if IS_WINDOWS:
        w = _win_console()
        if w is None:
            return False
        return w.enable_vt_output()
    return True


# ---------------------------------------------------------------------------
# Raw mode
# ---------------------------------------------------------------------------

class RawMode:
    """Context manager: unbuffered, no-echo input with VT key sequences."""

    def __init__(self):
        self._saved = None
        self.active = False

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        if IS_WINDOWS:
            w = _win_console()
            self.active = bool(w and w.raw_input())
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            try:
                self._saved = termios.tcgetattr(fd)
                tty.setraw(fd, termios.TCSANOW)
                # Keep output post-processing so "\n" still behaves if we print.
                attrs = termios.tcgetattr(fd)
                attrs[1] |= termios.OPOST
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
                self.active = True
            except Exception:
                self._saved = None
        return self

    def __exit__(self, *exc):
        if IS_WINDOWS:
            w = _win_console()
            if w is not None:
                w.restore()
                w.enable_vt_output()
        elif self._saved is not None:
            import termios

            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            except Exception:
                pass
        self.active = False
        return False


def restore_console() -> None:
    if IS_WINDOWS and _win is not None:
        _win.restore()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class Output:
    """Writes whole frames as UTF-8 bytes in as few syscalls as possible."""

    def __init__(self, stream=None):
        stream = stream or sys.stdout
        self._buf = getattr(stream, "buffer", None)
        self._stream = stream
        self.bytes_written = 0

    def write(self, data) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8", "replace")
        self.bytes_written += len(data)
        if self._buf is not None:
            self._buf.write(data)
            self._buf.flush()
        else:  # pragma: no cover - exotic streams
            self._stream.write(data.decode("utf-8", "replace"))
            self._stream.flush()


def terminal_size(fallback=(120, 36)) -> tuple[int, int]:
    try:
        sz = os.get_terminal_size(sys.__stdout__.fileno())
        if sz.columns > 0 and sz.lines > 0:
            return sz.columns, sz.lines
    except (OSError, ValueError, AttributeError):
        pass
    try:
        import shutil

        sz = shutil.get_terminal_size(fallback)
        return sz.columns, sz.lines
    except Exception:
        return fallback
