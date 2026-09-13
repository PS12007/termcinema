"""Keyboard, mouse and terminal-reply input.

A background thread reads raw input (select() on POSIX, the console
input queue on Windows with VT input enabled) and a small state-machine
parser turns the byte stream into events:

* ``Key("left")``, ``Key("q")``, ``Key("shift+right")`` ...
* ``Mouse(x, y, button, kind)`` from SGR (1006) mouse reports
* ``Reply(seq)`` for answers to terminal queries (DA1, XTWINOPS, kitty
  graphics probe) so capability detection can share the same reader.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass

from .control import IS_WINDOWS

if IS_WINDOWS:
    import msvcrt  # type: ignore
else:
    import select
    import sys


@dataclass(frozen=True)
class Key:
    name: str

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


@dataclass(frozen=True)
class Mouse:
    x: int          # 0-based column
    y: int          # 0-based row
    button: int     # 0 left, 1 middle, 2 right
    kind: str       # press | release | drag | wheel_up | wheel_down


@dataclass(frozen=True)
class Reply:
    seq: str


_CSI_TILDE = {
    "1": "home", "2": "insert", "3": "delete", "4": "end", "5": "pgup", "6": "pgdn",
    "7": "home", "8": "end", "11": "f1", "12": "f2", "13": "f3", "14": "f4",
    "15": "f5", "17": "f6", "18": "f7", "19": "f8", "20": "f9", "21": "f10",
    "23": "f11", "24": "f12",
}
_CSI_LETTER = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
               "P": "f1", "Q": "f2", "R": "f3", "S": "f4", "Z": "shift+tab"}
_MODS = {2: "shift+", 3: "alt+", 4: "shift+alt+", 5: "ctrl+", 6: "ctrl+shift+", 7: "ctrl+alt+", 8: "ctrl+shift+alt+"}
_WIN_LEGACY = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
               "I": "pgup", "Q": "pgdn", "S": "delete", "R": "insert", ";": "f1", "<": "f2",
               "s": "ctrl+left", "t": "ctrl+right"}


class Parser:
    """Incremental VT input parser."""

    def __init__(self):
        self.buf = ""

    def feed(self, data: str) -> list:
        self.buf += data
        events = []
        while self.buf:
            ev, used = self._parse_one(self.buf)
            if used == 0:
                break  # incomplete sequence, wait for more
            self.buf = self.buf[used:]
            if ev is not None:
                events.append(ev)
        return events

    def flush(self) -> list:
        """Called when input has gone quiet: a dangling ESC is a real Esc key."""
        if not self.buf:
            return []
        events = []
        if self.buf.startswith("\x1b"):
            events.append(Key("esc"))
            self.buf = self.buf[1:]
        events.extend(self.feed(""))
        if self.buf:  # unparseable garbage; drop it
            self.buf = ""
        return events

    # -- internals ---------------------------------------------------------
    def _parse_one(self, s: str):
        c = s[0]
        if c == "\x1b":
            return self._parse_escape(s)
        if c in ("\x00", "\xe0") and IS_WINDOWS:
            if len(s) < 2:
                return None, 0
            return Key(_WIN_LEGACY.get(s[1], "unknown")), 2
        if c in "\r\n":
            return Key("enter"), 1
        if c == "\t":
            return Key("tab"), 1
        if c in "\x7f\x08":
            return Key("backspace"), 1
        if c == " ":
            return Key("space"), 1
        o = ord(c)
        if o < 32:
            return Key("ctrl+" + chr(o + 96)), 1
        return Key(c), 1

    def _parse_escape(self, s: str):
        if len(s) < 2:
            return None, 0
        c1 = s[1]
        if c1 == "[":
            i = 2
            while i < len(s) and not ("\x40" <= s[i] <= "\x7e" and not (i == 2 and s[i] in "<?=>")):
                i += 1
            if i >= len(s):
                return None, 0
            body, final = s[2:i], s[i]
            return self._csi(body, final), i + 1
        if c1 == "O":
            if len(s) < 3:
                return None, 0
            return Key(_CSI_LETTER.get(s[2], "unknown")), 3
        if c1 in "_P]^":
            # APC / DCS / OSC / PM string, terminated by ST (ESC \) or BEL.
            for j in range(2, len(s)):
                if s[j] == "\x07":
                    return Reply(s[: j + 1]), j + 1
                if s[j] == "\x1b" and j + 1 < len(s) and s[j + 1] == "\\":
                    return Reply(s[: j + 2]), j + 2
            return None, 0
        if c1 == "\x1b":
            return Key("esc"), 1
        return Key("alt+" + c1), 2

    def _csi(self, body: str, final: str):
        if body.startswith("<") and final in "Mm":
            try:
                b, x, y = (int(v) for v in body[1:].split(";"))
            except ValueError:
                return None
            if b & 64:
                return Mouse(x - 1, y - 1, 0, "wheel_up" if (b & 1) == 0 else "wheel_down")
            button = b & 3
            if final == "m":
                return Mouse(x - 1, y - 1, button, "release")
            return Mouse(x - 1, y - 1, button, "drag" if b & 32 else "press")
        if body.startswith("?") or final in "tcSyn" or "$" in body:
            return Reply("\x1b[" + body + final)
        if final == "~":
            parts = body.split(";")
            name = _CSI_TILDE.get(parts[0], "unknown")
            if len(parts) > 1 and parts[1].isdigit():
                name = _MODS.get(int(parts[1]), "") + name
            return Key(name)
        if final in _CSI_LETTER:
            name = _CSI_LETTER[final]
            parts = body.split(";")
            if len(parts) > 1 and parts[1].isdigit():
                name = _MODS.get(int(parts[1]), "") + name
            return Key(name)
        return None


class InputReader:
    """Background reader thread feeding a queue of parsed events."""

    def __init__(self):
        self.events: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._parser = Parser()

    def start(self) -> "InputReader":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="termcinema-input", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def get(self, timeout: float = 0.0):
        try:
            if timeout <= 0:
                return self.events.get_nowait()
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list:
        out = []
        while True:
            ev = self.get()
            if ev is None:
                return out
            out.append(ev)

    def _emit(self, events) -> None:
        for ev in events:
            self.events.put(ev)

    def _run(self) -> None:
        try:
            if IS_WINDOWS:
                self._run_windows()
            else:
                self._run_posix()
        except Exception:
            pass

    def _run_windows(self) -> None:
        quiet_since = None
        while not self._stop.is_set():
            chunk = []
            while msvcrt.kbhit():
                chunk.append(msvcrt.getwch())
                if len(chunk) > 4096:
                    break
            if chunk:
                self._emit(self._parser.feed("".join(chunk)))
                quiet_since = time.monotonic()
            else:
                if self._parser.buf and quiet_since and time.monotonic() - quiet_since > 0.04:
                    self._emit(self._parser.flush())
                self._stop.wait(0.008)

    def _run_posix(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.04)
            if not r:
                if self._parser.buf:
                    self._emit(self._parser.flush())
                continue
            data = os.read(fd, 4096)
            if not data:
                break
            self._emit(self._parser.feed(data.decode("utf-8", "replace")))
