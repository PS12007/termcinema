"""Non-blocking keyboard input for playback controls.

POSIX: puts the terminal in cbreak mode and polls stdin with select().
Windows: polls msvcrt.kbhit()/getch().

Runs in its own daemon thread and pushes single-character (or short
escape-sequence) keys onto a queue the player reads from each frame.
"""
from __future__ import annotations

import os
import queue
import threading
import sys

_IS_WINDOWS = os.name == "nt"

if not _IS_WINDOWS:
    import select
    import termios
    import tty
else:
    import msvcrt  # type: ignore


class InputHandler:
    def __init__(self):
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        if not _IS_WINDOWS:
            try:
                self._old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self._old_settings = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if not _IS_WINDOWS and self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def _run(self) -> None:
        if _IS_WINDOWS:
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    self.q.put(ch)
                else:
                    self._stop.wait(0.02)
            return

        while not self._stop.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
            except Exception:
                break
            if r:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    # Might be an arrow-key escape sequence: ESC [ A/B/C/D
                    rest = ""
                    r2, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if r2:
                        rest += sys.stdin.read(1)
                        r3, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if r3:
                            rest += sys.stdin.read(1)
                    ch = "\x1b" + rest
                self.q.put(ch)

    def poll(self) -> str | None:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


# Normalized key names for the arrow-key escape sequences we care about.
ARROW_UP = "\x1b[A"
ARROW_DOWN = "\x1b[B"
ARROW_RIGHT = "\x1b[C"
ARROW_LEFT = "\x1b[D"
