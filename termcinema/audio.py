"""Audio playback.

We shell out to `ffplay` (part of the FFmpeg suite, already required for
video decoding on most systems) in audio-only mode. This avoids pulling
in a heavy audio dependency and gives us robust decoding of whatever
codec the source uses for free. Sync is timestamp-based: the player
tracks wall-clock-since-start and compares it to each video frame's
presentation time, rather than trusting audio and video to free-run in
lockstep.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from typing import Optional


class AudioPlayer:
    def __init__(self, source: str):
        self.source = source
        self.proc: Optional[subprocess.Popen] = None
        self._start_time = 0.0
        self._paused_at: Optional[float] = None
        self._offset = 0.0
        self.volume = 100  # 0-100, ffplay -volume takes 0-100
        self.available = shutil.which("ffplay") is not None

    def start(self, start_seconds: float = 0.0) -> None:
        if not self.available:
            return
        self.stop()
        cmd = [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-volume", str(self.volume),
            "-ss", str(max(start_seconds, 0.0)),
            self.source,
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self._offset = start_seconds
        self._start_time = time.monotonic()
        self._paused_at = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def set_volume(self, volume: int, resume_at: float) -> None:
        self.volume = max(0, min(100, volume))
        # ffplay has no live volume IPC via this simple invocation; the
        # practical approach is a quick restart at the same position.
        if self.proc:
            self.start(resume_at)

    def elapsed(self) -> float:
        """Seconds of audio played since (re)start, ignoring pauses."""
        if self._paused_at is not None:
            return (self._paused_at - self._start_time) + self._offset
        return (time.monotonic() - self._start_time) + self._offset

    def pause(self) -> None:
        # ffplay in this simple mode can't be paused in place without a
        # control channel, so we stop it; resume() restarts from position.
        if self._paused_at is None:
            self._paused_at = time.monotonic()
        self.stop()

    def resume(self) -> None:
        pos = self.elapsed()
        self.start(pos)
