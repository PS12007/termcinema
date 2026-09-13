"""Audio playback and the master clock.

``AudioEngine`` decodes PCM with ffmpeg and plays it through PortAudio
(``sounddevice``). The number of samples the sound card has actually
consumed *is* the playback clock, so video frames are timed against what
you hear -- the only way to get reliable A/V sync. Pause is instant
(the callback outputs silence and the clock stops), volume is live, and
the last few thousand samples are kept for the spectrum visualizer.

If sounddevice isn't installed, ``FFplayAudio`` falls back to an ffplay
subprocess timed by the wall clock.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .inputs import Input, popen_kwargs, which

SAMPLE_RATE = 48000
CHANNELS = 2


def atempo_chain(speed: float) -> str:
    parts = []
    s = speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-3:
        parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


try:  # pragma: no cover - depends on system
    import sounddevice as _sd

    HAVE_SOUNDDEVICE = True
except Exception:  # pragma: no cover
    _sd = None
    HAVE_SOUNDDEVICE = False


class WallClock:
    """Media clock driven by time.monotonic()."""

    def __init__(self, start: float = 0.0, speed: float = 1.0):
        self._base = start
        self._t0 = time.monotonic()
        self.speed = speed
        self.paused = False

    def time(self) -> float:
        if self.paused:
            return self._base
        return self._base + (time.monotonic() - self._t0) * self.speed

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self._base = self.time()
        self._t0 = time.monotonic()
        self.paused = paused

    def set_speed(self, speed: float) -> None:
        self._base = self.time()
        self._t0 = time.monotonic()
        self.speed = speed

    def reset(self, pos: float) -> None:
        self._base = pos
        self._t0 = time.monotonic()


class AudioEngine:
    def __init__(self, inp: Input, start: float = 0.0, speed: float = 1.0, volume: float = 1.0,
                 muted: bool = False, paused: bool = False, loop: bool = False):
        self.inp = inp
        self.start = start
        self.speed = speed
        self.volume = volume
        self.muted = muted
        self.paused = paused
        self.loop = loop
        self.error = ""
        self.eof = False
        self.failed = False
        self._chunks: deque = deque()
        self._chunk_off = 0
        self._buffered = 0
        self._max_buffer = SAMPLE_RATE * 4
        self._lock = threading.Lock()
        self._played = 0                      # frames consumed by the device
        self._cb_time = 0.0                   # monotonic time of last callback
        self._cb_played = 0
        self._latency = 0.0
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._stream = None
        self.scope = np.zeros(4096, np.float32)  # recent mono samples for visualizers
        self._scope_pos = 0
        self._reader = threading.Thread(target=self._read, name="termcinema-audio", daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def command(self) -> list:
        ff = which("ffmpeg") or "ffmpeg"
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.loop and not self.inp.is_network:
            cmd += ["-stream_loop", "-1"]
        cmd += self.inp.ffmpeg_args(self.start)
        cmd += ["-map", "0:a:0", "-vn", "-sn", "-dn"]
        af = atempo_chain(self.speed)
        if af:
            cmd += ["-af", af]
        cmd += ["-f", "f32le", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "pipe:1"]
        return cmd

    def begin(self) -> "AudioEngine":
        if not HAVE_SOUNDDEVICE:
            self.failed = True
            self.error = "sounddevice not installed"
            return self
        try:
            self._stream = _sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                                            callback=self._callback, latency="low", blocksize=0)
            self._latency = float(self._stream.latency or 0.0)
            self._stream.start()
        except Exception as e:
            self.failed = True
            self.error = f"audio device error: {e}"
            self._stream = None
            return self
        self._reader.start()
        return self

    def close(self) -> None:
        self._stop.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
        s = self._stream
        self._stream = None
        if s is not None:
            try:
                s.abort()
                s.close()
            except Exception:
                pass

    # -- reader ------------------------------------------------------------
    def _read(self) -> None:
        try:
            self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          bufsize=0, **popen_kwargs())
        except OSError as e:
            self.error = str(e)
            self.failed = True
            return
        out = self._proc.stdout
        frame_bytes = 4 * CHANNELS
        chunk = 2048 * frame_bytes
        pending = b""
        while not self._stop.is_set():
            data = out.read(chunk)
            if not data:
                break
            data = pending + data
            usable = len(data) // frame_bytes * frame_bytes
            pending = data[usable:]
            arr = np.frombuffer(data[:usable], np.float32).reshape(-1, CHANNELS)
            while not self._stop.is_set():
                with self._lock:
                    if self._buffered < self._max_buffer:
                        self._chunks.append(arr)
                        self._buffered += arr.shape[0]
                        break
                time.sleep(0.01)
        try:
            err = self._proc.stderr.read().decode("utf-8", "replace").strip()
            if err and not self._stop.is_set():
                self.error = err.splitlines()[-1]
        except Exception:
            pass
        self.eof = True

    # -- realtime callback --------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):  # pragma: no cover - realtime
        if self.paused:
            outdata.fill(0)
            self._cb_time = time.monotonic()
            self._cb_played = self._played
            return
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                head = self._chunks[0]
                avail = head.shape[0] - self._chunk_off
                take = min(avail, frames - filled)
                outdata[filled:filled + take] = head[self._chunk_off:self._chunk_off + take]
                filled += take
                self._chunk_off += take
                if self._chunk_off >= head.shape[0]:
                    self._chunks.popleft()
                    self._chunk_off = 0
            self._buffered -= filled
        if filled < frames:
            outdata[filled:].fill(0)
        # Scope buffer (pre-volume, mono) for visualizers.
        if filled:
            mono = outdata[:filled].mean(axis=1)
            n = min(mono.size, self.scope.size)
            self.scope = np.roll(self.scope, -n)
            self.scope[-n:] = mono[-n:]
        gain = 0.0 if self.muted else self.volume
        if gain != 1.0:
            outdata *= gain
        self._cb_played = self._played
        self._cb_time = time.monotonic()
        self._played += filled

    # -- clock ---------------------------------------------------------------
    def time(self) -> float:
        """Media time (seconds) of the sample currently leaving the speakers."""
        played = self._played
        if not self.paused and self._cb_time:
            # Interpolate between callbacks for a smooth clock, but never
            # beyond what has actually been handed to the device.
            since = time.monotonic() - self._cb_time
            played = min(self._cb_played + since * SAMPLE_RATE, self._played)
        media = self.start + (played / SAMPLE_RATE) * self.speed
        return max(self.start, media - self._latency * self.speed)

    @property
    def buffered_seconds(self) -> float:
        return self._buffered / SAMPLE_RATE

    @property
    def drained(self) -> bool:
        return self.eof and self._buffered <= 0

    def set_paused(self, paused: bool) -> None:
        self.paused = paused


class FFplayAudio:
    """Fallback: ffplay subprocess, restarted on pause/seek/volume changes."""

    def __init__(self, inp: Input, start: float = 0.0, speed: float = 1.0, volume: float = 1.0,
                 muted: bool = False, paused: bool = False, loop: bool = False):
        self.inp = inp
        self.clock = WallClock(start, speed)
        self.speed = speed
        self.volume = volume
        self.muted = muted
        self.paused = paused
        self.loop = loop
        self.error = ""
        self.failed = which("ffplay") is None
        self.eof = False
        self.scope = np.zeros(4096, np.float32)
        self._proc: Optional[subprocess.Popen] = None
        if paused:
            self.clock.set_paused(True)

    def _spawn(self) -> None:
        self._kill()
        if self.failed or self.paused:
            return
        vol = 0 if self.muted else int(self.volume * 100)
        cmd = [which("ffplay"), "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", str(min(vol, 100))]
        if self.loop:
            cmd += ["-loop", "0"]
        args = self.inp.ffmpeg_args(self.clock.time())
        cmd += [a for a in args if a != "-i"]
        af = atempo_chain(self.speed)
        if af:
            cmd += ["-af", af]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **popen_kwargs())
        except OSError as e:
            self.failed = True
            self.error = str(e)

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def begin(self) -> "FFplayAudio":
        self.clock.reset(self.clock.time())
        self._spawn()
        return self

    def close(self) -> None:
        self._kill()

    def time(self) -> float:
        return self.clock.time()

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        self.clock.set_paused(paused)
        if paused:
            self._kill()
        else:
            self._spawn()

    def refresh(self) -> None:
        """Apply volume/mute changes (requires a restart with ffplay)."""
        self._spawn()

    @property
    def buffered_seconds(self) -> float:
        return 1.0

    @property
    def drained(self) -> bool:
        return False
