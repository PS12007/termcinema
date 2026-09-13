"""ffmpeg-based video decoding into raw RGB frames.

ffmpeg does the heavy lifting -- demuxing anything, hardware decoding,
streaming over HTTP/HLS, and high-quality area-averaged downscaling --
and pipes constant-frame-rate rgb24 frames to a reader thread. Because
the output is CFR, frame N's presentation time is simply
``start + N / fps``.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .inputs import Input, popen_kwargs, which


@dataclass
class Frame:
    pts: float
    image: Optional[np.ndarray]  # None marks end of stream
    index: int = 0


class VideoDecoder:
    def __init__(self, inp: Input, width: int, height: int, fps: float, start: float = 0.0,
                 hwaccel: bool = False, queue_size: int = 12, loop: bool = False):
        self.inp = inp
        self.width = max(2, int(width) // 2 * 2)
        self.height = max(2, int(height) // 2 * 2)
        self.fps = fps if fps and fps > 0 else 30.0
        self.start = start
        self.hwaccel = hwaccel
        self.loop = loop
        self.frames: "queue.Queue[Frame]" = queue.Queue(maxsize=queue_size)
        self.error: str = ""
        self.eof = False
        self.started_at = time.monotonic()
        self.first_frame_at: Optional[float] = None
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread = threading.Thread(target=self._run, name="termcinema-video", daemon=True)
        self._err_thread: Optional[threading.Thread] = None

    def command(self) -> list:
        ff = which("ffmpeg") or "ffmpeg"
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.hwaccel:
            cmd += ["-hwaccel", "auto"]
        if self.loop and not self.inp.is_network:
            cmd += ["-stream_loop", "-1"]
        cmd += self.inp.ffmpeg_args(self.start)
        vf = (f"fps={self.fps:.6f}:round=near,"
              f"scale={self.width}:{self.height}:flags=area,format=rgb24")
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        return cmd

    def start_decoding(self) -> "VideoDecoder":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
        # Unblock a producer stuck on a full queue.
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

    def _put(self, fr: Frame) -> bool:
        while not self._stop.is_set():
            try:
                self.frames.put(fr, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _drain_stderr(self, pipe) -> None:
        chunks = []
        try:
            for line in iter(pipe.readline, b""):
                chunks.append(line.decode("utf-8", "replace"))
                if len(chunks) > 40:
                    chunks.pop(0)
        except Exception:
            pass
        self.error = "".join(chunks).strip()

    def _run(self) -> None:
        try:
            self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          bufsize=0, **popen_kwargs())
        except OSError as e:
            self.error = f"could not start ffmpeg: {e}"
            self._put(Frame(self.start, None))
            return
        self._err_thread = threading.Thread(target=self._drain_stderr, args=(self._proc.stderr,), daemon=True)
        self._err_thread.start()
        size = self.width * self.height * 3
        out = self._proc.stdout
        n = 0
        while not self._stop.is_set():
            buf = bytearray(size)
            view = memoryview(buf)
            got = 0
            while got < size:
                try:
                    r = out.readinto(view[got:])
                except Exception:
                    r = 0
                if not r:
                    break
                got += r
            if got < size:
                break
            img = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
            if self.first_frame_at is None:
                self.first_frame_at = time.monotonic()
            if not self._put(Frame(self.start + n / self.fps, img, n)):
                break
            n += 1
        self.eof = True
        if not self._stop.is_set():
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
            if self._err_thread:
                self._err_thread.join(timeout=1)
            self._put(Frame(self.start + n / self.fps, None, n))


def grab_frame(inp: Input, width: int, height: int, at: float = 0.0, timeout: float = 30.0) -> Optional[np.ndarray]:
    """Decode a single frame (for --snapshot, thumbnails and still images)."""
    ff = which("ffmpeg") or "ffmpeg"
    width, height = max(2, width // 2 * 2), max(2, height // 2 * 2)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin"] + inp.ffmpeg_args(at) + [
        "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={width}:{height}:flags=area,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        kw = {k: v for k, v in popen_kwargs().items() if k != "stdin"}
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, **kw)
    except Exception:
        return None
    data = res.stdout
    if len(data) < width * height * 3:
        return None
    return np.frombuffer(data[: width * height * 3], np.uint8).reshape(height, width, 3)
