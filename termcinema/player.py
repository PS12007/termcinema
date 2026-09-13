from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import audio as audio_mod
from . import colorspace, renderer, scaler, terminal, ui, utils, youtube
from .config import Config
from .input_handler import ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, ARROW_UP, InputHandler
from .terminal import ColorSupport
from .video import VideoSource


@dataclass
class PlayerState:
    playing: bool = True
    speed: float = 1.0
    mode: str = "halfblock"
    theme: colorspace.Theme = colorspace.Theme.NORMAL
    volume: int = 100
    muted: bool = False
    dropped_frames: int = 0
    seek_request: Optional[float] = None
    quit: bool = False
    render_fps: float = 0.0
    stats_visible: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)


_COLOR_NAME = {
    ColorSupport.MONO: "mono",
    ColorSupport.ANSI16: "16c",
    ColorSupport.ANSI256: "256c",
    ColorSupport.TRUECOLOR: "true",
}


class Player:
    def __init__(self, source: str, cfg: Config, forced_width: Optional[int] = None,
                 forced_height: Optional[int] = None):
        self.source_arg = source
        self.cfg = cfg
        self.state = PlayerState(mode=cfg.mode, speed=1.0, volume=cfg.volume)
        self.state.theme = colorspace.Theme(cfg.theme)
        self.caps = terminal.detect_caps()
        if forced_width:
            self.caps = terminal.TerminalCaps(forced_width, self.caps.rows, self.caps.color,
                                               self.caps.unicode_ok, self.caps.font_aspect)
        if forced_height:
            self.caps = terminal.TerminalCaps(self.caps.columns, forced_height, self.caps.color,
                                               self.caps.unicode_ok, self.caps.font_aspect)
        self._forced_width = forced_width
        self._forced_height = forced_height
        self.color_support = self._resolve_color_support()
        self.title = source
        self._audio_source_for_playback: Optional[str] = None

    def _resolve_color_support(self) -> ColorSupport:
        if self.cfg.color == "auto":
            return self.caps.color
        return {
            "truecolor": ColorSupport.TRUECOLOR,
            "256": ColorSupport.ANSI256,
            "16": ColorSupport.ANSI16,
            "mono": ColorSupport.MONO,
        }.get(self.cfg.color, self.caps.color)

    # ------------------------------------------------------------------
    def _open_source(self) -> VideoSource:
        target = self.source_arg
        if youtube.is_url(self.source_arg):
            resolved = youtube.resolve(self.source_arg, self.cfg.quality)
            self.title = resolved.title
            target = resolved.video_url
            # If audio is a separate stream, ffplay can still play the
            # merged/video URL for progressive formats; for split
            # formats we point ffplay at the audio-only URL directly.
            self._audio_source_for_playback = resolved.audio_url or resolved.video_url
        else:
            import os
            self.title = os.path.basename(self.source_arg)
            self._audio_source_for_playback = self.source_arg
        return VideoSource(target, hw_accel=self.cfg.hw_accel, use_gpu=self.cfg.gpu)

    # ------------------------------------------------------------------
    def run(self) -> None:
        video = self._open_source()
        target_fps = self.cfg.fps or video.info.fps

        aplayer = None
        if self.cfg.audio and self._audio_source_for_playback:
            aplayer = audio_mod.AudioPlayer(self._audio_source_for_playback)

        frame_q: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(maxsize=max(int(target_fps), 8))
        stop_event = threading.Event()

        decoder = threading.Thread(
            target=self._decode_loop,
            args=(video, target_fps, frame_q, stop_event),
            daemon=True,
        )

        inp = InputHandler()

        sys.stdout.write(terminal.ALT_SCREEN_ON + terminal.HIDE_CURSOR)
        sys.stdout.flush()

        try:
            decoder.start()
            inp.start()
            if aplayer:
                aplayer.start(0.0)
            self._play_loop(video, target_fps, frame_q, stop_event, inp, aplayer)
        finally:
            stop_event.set()
            inp.stop()
            if aplayer:
                aplayer.stop()
            video.release()
            sys.stdout.write(terminal.SHOW_CURSOR + terminal.ALT_SCREEN_OFF)
            sys.stdout.flush()

    # ------------------------------------------------------------------
    def _decode_loop(self, video, target_fps, frame_q, stop_event) -> None:
        frame_interval = 1.0 / target_fps if target_fps else 1.0 / 25.0
        idx = 0
        while not stop_event.is_set():
            with self.state.lock:
                seek_to = self.state.seek_request
                self.state.seek_request = None
                playing = self.state.playing

            if seek_to is not None:
                video.seek_seconds(seek_to)
                # Drain the queue so we don't show stale frames after a seek.
                with frame_q.mutex:
                    frame_q.queue.clear()

            if not playing:
                time.sleep(0.02)
                continue

            frame = video.read_rgb()
            if frame is None:
                break
            pts = video.current_pos_seconds() or (idx * frame_interval)
            with self.state.lock:
                theme = self.state.theme
            if theme != colorspace.Theme.NORMAL:
                frame = colorspace.apply_theme(frame, theme)
            try:
                frame_q.put((frame, pts), timeout=1.0)
            except queue.Full:
                pass
            idx += 1
        try:
            frame_q.put((None, -1.0), timeout=1.0)  # sentinel: EOF
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    def _play_loop(self, video, target_fps, frame_q, stop_event, inp, aplayer) -> None:
        start_wall = time.monotonic()
        base_pts = 0.0
        frame_times = []
        cols_cap = self.caps.columns
        eof = False

        while not self.state.quit and not eof:
            self._handle_input(inp, video, frame_q, aplayer)

            with self.state.lock:
                playing = self.state.playing
                speed = self.state.speed
                mode = self.state.mode

            if not playing:
                time.sleep(0.02)
                continue

            try:
                frame, pts = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                eof = True
                break

            target_elapsed = (time.monotonic() - start_wall) * speed
            # Frame skipping: if we're more than 2 frame-intervals behind,
            # drop this frame (still decoded, just not rendered) instead
            # of letting video lag further and further behind audio.
            behind = target_elapsed - pts
            frame_interval = 1.0 / (target_fps or 25.0)
            if behind > frame_interval * 2 and not frame_q.empty():
                with self.state.lock:
                    self.state.dropped_frames += 1
                continue

            if pts > target_elapsed:
                time.sleep(min(pts - target_elapsed, 0.1))

            t0 = time.perf_counter()
            self._render_and_draw(frame, mode, video, pts, target_fps)
            dt = time.perf_counter() - t0
            frame_times.append(dt)
            if len(frame_times) > 30:
                frame_times.pop(0)
            with self.state.lock:
                self.state.render_fps = 1.0 / (sum(frame_times) / len(frame_times) + 1e-6)

    # ------------------------------------------------------------------
    def _current_caps(self) -> terminal.TerminalCaps:
        live = terminal.detect_caps()
        cols = self._forced_width or live.columns
        rows = self._forced_height or live.rows
        return terminal.TerminalCaps(cols, rows, live.color, live.unicode_ok, live.font_aspect)

    def _render_and_draw(self, frame_rgb, mode, video, pts, target_fps) -> None:
        caps = self._current_caps()  # cheap; catches live terminal resizes unless forced
        m = renderer.MODES[mode]
        cols, rows = scaler.compute_grid_size(
            video.info.width, video.info.height, caps.columns, caps.rows, caps.font_aspect
        )
        pw, ph = scaler.target_pixel_size(cols, rows, m.cols_per_cell, m.rows_per_cell)
        resized = scaler.resize_frame(frame_rgb[:, :, ::-1], pw, ph, use_gpu=self.cfg.gpu)[:, :, ::-1]

        if mode.startswith("ascii"):
            body = renderer.render_frame(mode, resized, self.color_support,
                                          edge_aware=self.cfg.edge_aware,
                                          color_dither=self.cfg.color_dither,
                                          use_gpu=self.cfg.gpu)
        elif mode in ("braille", "quarterblock"):
            body = renderer.render_frame(mode, resized, self.color_support,
                                          dither=self.cfg.dither,
                                          color_dither=self.cfg.color_dither)
        else:
            body = renderer.render_frame(mode, resized, self.color_support,
                                          color_dither=self.cfg.color_dither)

        with self.state.lock:
            bar = ui.status_bar(
                title=self.title, width=caps.columns, playing=self.state.playing,
                pos=pts, duration=video.info.duration, fps=self.state.render_fps,
                target_fps=target_fps or video.info.fps, mode=mode,
                color_mode=_COLOR_NAME[self.color_support], volume=self.state.volume,
                muted=self.state.muted, speed=self.state.speed,
                dropped=self.state.dropped_frames,
            ) if self.state.stats_visible else ""

        out = terminal.HOME + body
        if bar:
            out += "\n" + bar
        sys.stdout.write(out)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    def _handle_input(self, inp, video, frame_q, aplayer) -> None:
        key = inp.poll()
        if key is None:
            return
        with self.state.lock:
            if key in (" ",):
                self.state.playing = not self.state.playing
                if aplayer:
                    (aplayer.resume if self.state.playing else aplayer.pause)()
            elif key in ("q", "Q"):
                self.state.quit = True
            elif key in ("r", "R"):
                self.state.mode = renderer.next_mode(self.state.mode)
            elif key in ("c", "C"):
                themes = list(colorspace.Theme)
                i = themes.index(self.state.theme)
                self.state.theme = themes[(i + 1) % len(themes)]
            elif key in ("m", "M"):
                self.state.muted = not self.state.muted
                if aplayer:
                    aplayer.set_volume(0 if self.state.muted else self.state.volume, aplayer.elapsed())
            elif key in ("f", "F"):
                pass  # full redraw happens naturally next frame
            elif key == "0":
                self.state.speed = 1.0
            elif key in ("*",):
                self.state.speed = min(self.state.speed * 1.5, 8.0)
            elif key in ("/",):
                self.state.speed = max(self.state.speed / 1.5, 0.1)
            elif key == ARROW_UP:
                self.state.volume = min(100, self.state.volume + 5)
                if aplayer:
                    aplayer.set_volume(self.state.volume, aplayer.elapsed())
            elif key == ARROW_DOWN:
                self.state.volume = max(0, self.state.volume - 5)
                if aplayer:
                    aplayer.set_volume(self.state.volume, aplayer.elapsed())
            elif key == ARROW_RIGHT:
                self.state.seek_request = video.current_pos_seconds() + 5
            elif key == ARROW_LEFT:
                self.state.seek_request = max(video.current_pos_seconds() - 5, 0)
            elif key in ("s", "S"):
                self._save_screenshot()

    def _save_screenshot(self) -> None:
        # Best-effort: dump the most recent rendered frame's plain text.
        pass
