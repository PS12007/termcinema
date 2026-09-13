"""The player: pipelines, clock, frame scheduling, drawing and controls."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import filters, render
from .config import Config, State
from .media import audio as audio_mod
from .media.inputs import Media
from .media.source import Item
from .media.subtitles import Track
from .media.video import VideoDecoder
from .render.encode import CONTINUATION, CellEncoder, CellFrame
from .term import control
from .term.caps import Caps, ColorDepth
from .term.keys import Key, Mouse
from .ui import layout as layout_mod
from .ui import osd
from .ui.canvas import Canvas, fmt_time
from .ui.visualizer import STYLES, Visualizer

EFFECT_NAMES = list(filters.EFFECTS)
DEPTHS = [ColorDepth.TRUECOLOR, ColorDepth.ANSI256, ColorDepth.ANSI16, ColorDepth.MONO]


@dataclass
class Stats:
    render_ms: float = 0.0
    encode_ms: float = 0.0
    write_ms: float = 0.0
    bytes_per_frame: float = 0.0
    fps: float = 0.0
    dropped: int = 0
    frames: int = 0
    _last_t: float = 0.0

    def ema(self, name: str, value: float, k: float = 0.15) -> None:
        setattr(self, name, getattr(self, name) * (1 - k) + value * k)

    def tick(self) -> None:
        now = time.monotonic()
        if self._last_t:
            dt = now - self._last_t
            if dt > 0:
                self.ema("fps", 1.0 / dt, 0.08)
        self._last_t = now
        self.frames += 1


class Player:
    def __init__(self, items: list, cfg: Config, caps: Caps, out: control.Output, reader,
                 state: State, start_at: Optional[float] = None, audio_enabled: bool = True,
                 cell_aspect: Optional[float] = None, forced_size: tuple = (None, None)):
        self.items = items
        self.index = 0
        self.cfg = cfg
        self.caps = caps
        self.out = out
        self.reader = reader
        self.state = state
        self.start_at = start_at
        self.audio_enabled = audio_enabled and cfg.audio
        self.cell_aspect = cell_aspect or caps.cell_aspect
        self.forced_size = forced_size

        self.modes = render.available(caps)
        mode = cfg.mode if cfg.mode != "auto" else render.auto_mode(caps)
        if mode not in render.ALL_MODES:
            mode = "blocks"
        if mode not in self.modes:
            self.modes.insert(0, mode)
        self.mode = mode
        depth = ColorDepth.parse(cfg.color) if cfg.color != "auto" else caps.color
        self.depth = depth if depth is not None else caps.color
        self.effect = cfg.effect if cfg.effect in filters.EFFECTS else "none"
        self.adjust = filters.Adjust(cfg.brightness, cfg.contrast, cfg.saturation, cfg.gamma, cfg.sharpen)
        self.fit = cfg.fit if cfg.fit in layout_mod.FIT_MODES else "fit"
        self.zoom = 1.0
        self.speed = cfg.speed
        self.volume = max(0.0, min(cfg.volume / 100.0, 2.0))
        self.muted = False
        self.paused = False
        self.loop = False
        self.dither = cfg.dither
        self.osd_mode = cfg.osd if cfg.osd in ("auto", "always", "off") else "auto"
        self.subs_on = cfg.subs
        self.base_tolerance = max(0, cfg.tolerance)
        self.tolerance = self.base_tolerance

        self.encoder = CellEncoder()
        self.visualizer = Visualizer()
        if cfg.visualizer in STYLES:
            self.visualizer.style = cfg.visualizer
        self.stats = Stats()

        self.media: Optional[Media] = None
        self.track: Optional[Track] = None
        self.decoder: Optional[VideoDecoder] = None
        self.audio = None
        self.clock = audio_mod.WallClock()
        self.current = None            # last displayed Frame
        self.pending = None
        self.last_image: Optional[np.ndarray] = None
        self.last_pts = 0.0
        self.show_first = True
        self.prerolling = False
        self.preroll_started = 0.0
        self.video_eof = False
        self.stepped = False
        self.seek_target: Optional[float] = None
        self.seek_due = 0.0
        self.action: Optional[str] = None
        self.quit = False

        self.show_help = False
        self.show_stats = False
        self.menu_open = False
        self.menu_sel = 0
        self.toast_text = ""
        self.toast_until = 0.0
        self.last_activity = time.monotonic()
        self.media_started = time.monotonic()
        self.bar_geometry = (-1, 0, 0)
        self.term_size = self._term_size()
        self.last_size_check = 0.0
        self.need_clear = True
        self.ui_signature = None
        self.force_image = True
        self.last_draw = 0.0
        self.bytes_window = []
        self._overlay_was_on_image = False
        self.last_screen: Optional[CellFrame] = None
        self.last_px: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # helpers used by the OSD
    # ------------------------------------------------------------------
    def position(self) -> float:
        if self.seek_target is not None:
            return self.seek_target
        if self.media and self.media.is_image:
            return 0.0
        if self.current is not None and (self.paused or self.prerolling):
            return self.current.pts
        return max(0.0, self.clock.time())

    def buffered_seconds(self) -> float:
        a = self.audio
        if a is not None and hasattr(a, "buffered_seconds"):
            return min(a.buffered_seconds, 30.0)
        d = self.decoder
        if d is not None:
            return d.frames.qsize() / max(d.fps, 1)
        return 0.0

    def has_audio(self) -> bool:
        return self.audio is not None and not getattr(self.audio, "failed", False)

    def current_chapter(self) -> str:
        m = self.media
        if not m or not m.chapters:
            return ""
        pos = self.position()
        name = ""
        for start, title in m.chapters:
            if start <= pos:
                name = title
        return name

    def toast(self, text: str, secs: float = 1.6) -> None:
        self.toast_text = text
        self.toast_until = time.monotonic() + secs

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def run(self) -> int:
        try:
            while not self.quit and 0 <= self.index < len(self.items):
                item = self.items[self.index]
                media = self._resolve(item)
                if media is None:
                    if self.quit:
                        break
                    self.index += 1
                    continue
                result = self._play(media, item)
                self._save_resume()
                if result == "quit":
                    break
                if result == "prev":
                    self.index = max(0, self.index - 1)
                elif result == "restart":
                    continue
                else:
                    self.index += 1
                    if self.index >= len(self.items) and self.loop and len(self.items) > 1:
                        self.index = 0
        finally:
            self._stop_pipeline()
            self._clear_images()
            self.state.save()
        return 0

    # ------------------------------------------------------------------
    # resolving (with a spinner, cancellable)
    # ------------------------------------------------------------------
    def _resolve(self, item: Item) -> Optional[Media]:
        result: dict = {}

        def work():
            try:
                result["media"] = item.resolver()
            except Exception as e:  # noqa: BLE001 - shown to the user
                result["error"] = str(e)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        started = time.monotonic()
        while t.is_alive():
            for ev in self.reader.drain():
                if isinstance(ev, Key) and ev.name in ("q", "esc", "ctrl+c"):
                    self.quit = True
                    return None
                if isinstance(ev, Key) and ev.name == "n":
                    return None
            if time.monotonic() - started > 0.15:
                self._draw_message(item.label, "Loading" + "." * (int(time.monotonic() * 3) % 4), spin=True)
            time.sleep(1 / 30)
        if "error" in result:
            self._draw_message(item.label, "Error: " + result["error"].splitlines()[0], spin=False, error=True)
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                ev = self.reader.get(timeout=0.05)
                if isinstance(ev, Key):
                    if ev.name in ("q", "esc", "ctrl+c"):
                        self.quit = True
                    break
            self.errors = getattr(self, "errors", []) + [(item.label, result["error"])]
            return None
        return result.get("media")

    def _draw_message(self, title: str, subtitle: str, spin: bool = False, error: bool = False) -> None:
        cols, rows = self._term_size()
        screen = self._blank_screen(rows, cols)
        cv = Canvas(rows, cols)
        osd.title_card(cv, title, subtitle)
        if spin:
            osd.spinner(cv, "")
        if error:
            cv.text(rows // 2 + 3, 2, "", osd.WARN)
        cv.compose(screen)
        if self.need_clear:
            self._clear_screen()
        data = self.encoder.encode(screen, 0, 0, self._text_depth())
        self.out.write(control.SYNC_BEGIN.encode() + data + control.SYNC_END.encode())

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _play(self, media: Media, item: Item) -> str:
        if media.loop and len(self.items) > 1:
            media.loop = False  # animated GIFs loop forever only when played on their own
        self.media = media
        self.track = None
        self.action = None
        self.current = None
        self.last_image = None
        self.video_eof = False
        self.zoom = 1.0
        self.media_started = time.monotonic()
        self.last_activity = time.monotonic()
        self.need_clear = True
        self.state.add_history(media.title, item.arg or media.webpage_url or media.title)
        if media.subtitle_loader is not None:
            threading.Thread(target=self._load_subs, args=(media,), daemon=True).start()
        if self.media.thumbnail and self.media.video is None:
            threading.Thread(target=self._load_thumbnail, args=(media,), daemon=True).start()
        else:
            self.visualizer.background = None

        start = 0.0
        if self.start_at is not None:
            start = self.start_at
            self.start_at = None
        elif self.cfg.resume and media.resume_key and media.seekable:
            pos = self.state.resume_position(media.resume_key)
            if pos > 0:
                start = pos
                self.toast(f"Resumed at {fmt_time(pos)}  ·  Home to restart", 4.0)
        if media.duration and start >= media.duration - 1:
            start = 0.0
        self._start_pipeline(start)
        try:
            return self._loop()
        finally:
            self._stop_pipeline()

    def _load_subs(self, media: Media) -> None:
        try:
            cues = media.subtitle_loader()
        except Exception:
            cues = []
        if cues and media is self.media:
            self.track = Track(cues)
            if self.subs_on:
                self.toast(f"Subtitles loaded ({len(cues)} lines)  ·  t to toggle")

    def _load_thumbnail(self, media: Media) -> None:
        from .media import ytdl

        try:
            data = ytdl.fetch_bytes(media.thumbnail)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None and media is self.media:
                img = cv2.GaussianBlur(img, (0, 0), 6)
                self.visualizer.background = img[:, :, ::-1]
        except Exception:
            pass

    def _hwaccel(self) -> bool:
        if self.cfg.hwaccel == "on":
            return True
        if self.cfg.hwaccel == "off":
            return False
        m = self.media
        return bool(m and m.video and not m.video.pre_args and m.height >= 1000)

    def _start_pipeline(self, pos: float, keep_audio: bool = False) -> None:
        m = self.media
        self._stop_pipeline(keep_audio=keep_audio)
        self.pending = None
        self.video_eof = False
        self.show_first = True
        self.stepped = False
        if m.video is not None:
            w, h = self._decode_size()
            fps = min(m.fps or 30.0, self.cfg.max_fps)
            self.decoder = VideoDecoder(m.video, w, h, fps, start=pos, hwaccel=self._hwaccel(),
                                        loop=m.loop).start_decoding()
        if not keep_audio:
            self.audio = None
            if m.audio is not None and self.audio_enabled and not m.is_image:
                cls = audio_mod.AudioEngine if audio_mod.HAVE_SOUNDDEVICE else audio_mod.FFplayAudio
                eng = cls(m.audio, start=pos, speed=self.speed, volume=self.volume, muted=self.muted,
                          paused=True, loop=m.loop)
                eng.begin()
                if eng.failed:
                    if cls is audio_mod.AudioEngine and audio_mod.which("ffplay"):
                        eng = audio_mod.FFplayAudio(m.audio, start=pos, speed=self.speed, volume=self.volume,
                                                    muted=self.muted, paused=True, loop=m.loop).begin()
                    if eng.failed:
                        self.toast(f"No audio: {eng.error[:60]}", 3)
                        eng = None
                self.audio = eng
        if self.audio is not None:
            self.clock = self.audio
        else:
            self.clock = audio_mod.WallClock(pos, self.speed)
            self.clock.set_paused(True)
        self.prerolling = m.video is not None
        self.preroll_started = time.monotonic()
        if not self.prerolling:
            self._set_clock_paused(self.paused)

    def _stop_pipeline(self, keep_audio: bool = False) -> None:
        if self.decoder is not None:
            self.decoder.stop()
            self.decoder = None
        if not keep_audio and self.audio is not None:
            self.audio.close()
            self.audio = None

    def _set_clock_paused(self, paused: bool) -> None:
        c = self.clock
        if c is not None:
            c.set_paused(paused)

    def _decode_size(self) -> tuple:
        m = self.media
        cols, rows = self.term_size
        r = render.get(self.mode)
        L = self._layout(r)
        cx, cy, cw, ch = L.crop
        if r.kind == "cells":
            need_w = L.w * 2 / max(cw, 1e-3)
        else:
            pw, _ = r.pixel_size(L.w, L.h, self.caps.cell_w, self.caps.cell_h)
            need_w = pw / max(cw, 1e-3)
        need_w = int(need_w * 1.0)
        if m.width:
            need_w = min(need_w, m.width)
        need_w = max(64, min(need_w, 1920))
        aspect = m.aspect
        need_h = max(36, int(round(need_w / aspect)))
        return need_w, need_h

    def _maybe_restart_decoder(self) -> None:
        d = self.decoder
        if d is None or self.media.is_image:
            return
        want_w, want_h = self._decode_size()
        if want_w > d.width * 1.2 or want_w < d.width * 0.55:
            pos = self.position()
            if self.media.seekable:
                self._restart_video(pos)
            elif self.media.is_live:
                self._restart_video(0.0)

    def _restart_video(self, pos: float) -> None:
        m = self.media
        if self.decoder is not None:
            self.decoder.stop()
        w, h = self._decode_size()
        fps = min(m.fps or 30.0, self.cfg.max_fps)
        self.decoder = VideoDecoder(m.video, w, h, fps, start=pos if m.seekable else 0.0,
                                    hwaccel=self._hwaccel(), loop=m.loop).start_decoding()
        self.pending = None
        self.video_eof = False

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _loop(self) -> str:
        m = self.media
        while True:
            for ev in self.reader.drain():
                self._handle(ev)
            if self.quit:
                return "quit"
            if self.action:
                return self.action

            self._check_resize()
            self._maybe_seek()

            new_frame = self._pump_video()
            if self.prerolling:
                timeout = 12.0 if (m.video and m.video.is_network) else 6.0
                if new_frame is not None or time.monotonic() - self.preroll_started > timeout:
                    self.prerolling = False
                    if self.stepped is False:
                        self._set_clock_paused(self.paused)

            ended = self._check_end()
            if ended:
                return ended

            if m.video is None:
                # audio-only: visualizer drives the frame rate
                if time.monotonic() - self.last_draw >= 1 / 30:
                    self._draw(None)
            elif new_frame is not None:
                self.current = new_frame
                self.last_image = new_frame.image
                self.last_pts = new_frame.pts
                self.force_image = True
                self._draw(new_frame.image)
            elif self._ui_changed() or (time.monotonic() - self.last_draw > 0.5):
                self._draw(None)

            self._sleep_until_next()

    def _sleep_until_next(self) -> None:
        d = self.decoder
        wait = 1 / 60
        if d is not None and not self.paused and not self.prerolling:
            nxt = self.pending
            if nxt is not None and nxt.image is not None:
                wait = min(max(nxt.pts - self.clock.time(), 0.0) / max(self.speed, 0.01), 1 / 60)
            else:
                wait = 0.004
        if wait > 0.0005:
            ev = self.reader.get(timeout=wait)
            if ev is not None:
                self._handle(ev)

    def _pump_video(self):
        d = self.decoder
        if d is None:
            return None
        live_latest = self.media.is_live and self.audio is None
        shown = None
        while True:
            if self.pending is None:
                try:
                    self.pending = d.frames.get_nowait()
                except queue.Empty:
                    break
            fr = self.pending
            if fr.image is None:
                self.video_eof = True
                break
            if self.show_first:
                self.show_first = False
                self.pending = None
                shown = fr
                if self.paused or self.prerolling:
                    break
                continue
            if self.paused:
                break
            if live_latest:
                if shown is not None:
                    self.stats.dropped += 1
                shown = fr
                self.pending = None
                continue
            now = self.clock.time()
            interval = 1.0 / max(d.fps, 1.0)
            if fr.pts <= now + interval * 0.3:
                if shown is not None:
                    self.stats.dropped += 1
                shown = fr
                self.pending = None
                continue
            break
        if self.video_eof and self.media.is_image and shown is None and self.current is None:
            return None
        return shown

    def _check_end(self) -> Optional[str]:
        m = self.media
        if m.is_image:
            if self.video_eof and len(self.items) > 1 and not self.paused and                     time.monotonic() - self.media_started > 6.0:
                return "next"
            return None
        if self.seek_target is not None or self.prerolling:
            return None
        a = self.audio if (self.audio is not None and not getattr(self.audio, "failed", False)) else None
        engine = isinstance(a, audio_mod.AudioEngine)

        if engine and a.drained and m.video is not None and self.clock is a and not self.video_eof:
            # Audio ran out before the picture: keep going on the wall clock.
            wc = audio_mod.WallClock(a.time(), self.speed)
            wc.set_paused(self.paused)
            self.clock = wc

        video_done = m.video is None or (self.video_eof and self.pending is not None and self.pending.image is None)
        if a is None:
            audio_done = True
        elif engine:
            audio_done = a.drained
        else:  # ffplay: no way to know, go by the clock
            audio_done = m.duration > 0 and self.clock.time() >= m.duration - 0.05
        if m.video is not None and video_done and not audio_done and m.duration > 0:
            audio_done = self.clock.time() >= m.duration - 0.05

        if not (video_done and audio_done):
            return None
        failed_audio = self.audio if self.audio is not None else None
        nothing_played = (m.video is None and (a is None or (engine and a._played == 0)))
        if nothing_played:
            err = getattr(failed_audio, "error", "") or "audio could not be played"
            self.toast("Audio error: " + err.splitlines()[-1][:90], 4)
            self._draw(None)
            time.sleep(3)
            return "next"
        if self.decoder is not None and self.decoder.error and self.current is None:
            self.toast("ffmpeg: " + self.decoder.error.splitlines()[-1][:90], 5)
            self._draw(None)
            time.sleep(2.5)
            return "next"
        if self.loop and m.seekable and len(self.items) == 1:
            self._start_pipeline(0.0)
            return None
        return "next"

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _term_size(self) -> tuple:
        cols, rows = control.terminal_size()
        fw, fh = self.forced_size
        return (fw or cols, fh or rows)

    def _check_resize(self) -> None:
        now = time.monotonic()
        if now - self.last_size_check < 0.25:
            return
        self.last_size_check = now
        size = self._term_size()
        if size != self.term_size:
            self.term_size = size
            self.caps.cols, self.caps.rows = size
            self.need_clear = True
            self.force_image = True
            self._maybe_restart_decoder()

    def _reserve_rows(self, r) -> int:
        return 2 if r.kind == "pixels" else 0

    def _layout(self, r=None):
        r = r or render.get(self.mode)
        cols, rows = self.term_size
        aspect = self.media.aspect if self.media else 16 / 9
        if self.media is not None and self.media.video is None:
            aspect = cols / max(rows * self.cell_aspect, 1)  # visualizer fills the screen
        return layout_mod.compute(cols, rows, aspect, self.cell_aspect, self.fit, self.zoom,
                                  reserve=self._reserve_rows(r))

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _text_depth(self) -> ColorDepth:
        return self.depth

    def _blank_screen(self, rows: int, cols: int) -> CellFrame:
        return CellFrame(np.full((rows, cols), 0x20, np.int32), np.zeros((rows, cols, 3), np.uint8),
                         np.zeros((rows, cols, 3), np.uint8))

    def _clear_screen(self) -> None:
        self.out.write("\x1b[0m\x1b[2J")
        self.encoder.invalidate()
        self.need_clear = False
        self.force_image = True

    def _clear_images(self) -> None:
        for name in render.PIXEL_MODES:
            if name == self.mode:
                data = render.get(name).clear()
                if data:
                    self.out.write(data)

    def _prepare(self, img: np.ndarray, out_w: int, out_h: int, crop: tuple) -> np.ndarray:
        h, w = img.shape[:2]
        cx, cy, cw, ch = crop
        x0, y0 = int(cx * w), int(cy * h)
        x1, y1 = max(x0 + 1, int((cx + cw) * w)), max(y0 + 1, int((cy + ch) * h))
        sub = img[y0:y1, x0:x1]
        if self.media is not None and self.media.kind == "webcam":
            sub = sub[:, ::-1]
        interp = cv2.INTER_AREA if out_w < sub.shape[1] else cv2.INTER_CUBIC
        px = cv2.resize(sub, (max(out_w, 1), max(out_h, 1)), interpolation=interp)
        px = filters.apply_adjust(px, self.adjust)
        px = filters.apply_effect(px, self.effect, time.monotonic())
        return np.ascontiguousarray(px)

    def _osd_visible(self) -> bool:
        if self.osd_mode == "off":
            return False
        if self.osd_mode == "always" or self.paused or self.menu_open:
            return True
        now = time.monotonic()
        return now - self.last_activity < 3.0 or now - self.media_started < 4.0

    def _ui_changed(self) -> bool:
        sig = (self._osd_visible(), int(self.position()), self.paused, self.show_help, self.show_stats,
               self.menu_open, self.menu_sel, self.toast_text if time.monotonic() < self.toast_until else "",
               self.mode, self.depth, self.effect, self.fit, round(self.zoom, 2), self.muted,
               round(self.volume, 2), self.speed, self.term_size, self.subs_on, self.loop,
               self.prerolling and time.monotonic() - self.preroll_started > 0.4,
               self.adjust.describe(), self.dither, self.osd_mode, self.visualizer.style,
               self.track.at(self.position()) if (self.track and self.subs_on) else "")
        if self.show_stats or self.prerolling:
            sig = sig + (time.monotonic() // 0.25,)
        changed = sig != self.ui_signature
        self.ui_signature = sig
        return changed

    def _build_canvas(self, rows: int, cols: int, L) -> Canvas:
        cv = Canvas(rows, cols)
        now = time.monotonic()
        bar_top = rows
        self.bar_geometry = (-1, 0, 0)
        if self._osd_visible() and rows >= 4:
            bar_top = osd.bottom_bar(cv, self)
        if self.subs_on and self.track is not None:
            txt = self.track.at(self.position())
            if txt:
                bottom = bar_top if bar_top < rows else rows - 1
                if render.get(self.mode).kind == "pixels":
                    bottom = min(bottom, rows - 2)
                osd.subtitles(cv, txt, bottom)
        if self.prerolling and now - self.preroll_started > 0.4:
            osd.spinner(cv, "Buffering")
        if now < self.toast_until and self.toast_text:
            osd.toast(cv, self.toast_text)
        if self.show_stats:
            osd.stats_panel(cv, self._stats_lines(L))
        if self.show_help:
            osd.help_panel(cv)
        if self.menu_open:
            osd.menu_panel(cv, [(label, value) for label, value, _ in self._menu_items()], self.menu_sel)
        return cv

    def _draw(self, image: Optional[np.ndarray]) -> None:
        cols, rows = self.term_size
        if self.need_clear:
            self._clear_screen()
        r = render.get(self.mode)
        L = self._layout(r)
        t0 = time.perf_counter()
        media_time = self.position()

        if self.media is not None and self.media.video is None:
            # visualizer frame
            samples = getattr(self.audio, "scope", None)
            if samples is None:
                samples = np.zeros(2048, np.float32)
            if r.kind == "cells":
                vw, vh = L.w * r.gw, L.h * r.gh
            else:
                vw, vh = r.pixel_size(L.w, L.h, self.caps.cell_w, self.caps.cell_h)
            image = self.visualizer.frame(samples, max(vw, 16), max(vh, 16), time.monotonic())
            crop = (0.0, 0.0, 1.0, 1.0)
            L = layout_mod.Layout(L.cols, L.rows, L.x0, L.y0, L.w, L.h, crop)
            self.force_image = True

        img = image if image is not None else self.last_image
        data_parts = []
        screen = self._blank_screen(rows, cols)
        canvas = self._build_canvas(rows, cols, L)

        if r.kind == "cells":
            if img is not None:
                px = self._prepare(img, L.w * r.gw, L.h * r.gh, L.crop) if image is not None or \
                    self.last_px is None or self.last_px.shape[:2] != (L.h * r.gh, L.w * r.gw) or \
                    self.force_image else self.last_px
                self.last_px = px
                self.force_image = False
                cf = r.render(px)
                h, w = cf.cps.shape
                screen.cps[L.y0:L.y0 + h, L.x0:L.x0 + w] = cf.cps
                screen.fg[L.y0:L.y0 + h, L.x0:L.x0 + w] = cf.fg
                screen.bg[L.y0:L.y0 + h, L.x0:L.x0 + w] = cf.bg
            t1 = time.perf_counter()
            canvas.compose(screen)
            data = self.encoder.encode(screen, 0, 0, self.depth, tolerance=self.tolerance, dither=self.dither)
            t2 = time.perf_counter()
            data_parts.append(data)
        else:
            overlay_on_image = bool((canvas.alpha[L.y0:L.y0 + L.h, L.x0:L.x0 + L.w] > 0).any()
                                    or (canvas.cps[L.y0:L.y0 + L.h, L.x0:L.x0 + L.w] >= 0).any())
            if self._overlay_was_on_image and not overlay_on_image:
                self._clear_screen()
            self._overlay_was_on_image = overlay_on_image
            if img is not None and (image is not None or self.force_image):
                pw, ph = r.pixel_size(L.w, L.h, self.caps.cell_w, self.caps.cell_h)
                px = self._prepare(img, pw, ph, L.crop)
                self.last_px = px
                data_parts.append(r.render(px, L.y0, L.x0, L.w, L.h))
                self.force_image = False
            t1 = time.perf_counter()
            # Cells over the picture are left alone unless the OSD covers them.
            screen.cps[L.y0:L.y0 + L.h, L.x0:L.x0 + L.w] = CONTINUATION
            covered = (canvas.alpha > 0) | (canvas.cps >= 0)
            region = np.zeros_like(covered)
            region[L.y0:L.y0 + L.h, L.x0:L.x0 + L.w] = True
            screen.cps[region & covered] = 0x20
            canvas.compose(screen)
            data_parts.append(self.encoder.encode(screen, 0, 0, self.depth if self.depth != ColorDepth.MONO
                                                  else ColorDepth.ANSI16, tolerance=0))
            t2 = time.perf_counter()

        payload = b"".join(data_parts)
        tw0 = time.perf_counter()
        if payload:
            self.out.write(control.SYNC_BEGIN.encode() + payload + control.SYNC_END.encode())
        tw1 = time.perf_counter()
        self.last_screen = screen
        self.last_draw = time.monotonic()
        if image is not None:
            st = self.stats
            st.tick()
            st.ema("render_ms", (t1 - t0) * 1000)
            st.ema("encode_ms", (t2 - t1) * 1000)
            st.ema("write_ms", (tw1 - tw0) * 1000)
            st.ema("bytes_per_frame", len(payload))
            self._adapt(tw1 - tw0)

    def _adapt(self, write_s: float) -> None:
        """Raise the encoder's color tolerance when the terminal can't keep up."""
        d = self.decoder
        if d is None or render.get(self.mode).kind != "cells" or self.depth != ColorDepth.TRUECOLOR:
            return
        budget = 1.0 / max(min(d.fps, 60.0), 1.0)
        if self.stats.write_ms / 1000 > budget * 0.6 and self.tolerance < 24:
            self.tolerance += 1
        elif self.stats.write_ms / 1000 < budget * 0.2 and self.tolerance > self.base_tolerance:
            self.tolerance -= 1

    def _stats_lines(self, L) -> list:
        d = self.decoder
        r = render.get(self.mode)
        st = self.stats
        a = self.audio
        target = d.fps if d else 30
        grid = f"{L.w}x{L.h} cells"
        if r.kind == "cells":
            grid += f"  ->  {L.w * r.gw}x{L.h * r.gh} px"
        drift = ""
        if d is not None and self.current is not None and not self.paused and not (self.media.is_live and self.audio is None):
            drift = f"  drift {1000 * (self.current.pts - self.clock.time()):+.0f}ms"
        lines = [
            ("renderer", f"{self.mode}  ({r.gw}x{r.gh} px/cell)" if r.kind == "cells" else f"{self.mode}  (pixels)"),
            ("grid", grid),
            ("decode", f"{d.width}x{d.height} @ {d.fps:g}fps  hw {'on' if d.hwaccel else 'off'}" if d else "audio only"),
            ("source", f"{self.media.width}x{self.media.height}  {self.media.kind}" if self.media else ""),
            ("color", f"{self.depth.label}  tolerance {self.tolerance}  dither {'on' if self.dither else 'off'}"),
            ("fps", f"{st.fps:5.1f} / {target:g}   dropped {st.dropped}"),
            ("timing", f"render {st.render_ms:4.1f}ms  encode {st.encode_ms:4.1f}ms  write {st.write_ms:4.1f}ms"),
            ("output", f"{st.bytes_per_frame / 1024:6.1f} KB/frame  {st.bytes_per_frame * st.fps / 1048576:5.2f} MB/s"),
            ("clock", ("audio" if isinstance(self.clock, audio_mod.AudioEngine) else "wall") +
             (f"  buffer {a.buffered_seconds:.1f}s" if a is not None else "") + drift),
            ("terminal", f"{self.caps.terminal}  {self.term_size[0]}x{self.term_size[1]}  cell aspect {self.cell_aspect:.2f}"),
            ("graphics", "  ".join(f"{k}:{'yes' if getattr(self.caps, k) else 'no'}" for k in ("kitty", "iterm", "sixel"))),
        ]
        return lines

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def _handle(self, ev) -> None:
        if isinstance(ev, Mouse):
            self._handle_mouse(ev)
            return
        if not isinstance(ev, Key):
            return
        k = ev.name
        self.last_activity = time.monotonic()

        if self.menu_open:
            if self._menu_key(k):
                return
        if self.show_help and k not in ("?", "h", "f1"):
            self.show_help = False
            if k in ("esc", "q"):
                return

        if k in ("q", "ctrl+c"):
            self.quit = True
        elif k == "esc":
            if self.show_stats:
                self.show_stats = False
            else:
                self.quit = True
        elif k in ("space", "k"):
            self.toggle_pause()
        elif k in ("right", "l", "left", "j", "pgup", "pgdn", "shift+right", "shift+left"):
            delta = {"right": 5, "left": -5, "l": 10, "j": -10, "pgup": 60, "pgdn": -60,
                     "shift+right": 30, "shift+left": -30}[k]
            self.seek_relative(delta)
        elif k.isdigit() and len(k) == 1:
            if self.media and self.media.duration > 0:
                self.seek_to(self.media.duration * int(k) / 10)
        elif k == "home":
            self.seek_to(0.0)
        elif k == "end":
            if self.media and self.media.duration > 0:
                self.seek_to(max(self.media.duration - 5, 0))
        elif k == "up":
            self.set_volume(self.volume + 0.05)
        elif k == "down":
            self.set_volume(self.volume - 0.05)
        elif k == "m":
            self.toggle_mute()
        elif k == "]":
            self.set_speed(self._step_speed(+1))
        elif k == "[":
            self.set_speed(self._step_speed(-1))
        elif k in ("backspace", "="):
            self.set_speed(1.0)
        elif k == ".":
            self.frame_step(+1)
        elif k == ",":
            self.frame_step(-1)
        elif k == "v":
            self.cycle_mode(+1)
        elif k == "V":
            self.cycle_mode(-1)
        elif k == "c":
            self.cycle_depth()
        elif k == "e":
            self.cycle_effect(+1)
        elif k == "E":
            self.cycle_effect(-1)
        elif k == "f":
            self.cycle_fit()
        elif k == "z":
            self.set_zoom(self.zoom * 1.25)
        elif k == "Z":
            self.set_zoom(self.zoom / 1.25)
        elif k == "d":
            self.dither = not self.dither
            self.toast(f"Dithering {'on' if self.dither else 'off'}" +
                       ("" if self.depth in (ColorDepth.ANSI256, ColorDepth.ANSI16) else "  (affects 256/16 colors)"))
        elif k == "r":
            self.adjust = filters.Adjust()
            self.effect = "none"
            self.zoom = 1.0
            self.fit = "fit"
            self.force_image = True
            self.need_clear = True
            self.toast("Picture reset")
        elif k == "t":
            self.subs_on = not self.subs_on
            if self.track is None:
                self.toast("No subtitles for this media" + ("" if self.subs_on else ""))
            else:
                self.toast(f"Subtitles {'on' if self.subs_on else 'off'}")
        elif k == "i":
            self.show_stats = not self.show_stats
        elif k in ("?", "h", "f1"):
            self.show_help = not self.show_help
        elif k == "o":
            order = ["auto", "always", "off"]
            self.osd_mode = order[(order.index(self.osd_mode) + 1) % 3]
            self.toast(f"OSD: {self.osd_mode}")
        elif k in ("tab", "enter"):
            self.menu_open = True
        elif k == "s":
            self.screenshot()
        elif k == "n":
            self.action = "next"
        elif k == "p":
            if self.position() > 5 and self.media and self.media.seekable:
                self.seek_to(0.0)
            else:
                self.action = "prev"
        elif k == "L":
            self.loop = not self.loop
            self.toast(f"Loop {'on' if self.loop else 'off'}")
        elif k == "a":
            style = self.visualizer.next_style()
            self.toast(f"Visualizer: {style}")

    def _handle_mouse(self, ev: Mouse) -> None:
        self.last_activity = time.monotonic()
        if ev.kind == "wheel_up":
            self.set_volume(self.volume + 0.05)
        elif ev.kind == "wheel_down":
            self.set_volume(self.volume - 0.05)
        elif ev.kind in ("press", "drag") and ev.button == 0:
            by, bx, bw = self.bar_geometry
            if ev.y == by and bw > 0 and bx - 1 <= ev.x <= bx + bw and self.media and self.media.duration > 0:
                frac = min(max((ev.x - bx) / max(bw - 1, 1), 0.0), 1.0)
                self.seek_to(frac * self.media.duration, debounce=0.25 if ev.kind == "drag" else 0.05)
            elif ev.kind == "press" and not self.menu_open:
                self.toggle_pause()
        elif ev.kind == "press" and ev.button == 2:
            self.menu_open = not self.menu_open

    # -- actions -----------------------------------------------------------
    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused and self.stepped and self.current is not None and self.media.seekable:
            self._start_pipeline(self.current.pts)
            return
        if not self.prerolling:
            self._set_clock_paused(self.paused)
        if self.paused and self.audio is None and isinstance(self.clock, audio_mod.WallClock):
            pass

    def seek_relative(self, delta: float) -> None:
        if not self.media or not self.media.seekable:
            self.toast("Can't seek in a live stream")
            return
        base = self.seek_target if self.seek_target is not None else self.position()
        self.seek_to(base + delta)
        arrow = "»" if delta > 0 else "«"
        self.toast(f"{arrow} {abs(delta):g}s   {fmt_time(self.seek_target)}", 1.0)

    def seek_to(self, t: float, debounce: float = 0.18) -> None:
        m = self.media
        if not m or not m.seekable:
            self.toast("Can't seek in a live stream")
            return
        t = max(0.0, min(t, max(m.duration - 0.5, 0.0)))
        self.seek_target = t
        self.seek_due = time.monotonic() + debounce

    def _maybe_seek(self) -> None:
        if self.seek_target is not None and time.monotonic() >= self.seek_due:
            t = self.seek_target
            self.seek_target = None
            self._start_pipeline(t)

    def frame_step(self, direction: int) -> None:
        if self.media is None or self.media.video is None:
            return
        if not self.paused:
            self.paused = True
            self._set_clock_paused(True)
        d = self.decoder
        if direction > 0 and d is not None:
            try:
                fr = self.pending or d.frames.get(timeout=1.0)
            except queue.Empty:
                return
            self.pending = None
            if fr.image is not None:
                self.current = fr
                self.last_image = fr.image
                self.force_image = True
                self.stepped = True
                self._draw(fr.image)
        elif direction < 0 and self.current is not None and self.media.seekable:
            fps = d.fps if d else 30
            self._start_pipeline(max(self.current.pts - 1.0 / fps, 0.0))

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(v, 2.0))
        self.muted = False
        a = self.audio
        if a is not None:
            a.volume = self.volume
            a.muted = False
            if isinstance(a, audio_mod.FFplayAudio):
                a.refresh()
        self.toast(f"Volume {int(round(self.volume * 100))}%", 1.0)

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        a = self.audio
        if a is not None:
            a.muted = self.muted
            if isinstance(a, audio_mod.FFplayAudio):
                a.refresh()
        self.toast("Muted" if self.muted else "Unmuted", 1.0)

    SPEEDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)

    def _step_speed(self, direction: int) -> float:
        speeds = self.SPEEDS
        if direction > 0:
            return next((s for s in speeds if s > self.speed + 1e-6), speeds[-1])
        return next((s for s in reversed(speeds) if s < self.speed - 1e-6), speeds[0])

    def set_speed(self, s: float) -> None:
        if self.media and self.media.is_live:
            self.toast("Speed is fixed for live sources")
            return
        if abs(s - self.speed) < 1e-6:
            self.toast(f"Speed {s:g}x", 1.0)
            return
        pos = self.position()
        self.speed = s
        if isinstance(self.clock, audio_mod.WallClock) and self.audio is None:
            self.clock.set_speed(s)
        else:
            self._start_pipeline(pos)
        self.toast(f"Speed {s:g}x", 1.0)

    def cycle_mode(self, direction: int) -> None:
        modes = self.modes
        i = modes.index(self.mode) if self.mode in modes else 0
        old_kind = render.get(self.mode).kind
        self._clear_images()
        self.mode = modes[(i + direction) % len(modes)]
        r = render.get(self.mode)
        self.need_clear = True
        self.force_image = True
        self.last_px = None
        if r.kind != old_kind or r.kind == "pixels":
            self._maybe_restart_decoder()
        self.toast(f"Mode: {self.mode}  ·  {render.describe(self.mode)}", 2.2)

    def cycle_depth(self) -> None:
        if render.get(self.mode).kind == "pixels":
            self.toast("Color depth applies to text modes")
            return
        i = DEPTHS.index(self.depth) if self.depth in DEPTHS else 0
        self.depth = DEPTHS[(i + 1) % len(DEPTHS)]
        self.encoder.invalidate()
        self.toast(f"Color: {self.depth.label}")

    def cycle_effect(self, direction: int) -> None:
        i = EFFECT_NAMES.index(self.effect)
        self.effect = EFFECT_NAMES[(i + direction) % len(EFFECT_NAMES)]
        self.force_image = True
        self.toast(f"Effect: {self.effect}")

    def cycle_fit(self) -> None:
        modes = layout_mod.FIT_MODES
        self.fit = modes[(modes.index(self.fit) + 1) % len(modes)]
        self.need_clear = True
        self.force_image = True
        self._maybe_restart_decoder()
        self.toast(f"Fit: {self.fit}")

    def set_zoom(self, z: float) -> None:
        self.zoom = max(1.0, min(z, 8.0))
        self.need_clear = True
        self.force_image = True
        self._maybe_restart_decoder()
        self.toast(f"Zoom {self.zoom:.2f}x")

    def screenshot(self) -> None:
        from .render import raster

        base = Path.home() / "Pictures"
        folder = (base if base.exists() else Path.cwd()) / "TermCinema"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in (self.media.title if self.media else "shot"))[:40].strip()
            stem = folder / f"{safe or 'shot'}-{stamp}"
            if render.get(self.mode).kind == "cells" and self.last_screen is not None:
                img = raster.rasterize(self.last_screen)
                cv2.imwrite(str(stem) + ".png", img[:, :, ::-1])
                ans = CellEncoder().encode(self.last_screen, 0, 0, self.depth)
                Path(str(stem) + ".ans").write_bytes(b"\x1b[2J" + ans + b"\x1b[0m\n")
            elif self.last_px is not None:
                cv2.imwrite(str(stem) + ".png", self.last_px[:, :, ::-1])
            self.toast(f"Saved {stem.name}.png", 3.0)
        except Exception as e:
            self.toast(f"Screenshot failed: {e}", 3.0)

    # -- settings menu -------------------------------------------------------
    def _menu_items(self) -> list:
        a = self.adjust

        def cyc(values, current, d):
            i = values.index(current) if current in values else 0
            return values[(i + d) % len(values)]

        items = [
            ("Render mode", self.mode, lambda d: self._set_mode(cyc(self.modes, self.mode, d))),
            ("Color depth", self.depth.label, lambda d: self._set_depth(cyc(DEPTHS, self.depth, d))),
            ("Effect", self.effect, lambda d: self.cycle_effect(d)),
            ("Fit", self.fit, lambda d: self._set_fit(cyc(list(layout_mod.FIT_MODES), self.fit, d))),
            ("Zoom", f"{self.zoom:.2f}x", lambda d: self.set_zoom(self.zoom * (1.25 if d > 0 else 0.8))),
            ("Brightness", f"{a.brightness:+.2f}", lambda d: self._adj("brightness", d * 0.05, -1, 1)),
            ("Contrast", f"{a.contrast:.2f}", lambda d: self._adj("contrast", d * 0.05, 0.2, 3)),
            ("Saturation", f"{a.saturation:.2f}", lambda d: self._adj("saturation", d * 0.1, 0, 3)),
            ("Gamma", f"{a.gamma:.2f}", lambda d: self._adj("gamma", d * 0.05, 0.3, 3)),
            ("Sharpen", f"{a.sharpen:.2f}", lambda d: self._adj("sharpen", d * 0.1, 0, 2)),
            ("Speed", f"{self.speed:g}x", lambda d: self.set_speed(self._step_speed(d))),
            ("Volume", f"{int(round(self.volume * 100))}%", lambda d: self.set_volume(self.volume + d * 0.05)),
            ("Subtitles", ("on" if self.subs_on else "off") + ("" if self.track else " (none)"),
             lambda d: setattr(self, "subs_on", not self.subs_on)),
            ("Loop", "on" if self.loop else "off", lambda d: setattr(self, "loop", not self.loop)),
            ("Dithering", "on" if self.dither else "off", lambda d: setattr(self, "dither", not self.dither)),
            ("OSD", self.osd_mode, lambda d: setattr(self, "osd_mode", cyc(["auto", "always", "off"], self.osd_mode, d))),
            ("Visualizer", self.visualizer.style, lambda d: setattr(self.visualizer, "style", cyc(list(STYLES), self.visualizer.style, d))),
            ("Cell aspect", f"{self.cell_aspect:.2f}", lambda d: self._set_cell_aspect(self.cell_aspect + d * 0.05)),
            ("Save as defaults", None, lambda d: self._save_defaults()),
        ]
        return items

    def _menu_key(self, k: str) -> bool:
        items = self._menu_items()
        if k in ("esc", "tab", "q"):
            self.menu_open = False
            self.need_clear = render.get(self.mode).kind == "pixels"
            return True
        if k in ("up", "k"):
            self.menu_sel = (self.menu_sel - 1) % len(items)
        elif k in ("down", "j"):
            self.menu_sel = (self.menu_sel + 1) % len(items)
        elif k in ("left", "h", "right", "l", "enter", "space"):
            d = -1 if k in ("left", "h") else 1
            label, value, fn = items[self.menu_sel]
            fn(d)
            self.force_image = True
        else:
            return False
        return True

    def _set_mode(self, m: str) -> None:
        i = self.modes.index(self.mode)
        j = self.modes.index(m)
        self.cycle_mode(j - i)

    def _set_depth(self, d) -> None:
        self.depth = d
        self.encoder.invalidate()

    def _set_fit(self, f: str) -> None:
        self.fit = f
        self.need_clear = True
        self._maybe_restart_decoder()

    def _set_cell_aspect(self, v: float) -> None:
        self.cell_aspect = max(1.0, min(v, 3.5))
        self.need_clear = True

    def _adj(self, name: str, delta: float, lo: float, hi: float) -> None:
        setattr(self.adjust, name, round(max(lo, min(getattr(self.adjust, name) + delta, hi)), 3))

    def _save_defaults(self) -> None:
        from . import config as config_mod

        c = self.cfg
        c.mode, c.color, c.effect, c.fit = self.mode, self.depth.label, self.effect, self.fit
        c.brightness, c.contrast, c.saturation = self.adjust.brightness, self.adjust.contrast, self.adjust.saturation
        c.gamma, c.sharpen = self.adjust.gamma, self.adjust.sharpen
        c.volume = int(round(self.volume * 100))
        c.osd, c.subs, c.dither, c.visualizer = self.osd_mode, self.subs_on, self.dither, self.visualizer.style
        try:
            path = config_mod.save(c)
            self.toast(f"Saved defaults to {path}", 3.0)
        except Exception as e:
            self.toast(f"Could not save: {e}", 3.0)

    def _save_resume(self) -> None:
        m = self.media
        if m is None or not self.cfg.resume or not m.seekable or not m.resume_key:
            return
        self.state.set_resume(m.resume_key, self.position(), m.duration)
