"""Command-line entry point."""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from . import config as config_mod
from . import filters, render
from .term import control
from .term.caps import ColorDepth

EPILOG = """\
sources:
  FILE / DIR               video, audio, GIF or image files; a folder plays everything in it
  URL                      YouTube, Twitch, Vimeo, SoundCloud... (anything yt-dlp supports),
                           playlists, direct .mp4/.m3u8 links, rtsp:// streams
  "search words"           search YouTube and pick from results with thumbnails
  webcam[:DEVICE]          your camera          screen    your desktop
  demo[:NAME]              mandelbrot, life, gradients, sierpinski, zoneplate, cellauto, synth

commands:
  termcinema               open the home screen (search, recent, demos)
  termcinema doctor        check dependencies and what your terminal supports
  termcinema bench FILE    measure renderer speed on your machine

examples:
  termcinema movie.mkv
  termcinema "https://youtu.be/..." --mode octant --effect vivid
  termcinema "lofi hip hop radio" --audio-only
  termcinema ~/Videos --shuffle --loop
  termcinema clip.mp4 --snapshot shot.png --at 1:23
  termcinema photo.jpg --print --width 80
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="termcinema",
        description="Watch anything in your terminal.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sources", nargs="*", help="files, folders, URLs, search terms, webcam, screen or demo")
    g = p.add_argument_group("picture")
    g.add_argument("-m", "--mode", choices=["auto"] + render.ALL_MODES, metavar="MODE",
                   help="renderer: " + ", ".join(render.ALL_MODES) + " (default: best for your terminal)")
    g.add_argument("-c", "--color", choices=["auto", "truecolor", "256", "16", "mono"], metavar="DEPTH",
                   help="color depth for text modes: auto, truecolor, 256, 16, mono")
    g.add_argument("-e", "--effect", choices=list(filters.EFFECTS), metavar="EFFECT", help="visual effect (see --list-effects)")
    g.add_argument("--fit", choices=["fit", "fill", "stretch"], help="how the picture fills the terminal")
    g.add_argument("--brightness", type=float)
    g.add_argument("--contrast", type=float)
    g.add_argument("--saturation", type=float)
    g.add_argument("--gamma", type=float)
    g.add_argument("--sharpen", type=float)
    g.add_argument("--dither", action="store_true", help="ordered dithering for 256/16-color modes")
    g.add_argument("--cell-aspect", type=float, help="character cell height/width ratio (default: detected, ~2.0)")
    g.add_argument("--width", type=int, help="columns to use (default: terminal width)")
    g.add_argument("--height", type=int, help="rows to use (default: terminal height)")
    g.add_argument("--tolerance", type=int, help="color change threshold for skipping redraws (0 = exact)")

    g = p.add_argument_group("playback")
    g.add_argument("-s", "--start", help="start position, e.g. 90, 1:30, 1:02:03")
    g.add_argument("--speed", type=float, help="playback speed (0.25 - 4)")
    g.add_argument("--volume", type=int, help="volume percent (0 - 200)")
    g.add_argument("--no-audio", action="store_true", help="disable sound")
    g.add_argument("--audio-only", action="store_true", help="skip video (shows a visualizer)")
    g.add_argument("--loop", action="store_true", help="loop the file or playlist")
    g.add_argument("--shuffle", action="store_true", help="shuffle the playlist")
    g.add_argument("-q", "--quality", type=int, help="max stream height for URLs (e.g. 360, 480, 720, 1080)")
    g.add_argument("--max-fps", type=float, help="cap the frame rate")
    g.add_argument("--hwaccel", choices=["auto", "on", "off"], help="hardware video decoding")
    g.add_argument("--subs", nargs="?", const="en", metavar="LANG", help="subtitles language (default en)")
    g.add_argument("--sub-file", help="load subtitles from an .srt/.vtt file")
    g.add_argument("--no-subs", action="store_true")
    g.add_argument("--no-resume", action="store_true", help="don't resume where you left off")
    g.add_argument("--osd", choices=["auto", "always", "off"], help="on-screen display behaviour")
    g.add_argument("--visualizer", choices=["spectrum", "mirror", "wave", "radial"])

    g = p.add_argument_group("output & tools")
    g.add_argument("--snapshot", metavar="PNG", help="render one frame to a PNG (as the terminal would show it) and exit")
    g.add_argument("--print", dest="print_frame", action="store_true", help="print one frame to stdout as ANSI art and exit")
    g.add_argument("--at", default="0", help="timestamp for --snapshot/--print")
    g.add_argument("--save-config", action="store_true", help="save these options as defaults")
    g.add_argument("--list-modes", action="store_true")
    g.add_argument("--list-effects", action="store_true")
    g.add_argument("--no-probe", action="store_true", help="skip querying the terminal for graphics support")
    g.add_argument("-V", "--version", action="version", version=f"termcinema {__version__}")
    return p


def apply_args(cfg: config_mod.Config, a) -> config_mod.Config:
    mapping = {
        "mode": a.mode, "color": a.color, "effect": a.effect, "fit": a.fit, "brightness": a.brightness,
        "contrast": a.contrast, "saturation": a.saturation, "gamma": a.gamma, "sharpen": a.sharpen,
        "tolerance": a.tolerance, "speed": a.speed, "volume": a.volume, "quality": a.quality,
        "max_fps": a.max_fps, "hwaccel": a.hwaccel, "osd": a.osd, "visualizer": a.visualizer,
    }
    for k, v in mapping.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.dither:
        cfg.dither = True
    if a.no_subs:
        cfg.subs = False
    if a.subs:
        cfg.subs = True
        cfg.sub_lang = a.subs
    if a.no_resume:
        cfg.resume = False
    if a.no_audio:
        cfg.audio = False
    cfg.speed = max(0.25, min(cfg.speed, 4.0))
    return cfg


class Session:
    """Owns the terminal while the interactive UI runs."""

    def __init__(self, probe: bool = True):
        self.probe = probe
        self.out = control.Output()
        self.raw = control.RawMode()
        self.reader = None
        self.caps = None

    def __enter__(self):
        from .term import caps as caps_mod
        from .term.keys import InputReader

        control.enable_vt()
        self.raw.__enter__()
        self.reader = InputReader().start()
        self.caps = caps_mod.detect(active=self.probe and self.raw.active, reader=self.reader)
        self.out.write(control.ALT_SCREEN_ON + control.HIDE_CURSOR + control.AUTOWRAP_OFF +
                       control.MOUSE_ON + "\x1b[0m\x1b[2J")
        return self

    def __exit__(self, *exc):
        try:
            self.out.write(control.MOUSE_OFF + control.SYNC_END + "\x1b[0m" + control.AUTOWRAP_ON +
                           control.SHOW_CURSOR + control.ALT_SCREEN_OFF)
        except Exception:
            pass
        if self.reader is not None:
            self.reader.stop()
        self.raw.__exit__(*exc)
        return False


def _parse_time(s: str) -> float:
    from .ui.canvas import parse_time

    try:
        return parse_time(s)
    except ValueError:
        raise SystemExit(f"termcinema: invalid time '{s}' (use seconds or h:mm:ss)")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if argv and argv[0] == "doctor":
        from .doctor import run_doctor

        return run_doctor()
    if argv and argv[0] == "bench":
        from .doctor import run_bench

        return run_bench(argv[1:])

    parser = build_parser()
    a = parser.parse_args(argv)

    if a.list_modes:
        for name in render.ALL_MODES:
            print(f"  {name:11s} {render.describe(name)}")
        return 0
    if a.list_effects:
        print("  " + "  ".join(filters.EFFECTS))
        return 0

    cfg = apply_args(config_mod.load(), a)
    if a.save_config:
        print(f"Saved defaults to {config_mod.save(cfg)}")
        if not a.sources:
            return 0

    from .media.inputs import which

    if not which("ffmpeg"):
        print("termcinema needs ffmpeg on your PATH.\n"
              "  Windows: winget install Gyan.FFmpeg     macOS: brew install ffmpeg     Linux: apt install ffmpeg",
              file=sys.stderr)
        return 2

    from .media import source as source_mod

    opts = source_mod.Options(max_height=cfg.quality, audio_only=a.audio_only,
                              sub_lang=cfg.sub_lang if cfg.subs else None, sub_file=a.sub_file)

    if a.snapshot or a.print_frame:
        from .doctor import snapshot

        if not a.sources:
            parser.error("--snapshot/--print need a source")
        return snapshot(a.sources[0], cfg, opts, _parse_time(a.at), a.snapshot, a.print_frame,
                        a.width, a.height, a.cell_aspect)

    start_at = _parse_time(a.start) if a.start else None
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if not a.sources and not interactive:
        parser.print_help()
        return 1

    state = config_mod.State()
    errors = []
    try:
        with Session(probe=not a.no_probe) as session:
            if a.sources:
                items, query = _expand(a.sources, opts, a.shuffle)
                if query is not None:
                    items = _search_flow(session, query, opts) or []
                if items:
                    errors += _play(session, items, cfg, state, a, start_at)
            else:
                _home_loop(session, cfg, state, opts, a)
    except KeyboardInterrupt:
        pass
    except source_mod.SourceError as e:
        print(f"termcinema: {e}", file=sys.stderr)
        return 1
    finally:
        state.save()
    for label, err in errors:
        print(f"termcinema: {label}: {err}", file=sys.stderr)
    return 0


def _expand(sources, opts, shuffle):
    from .media import source as source_mod

    items = []
    query = None
    for s in sources:
        try:
            items += source_mod.expand([s], opts, shuffle=False)
        except source_mod.SourceError as e:
            msg = str(e)
            if msg.startswith("__search__:"):
                query = msg.split(":", 1)[1] if not items else query
                if len(sources) > 1:
                    query = " ".join(sources)
                    return [], query
            else:
                raise
    if shuffle:
        import random

        random.shuffle(items)
    return items, query if not items else None


def _search_flow(session: Session, query: str, opts):
    from .media import source as source_mod
    from .media import ytdl
    from .ui import home

    while True:
        ok, res = home.loading(session.out, session.reader, session.caps, f"Searching YouTube for “{query}”",
                               lambda: ytdl.search(query, 20))
        if not ok:
            if res is not None:
                _flash(session, f"Search failed: {res}")
            return None
        if not res:
            _flash(session, "No results.")
            return None
        urls = home.Picker(session.out, session.reader, session.caps, query, res).run()
        if not urls:
            return None
        titles = {r.url: r.title for r in res}
        return [source_mod.Item(titles.get(u, u),
                                lambda u=u: ytdl.resolve(u, opts.max_height, opts.audio_only, opts.sub_lang),
                                "stream", u) for u in urls]


def _flash(session: Session, message: str, secs: float = 2.5) -> None:
    from .ui import home
    from .term.keys import Key

    end = time.monotonic() + secs
    screen = home._Screen(session.out, session.caps.color)
    while time.monotonic() < end:
        frame, cv = screen.begin()
        rows, cols = frame.cps.shape
        from .ui.canvas import WARN, truncate, text_width

        msg = truncate(message, cols - 4)
        cv.text(rows // 2, (cols - text_width(msg)) // 2, msg, WARN)
        screen.flush(frame, cv)
        ev = session.reader.get(timeout=0.1)
        if isinstance(ev, Key):
            break


def _play(session: Session, items, cfg, state, a, start_at):
    from .player import Player

    player = Player(items, cfg, session.caps, session.out, session.reader, state, start_at=start_at,
                    audio_enabled=not a.no_audio, cell_aspect=a.cell_aspect, forced_size=(a.width, a.height))
    if a.loop:
        player.loop = True
    try:
        player.run()
    finally:
        session.out.write("\x1b[0m\x1b[2J")
    return getattr(player, "errors", [])


def _home_loop(session: Session, cfg, state, opts, a):
    from .media import source as source_mod
    from .ui import home

    caps = session.caps
    gfx = [k for k in ("kitty", "iterm", "sixel") if getattr(caps, k)]
    status = f"{caps.terminal} · {caps.color.label}" + (f" · {'/'.join(gfx)}" if gfx else "")
    while True:
        action, payload = home.Home(session.out, session.reader, caps, state.history, status).run()
        if action == "quit":
            return
        if action == "search":
            items = _search_flow(session, payload, opts)
        else:
            try:
                items, query = _expand(payload, opts, False)
                if query is not None:
                    items = _search_flow(session, query, opts)
            except source_mod.SourceError as e:
                _flash(session, str(e))
                continue
        if items:
            errors = _play(session, items, cfg, state, a, None)
            for label, err in errors:
                _flash(session, f"{label}: {err.splitlines()[0]}", 3.0)
        session.out.write("\x1b[0m\x1b[2J")


if __name__ == "__main__":
    sys.exit(main())
