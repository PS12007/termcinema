"""`termcinema doctor`, `termcinema bench`, and single-frame snapshots."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np

from . import __version__, render
from .term import control
from .term.caps import ColorDepth

OK = "\x1b[38;2;90;230;140m✔\x1b[0m"
BAD = "\x1b[38;2;255;90;90m✘\x1b[0m"
MEH = "\x1b[38;2;255;190;70m●\x1b[0m"
DIM = "\x1b[38;2;150;150;165m"
ACC = "\x1b[38;2;255;70;140m"
R = "\x1b[0m"


def _version(cmd: list) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=10)
        line = (res.stdout or res.stderr).decode("utf-8", "replace").splitlines()[0]
        return line.strip()
    except Exception:
        return ""


def run_doctor() -> int:
    control.enable_vt()
    from .media.inputs import which

    print(f"\n{ACC}TermCinema {__version__}{R} {DIM}· doctor{R}\n")
    print(f"  {OK} Python {platform.python_version()} on {platform.system()} {platform.release()}")
    problems = 0

    ff = which("ffmpeg")
    if ff:
        v = _version([ff, "-version"]).replace("ffmpeg version ", "")
        print(f"  {OK} ffmpeg {v.split(' ')[0]}  {DIM}{ff}{R}")
    else:
        problems += 1
        print(f"  {BAD} ffmpeg not found — required.  {DIM}winget install Gyan.FFmpeg | brew install ffmpeg | apt install ffmpeg{R}")
    fp = which("ffprobe")
    print(f"  {OK if fp else BAD} ffprobe {'found' if fp else 'not found'}")
    if not fp:
        problems += 1

    try:
        import yt_dlp

        print(f"  {OK} yt-dlp {yt_dlp.version.__version__}  {DIM}(keep it fresh: pip install -U yt-dlp){R}")
    except ImportError:
        print(f"  {MEH} yt-dlp missing — URLs & YouTube search disabled.  {DIM}pip install yt-dlp{R}")

    from .media import audio

    if audio.HAVE_SOUNDDEVICE:
        try:
            import sounddevice as sd

            dev = sd.query_devices(kind="output")
            print(f"  {OK} audio: sounddevice → {dev['name']}")
        except Exception as e:
            print(f"  {MEH} audio: sounddevice installed but no output device ({e})")
    elif which("ffplay"):
        print(f"  {MEH} audio: using ffplay fallback  {DIM}(pip install sounddevice for instant pause & perfect sync){R}")
    else:
        print(f"  {BAD} audio: no backend  {DIM}pip install sounddevice{R}")

    from .render import pixels

    print(f"  {OK if pixels.HAVE_NUMBA else MEH} numba {'available — fast sixel encoding' if pixels.HAVE_NUMBA else 'missing — sixel will be slow (pip install numba)'}")

    # Terminal
    print(f"\n{ACC}Terminal{R}\n")
    caps = None
    if sys.stdin.isatty() and sys.stdout.isatty():
        from .term import caps as caps_mod
        from .term.keys import InputReader

        with control.RawMode() as raw:
            reader = InputReader().start()
            caps = caps_mod.detect(active=raw.active, reader=reader)
            reader.stop()
    else:
        from .term import caps as caps_mod

        caps = caps_mod.detect(active=False)
    print(f"  {OK} {caps.terminal}  {caps.cols}x{caps.rows}  {DIM}(queried: {'yes' if caps.probed else 'no'}){R}")
    print(f"  {OK if caps.color == ColorDepth.TRUECOLOR else MEH} color: {caps.color.label}")
    if caps.cell_w:
        print(f"  {OK} cell size: {caps.cell_w:.0f}x{caps.cell_h:.0f} px (aspect {caps.cell_aspect:.2f})")
    else:
        print(f"  {MEH} cell size: unknown, assuming aspect 2.0  {DIM}(fix stretching with --cell-aspect){R}")
    for name, label in (("kitty", "kitty graphics"), ("iterm", "iTerm2 images"), ("sixel", "sixel")):
        print(f"  {OK if getattr(caps, name) else DIM + '·' + R} {label}: {'yes' if getattr(caps, name) else 'no'}")
    best = render.auto_mode(caps)
    print(f"\n  → default renderer here: {ACC}{best}{R}")
    if not caps.best_pixel_mode:
        tips = {
            "vscode": "enable \"terminal.integrated.enableImages\" in VS Code settings for real-pixel video",
            "windows-terminal": "update Windows Terminal to 1.22+ for sixel graphics",
            "windows-console": "use Windows Terminal for sixel graphics and faster output",
            "apple-terminal": "iTerm2, WezTerm, kitty or Ghostty give true-color and pixel graphics",
        }
        tip = tips.get(caps.terminal)
        if tip:
            print(f"  {MEH} tip: {tip}")

    # Visual test: gradient + glyph coverage
    print(f"\n{ACC}Color & glyph test{R} {DIM}(should be a smooth rainbow and solid shapes){R}\n")
    width = min(caps.cols - 4, 72)
    line = []
    for i in range(width):
        h = i / width
        r, g, b = _hsv(h)
        line.append(f"\x1b[48;2;{r};{g};{b}m ")
    print("  " + "".join(line) + R)
    samples = [("blocks", "▀▄▌▐▖▗▘▝▚▞▂▆"), ("sextant", "🬀🬁🬂🬃🬄🬅🬆🬇🬈🬉🬊🬋"), ("octant", "𜴀𜴁𜴂𜴃𜴄𜴅𜴆𜴇𜴈𜴉𜴊𜴋"),
               ("braille", "⠁⠃⠇⡇⣇⣧⣷⣿⢿⡿⠿⠟")]
    for name, chars in samples:
        print(f"  {name:8s} \x1b[38;2;255;120;180;48;2;40;20;60m {' '.join(chars)} {R}")
    print(f"  {DIM}If a row shows boxes or '?', your font lacks those glyphs — use --mode blocks.{R}\n")
    return 1 if problems else 0


def _hsv(h: float):
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def run_bench(argv: list) -> int:
    import argparse

    import cv2

    from .render.encode import CellEncoder
    from .media.inputs import Input
    from .media.video import VideoDecoder

    p = argparse.ArgumentParser(prog="termcinema bench")
    p.add_argument("source", nargs="?", default="demo:testsrc")
    p.add_argument("--cols", type=int, default=160)
    p.add_argument("--rows", type=int, default=45)
    p.add_argument("--frames", type=int, default=90)
    a = p.parse_args(argv)
    control.enable_vt()

    frames = []
    if a.source.startswith("demo"):
        from .media.source import demo_media

        m = demo_media(a.source.split(":", 1)[1] if ":" in a.source else "testsrc")
        inp = m.video
    else:
        inp = Input(a.source)
    d = VideoDecoder(inp, 640, 360, 30).start_decoding()
    t = time.perf_counter()
    while len(frames) < a.frames:
        f = d.frames.get(timeout=30)
        if f.image is None:
            break
        frames.append(f.image)
    decode = (time.perf_counter() - t) / max(len(frames), 1)
    d.stop()
    if not frames:
        print("no frames decoded")
        return 1
    print(f"\n{ACC}TermCinema bench{R}  {a.cols}x{a.rows} cells, {len(frames)} frames of {a.source}")
    print(f"  decode  {decode * 1000:6.2f} ms/frame  ({1 / decode:.0f} fps)\n")
    print(f"  {'mode':11s} {'render':>9s} {'encode':>9s} {'KB/frame':>9s} {'max fps':>8s}")
    for name in render.ALL_MODES:
        r = render.get(name)
        enc = CellEncoder()
        tr = te = 0.0
        nbytes = 0
        for img in frames:
            if r.kind == "cells":
                px = cv2.resize(img, (a.cols * r.gw, a.rows * r.gh), interpolation=cv2.INTER_AREA)
                t0 = time.perf_counter()
                cf = r.render(px)
                t1 = time.perf_counter()
                out = enc.encode(cf, 0, 0, ColorDepth.TRUECOLOR, tolerance=3)
                t2 = time.perf_counter()
            else:
                pw, ph = r.pixel_size(a.cols, a.rows, 8, 16)
                px = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
                t0 = t1 = time.perf_counter()
                out = r.render(px, 0, 0, a.cols, a.rows)
                t2 = time.perf_counter()
            tr += t1 - t0
            te += t2 - t1
            nbytes += len(out)
        n = len(frames)
        total = (tr + te) / n
        print(f"  {name:11s} {tr / n * 1000:7.2f}ms {te / n * 1000:7.2f}ms {nbytes / n / 1024:9.1f} {1 / max(total, 1e-6):8.0f}")
    print()
    return 0


def snapshot(source: str, cfg, opts, at: float, png_path, print_frame: bool, width, height, cell_aspect) -> int:
    import cv2

    from .media import source as source_mod
    from .media.video import grab_frame
    from .render import raster
    from .render.encode import CellEncoder, CellFrame
    from .term import caps as caps_mod
    from .ui import layout as layout_mod
    from . import filters

    control.enable_vt()
    try:
        items = source_mod.expand([source], opts)
    except source_mod.SourceError as e:
        print(f"termcinema: {str(e).replace('__search__:', 'not found: ')}", file=sys.stderr)
        return 1
    media = items[0].resolver()
    if media.video is None:
        print("termcinema: source has no video", file=sys.stderr)
        return 1
    caps = caps_mod.detect(active=False)
    cols = width or (caps.cols if sys.stdout.isatty() else 120)
    rows = height or ((caps.rows - 1) if sys.stdout.isatty() else 34)
    mode = cfg.mode if cfg.mode not in ("auto",) else "blocks"
    r = render.get(mode)
    if r.kind != "cells":
        mode, r = "blocks", render.get("blocks")
    aspect = cell_aspect or 2.0
    L = layout_mod.compute(cols, rows, media.aspect, aspect, cfg.fit)
    if print_frame:
        L = layout_mod.Layout(L.w, L.h, 0, 0, L.w, L.h, L.crop)
    src_w = min(max(L.w * 4, 320), media.width or 1920)
    img = grab_frame(media.video, src_w, int(src_w / media.aspect), at)
    if img is None:
        print("termcinema: could not decode a frame", file=sys.stderr)
        return 1
    h, w = img.shape[:2]
    cx, cy, cw, ch = L.crop
    img = img[int(cy * h):int((cy + ch) * h), int(cx * w):int((cx + cw) * w)]
    px = cv2.resize(img, (L.w * r.gw, L.h * r.gh), interpolation=cv2.INTER_AREA)
    px = filters.apply_effect(filters.apply_adjust(px, filters.Adjust(cfg.brightness, cfg.contrast, cfg.saturation,
                                                                        cfg.gamma, cfg.sharpen)), cfg.effect)
    cf = r.render(np.ascontiguousarray(px))
    if png_path:
        out = raster.rasterize(cf)
        cv2.imwrite(png_path, out[:, :, ::-1])
        print(f"saved {png_path}  ({L.w}x{L.h} cells, {mode})")
    if print_frame:
        depth = ColorDepth.parse(cfg.color) if cfg.color != "auto" else caps.color
        data = _linear_ansi(cf, depth or ColorDepth.TRUECOLOR)
        control.Output().write(data)
    return 0


def _linear_ansi(cf, depth) -> bytes:
    """ANSI art without cursor positioning, for printing into scrollback."""
    from .render.encode import CellEncoder, CellFrame

    enc = CellEncoder()
    lines = []
    for y in range(cf.cps.shape[0]):
        row = CellFrame(cf.cps[y:y + 1], cf.fg[y:y + 1], None if cf.bg is None else cf.bg[y:y + 1])
        enc.invalidate()
        b = enc.encode(row, 0, 0, depth)
        # drop the leading cursor-position sequence
        i = b.find(b"H")
        lines.append(b[:4] + b[i + 1:] if b.startswith(b"\x1b[0m\x1b[") else b)
    return b"\n".join(lines) + b"\n"
