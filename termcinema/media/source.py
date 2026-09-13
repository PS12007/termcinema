"""Turn whatever the user typed into playable media.

Understands: local files (video, audio, animated GIF, still images),
directories, direct media URLs, anything yt-dlp supports (YouTube,
Twitch, Vimeo, TikTok, SoundCloud, ...) including playlists, the
webcam, the desktop, and built-in generative demos.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import subtitles
from .inputs import Input, Media, popen_kwargs, probe, which

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts",
             ".m2ts", ".3gp", ".ogv", ".gif", ".apng", ".m3u8", ".mpd"}
AUDIO_EXT = {".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".alac"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".heic", ".jxl"}
MEDIA_EXT = VIDEO_EXT | AUDIO_EXT | IMAGE_EXT


class SourceError(RuntimeError):
    pass


@dataclass
class Item:
    """A playlist entry; resolved to Media lazily when it's about to play."""
    label: str
    resolver: Callable[[], Media]
    kind: str = "file"
    arg: str = ""


@dataclass
class Options:
    max_height: int = 720
    audio_only: bool = False
    sub_lang: Optional[str] = None
    sub_file: Optional[str] = None


DEMOS = {
    "mandelbrot": ("Mandelbrot zoom", "mandelbrot=s=960x540:rate=30:maxiter=2048:end_scale=0.00002:end_pts=120:outer=normalized_iteration_count"),
    "life": ("Game of Life", "life=s=240x135:mold=25:r=24:ratio=0.08:death_color=#401060:life_color=#40ffb0,scale=960:540:flags=neighbor"),
    "gradients": ("Flowing gradients", "gradients=s=960x540:speed=0.015:n=6:type=spiral:r=30"),
    "sierpinski": ("Sierpinski", "sierpinski=s=960x540:type=carpet:jump=10:rate=30"),
    "zoneplate": ("Zone plate", "zoneplate=s=960x540:r=30:kt2=18:ky2=40:kx2=40"),
    "testsrc": ("Test pattern", "testsrc2=s=960x540:r=30"),
    "cellauto": ("Cellular automaton", "cellauto=s=320x180:rule=110:r=30:scroll=1:full=1,scale=960:540:flags=neighbor"),
}

# A little generative arpeggio for the audio-visualizer demo.
SYNTH = ("aevalsrc=exprs='"
         "0.20*sin(2*PI*t*110*pow(2,mod(floor(mod(t*4,16))*5,12)/12))*exp(-5*mod(t*4,1))"
         "+0.10*sin(2*PI*t*220*pow(2,mod(floor(mod(t*2,8))*7,12)/12))*(0.6+0.4*sin(PI*t*0.25))"
         "+0.08*sin(2*PI*t*55)*lt(mod(t,2),1.5)"
         "+0.03*(random(0)-0.5)*exp(-30*mod(t*2,1))"
         "':s=48000:c=stereo")


def is_url(s: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s))


def looks_like_path(s: str) -> bool:
    return (os.sep in s or "/" in s or s.startswith((".", "~")) or
            Path(s).suffix.lower() in MEDIA_EXT)


def _probe_media(inp: Input, title: str, kind: str, opts: Options, path: Optional[str] = None) -> Media:
    info = probe(inp)
    if not info:
        raise SourceError(f"Can't read media: {inp.url}  (is ffmpeg installed and on PATH?)")
    has_v = info.get("has_video", False)
    has_a = info.get("has_audio", False)
    fmt = info.get("format_name", "")
    frames = info.get("frames", 0)
    is_image = has_v and not has_a and (
        fmt.endswith("_pipe") or fmt == "image2" or (Path(inp.url).suffix.lower() in IMAGE_EXT and frames <= 1)
    ) and not fmt.startswith("gif")
    is_gif = fmt.startswith("gif") or fmt == "apng" or Path(inp.url).suffix.lower() in (".gif", ".apng")
    if opts.audio_only:
        has_v = False
    media = Media(
        title=info.get("title") or title,
        video=inp if has_v else None,
        audio=inp if has_a else None,
        duration=0.0 if is_image else info.get("duration", 0.0),
        fps=info.get("fps", 0.0),
        width=info.get("width", 0),
        height=info.get("height", 0),
        is_image=is_image,
        loop=is_gif,
        kind=kind,
        chapters=info.get("chapters", []),
        resume_key=os.path.abspath(path) if path else inp.url,
    )
    if not has_v and not has_a:
        raise SourceError(f"No audio or video streams in {title}")
    # Subtitles: explicit file > sidecar > embedded text stream.
    if path and not is_image:
        sub = opts.sub_file or subtitles.find_sidecar(path)
        if sub:
            media.subtitle_loader = lambda s=sub: subtitles.load(s)
        elif info.get("subtitle_streams"):
            streams = info["subtitle_streams"]
            pick = 0
            if opts.sub_lang:
                for s in streams:
                    if s["lang"].lower().startswith(opts.sub_lang.lower()):
                        pick = s["index"]
                        break
            if streams[pick]["codec"] not in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"):
                media.subtitle_loader = lambda p=path, i=pick: extract_embedded_subs(p, i)
    elif opts.sub_file:
        media.subtitle_loader = lambda s=opts.sub_file: subtitles.load(s)
    return media


def extract_embedded_subs(path: str, index: int) -> list:
    ff = which("ffmpeg") or "ffmpeg"
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path, "-map", f"0:s:{index}", "-f", "srt", "pipe:1"]
    try:
        kw = {k: v for k, v in popen_kwargs().items() if k != "stdin"}
        res = subprocess.run(cmd, capture_output=True, timeout=60, stdin=subprocess.DEVNULL, **kw)
        return subtitles.parse(res.stdout.decode("utf-8", "replace"))
    except Exception:
        return []


def _webcam_input(device: str) -> Input:
    if sys.platform.startswith("win"):
        name = device or _first_dshow_camera()
        if not name:
            raise SourceError("No webcam found (DirectShow).")
        return Input(f"video={name}", pre_args=["-f", "dshow", "-rtbufsize", "64M"], seekable=False)
    if sys.platform == "darwin":
        return Input(device or "0", pre_args=["-f", "avfoundation", "-framerate", "30"], seekable=False)
    return Input(device or "/dev/video0", pre_args=["-f", "v4l2"], seekable=False)


def _first_dshow_camera() -> Optional[str]:
    ff = which("ffmpeg") or "ffmpeg"
    try:
        kw = {k: v for k, v in popen_kwargs().items() if k != "stdin"}
        res = subprocess.run([ff, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                             capture_output=True, timeout=10, stdin=subprocess.DEVNULL, **kw)
    except Exception:
        return None
    text = res.stderr.decode("utf-8", "replace")
    for line in text.splitlines():
        m = re.search(r'"([^"]+)"\s*\(video\)', line)
        if m:
            return m.group(1)
    return None


def _screen_input(device: str) -> Input:
    if sys.platform.startswith("win"):
        return Input(device or "desktop", pre_args=["-f", "gdigrab", "-framerate", "30", "-draw_mouse", "1"], seekable=False)
    if sys.platform == "darwin":
        return Input(device or "Capture screen 0", pre_args=["-f", "avfoundation", "-framerate", "30", "-capture_cursor", "1"], seekable=False)
    display = device or os.environ.get("DISPLAY", ":0.0")
    return Input(display, pre_args=["-f", "x11grab", "-framerate", "30"], seekable=False)


def _live_media(title: str, inp: Input, kind: str, w: int = 1280, h: int = 720, fps: float = 30.0) -> Media:
    return Media(title=title, video=inp, width=w, height=h, fps=fps, kind=kind, is_live=True)


def expand(args: list, opts: Options, shuffle: bool = False) -> list:
    """Expand CLI source arguments into playlist Items (no network yet)."""
    items: list = []
    for arg in args:
        low = arg.lower()
        if not os.path.exists(arg):
            if low in ("webcam", "camera", "cam") or low.startswith(("webcam:", "camera:")):
                dev = arg.split(":", 1)[1] if ":" in arg else ""
                items.append(Item("Webcam", lambda d=dev: _mirror(_live_media("Webcam", _webcam_input(d), "webcam")), "webcam", arg))
                continue
            if low in ("screen", "desktop") or low.startswith("screen:"):
                dev = arg.split(":", 1)[1] if ":" in arg else ""
                items.append(Item("Desktop", lambda d=dev: _screen_media(d), "screen", arg))
                continue
            if low == "demo" or low.startswith("demo:"):
                name = arg.split(":", 1)[1] if ":" in arg else "mandelbrot"
                items.append(Item(f"Demo: {name}", lambda n=name: demo_media(n), "demo", arg))
                continue
        if os.path.isdir(arg):
            files = sorted(p for p in Path(arg).iterdir() if p.suffix.lower() in MEDIA_EXT)
            for p in files:
                items.append(Item(p.name, lambda s=str(p): _probe_media(Input(s), Path(s).name, "file", opts, s), "file", str(p)))
            continue
        if os.path.exists(arg):
            items.append(Item(Path(arg).name, lambda s=arg: _probe_media(Input(s), Path(s).name, "file", opts, s), "file", arg))
            continue
        if is_url(arg):
            items.extend(_url_items(arg, opts))
            continue
        if looks_like_path(arg):
            raise SourceError(f"File not found: {arg}")
        raise SourceError(f"__search__:{arg}")
    if shuffle:
        import random

        random.shuffle(items)
    return items


def _mirror(m: Media) -> Media:
    m.kind = "webcam"
    return m


def _screen_media(dev: str) -> Media:
    inp = _screen_input(dev)
    info = probe(inp, timeout=8)
    w, h = info.get("width", 1920), info.get("height", 1080)
    return _live_media("Desktop", inp, "screen", w or 1920, h or 1080, 30.0)


def demo_media(name: str) -> Media:
    if name == "synth":
        a = Input(SYNTH, pre_args=["-re", "-f", "lavfi"], seekable=False)
        return Media(title="Demo: synth visualizer", audio=a, kind="demo", is_live=True)
    if name not in DEMOS:
        raise SourceError(f"Unknown demo '{name}'. Try: {', '.join(list(DEMOS) + ['synth'])}")
    title, graph = DEMOS[name]
    v = Input(graph, pre_args=["-re", "-f", "lavfi"], seekable=False)
    return Media(title=f"Demo: {title}", video=v, width=960, height=540, fps=30.0, kind="demo", is_live=True)


def _url_items(url: str, opts: Options) -> list:
    from . import ytdl

    path = url.split("?", 1)[0].lower()
    if any(path.endswith(ext) for ext in MEDIA_EXT):
        return [Item(url.rsplit("/", 1)[-1], lambda u=url: _probe_media(Input(u), u.rsplit("/", 1)[-1], "url", opts), "url", url)]
    if not url.startswith(("http://", "https://")):
        # rtsp://, rtmp://, srt://, udp:// ... hand straight to ffmpeg.
        return [Item(url, lambda u=url: _live_url(u, opts), "url", url)]
    is_watch = "watch?v=" in url or "youtu.be/" in url
    if not is_watch:
        try:
            entries = ytdl.expand_playlist(url)
        except ytdl.YtdlError:
            entries = None
        if entries:
            return [Item(t, lambda u=u: ytdl.resolve(u, opts.max_height, opts.audio_only, opts.sub_lang), "stream", u)
                    for t, u in entries]
    return [Item(url, lambda u=url: ytdl.resolve(u, opts.max_height, opts.audio_only, opts.sub_lang), "stream", url)]


def _live_url(url: str, opts: Options) -> Media:
    inp = Input(url, seekable=False)
    m = _probe_media(inp, url, "url", opts)
    m.is_live = True
    return m
