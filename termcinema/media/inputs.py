"""Media descriptions and ffmpeg input plumbing shared by decoders."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

from .subtitles import Cue

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def popen_kwargs() -> dict:
    kw = {"stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = _NO_WINDOW
    return kw


def which(tool: str) -> Optional[str]:
    return shutil.which(tool)


@dataclass
class Input:
    """One ffmpeg input: ``[pre_args...] -i url``."""
    url: str
    pre_args: list = field(default_factory=list)
    headers: dict = field(default_factory=dict)
    seekable: bool = True

    @property
    def is_network(self) -> bool:
        return self.url.startswith(("http://", "https://"))

    def ffmpeg_args(self, start: float = 0.0) -> list:
        args = []
        if self.is_network:
            args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
            if self.headers:
                ua = self.headers.get("User-Agent")
                if ua:
                    args += ["-user_agent", ua]
                other = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items() if k.lower() != "user-agent")
                if other:
                    args += ["-headers", other]
        if start > 0.01 and self.seekable:
            args += ["-ss", f"{start:.3f}"]
        args += list(self.pre_args)
        args += ["-i", self.url]
        return args


@dataclass
class Media:
    title: str
    video: Optional[Input] = None
    audio: Optional[Input] = None
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    is_live: bool = False
    is_image: bool = False
    loop: bool = False
    uploader: str = ""
    webpage_url: str = ""
    thumbnail: str = ""
    kind: str = "file"
    subtitles: list = field(default_factory=list)          # list[Cue]
    subtitle_loader: Optional[Callable[[], list]] = None   # lazy subtitle fetch
    chapters: list = field(default_factory=list)           # [(start, title)]
    resume_key: str = ""

    @property
    def seekable(self) -> bool:
        return not self.is_live and self.duration > 0 and (self.video or self.audio) is not None and \
            (self.video.seekable if self.video else self.audio.seekable)

    @property
    def aspect(self) -> float:
        if self.width > 0 and self.height > 0:
            return self.width / self.height
        return 16 / 9


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        v = stream.get(key) or "0/0"
        try:
            f = float(Fraction(v)) if "/0" not in v else 0.0
        except (ValueError, ZeroDivisionError):
            f = 0.0
        if 0 < f <= 240:
            return f
    return 0.0


def probe(inp: Input, timeout: float = 25.0) -> dict:
    """Run ffprobe and return a summary dict (empty dict on failure)."""
    ffprobe = which("ffprobe")
    if not ffprobe:
        return {}
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-show_chapters"]
    cmd += [a for a in inp.ffmpeg_args() if a != "-i"]
    # ffprobe takes the url positionally; strip the -i we added and keep order.
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, **{k: v for k, v in popen_kwargs().items() if k != "stdin"})
        data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    except Exception:
        return {}
    return summarize(data)


def summarize(data: dict) -> dict:
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    out = {"has_video": False, "has_audio": False, "subtitle_streams": [], "duration": 0.0}
    try:
        out["duration"] = float(fmt.get("duration") or 0.0)
    except ValueError:
        pass
    for s in streams:
        ct = s.get("codec_type")
        disp = s.get("disposition", {}) or {}
        if ct == "video" and not out["has_video"] and not disp.get("attached_pic"):
            w, h = int(s.get("width") or 0), int(s.get("height") or 0)
            sar = s.get("sample_aspect_ratio") or "1:1"
            try:
                num, den = (int(x) for x in sar.split(":"))
                if num > 0 and den > 0 and num != den:
                    w = int(round(w * num / den))
            except ValueError:
                pass
            rot = 0
            for sd in s.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(float(sd["rotation"]))
            rot = rot or int(float((s.get("tags") or {}).get("rotate", 0) or 0))
            if abs(rot) % 180 == 90:
                w, h = h, w
            out.update(has_video=True, width=w, height=h, fps=_fps(s), vcodec=s.get("codec_name", ""))
            nb = s.get("nb_frames")
            out["frames"] = int(nb) if nb and str(nb).isdigit() else 0
            if not out["duration"]:
                try:
                    out["duration"] = float(s.get("duration") or 0.0)
                except ValueError:
                    pass
        elif ct == "audio" and not out["has_audio"]:
            out.update(has_audio=True, acodec=s.get("codec_name", ""))
        elif ct == "subtitle":
            tags = s.get("tags") or {}
            out["subtitle_streams"].append({
                "index": len(out["subtitle_streams"]),
                "codec": s.get("codec_name", ""),
                "lang": tags.get("language", ""),
                "title": tags.get("title", ""),
            })
    out["format_name"] = fmt.get("format_name", "")
    out["title"] = (fmt.get("tags") or {}).get("title", "")
    out["chapters"] = [(float(c.get("start_time", 0)), (c.get("tags") or {}).get("title", ""))
                       for c in data.get("chapters", []) or []]
    return out
