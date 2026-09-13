"""Resolve YouTube (and other yt-dlp-supported site) URLs to a direct,
seekable media URL that OpenCV/FFmpeg can open directly -- no need to
download the whole file first ("true streaming").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResolvedStream:
    video_url: str
    audio_url: Optional[str]
    title: str
    is_live: bool


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def resolve(url: str, quality: str = "best") -> ResolvedStream:
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for URL playback. Install it with:\n"
            "  pip install yt-dlp"
        ) from e

    fmt = {
        "best": "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio/best[ext=mp4]/best",
        "worst": "worstvideo+worstaudio/worst",
        "audio": "bestaudio",
    }.get(quality, quality)

    ydl_opts = {
        "format": fmt,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    is_live = bool(info.get("is_live"))

    # A merged progressive format has both audio+video in one URL. A
    # split (video-only + audio-only) selection gives us two formats we
    # combine with ffmpeg's demuxer via two separate inputs.
    if "url" in info:
        video_url = info["url"]
        audio_url = None
    else:
        requested = info.get("requested_formats") or []
        video_url = None
        audio_url = None
        for f in requested:
            if f.get("vcodec") not in (None, "none"):
                video_url = f["url"]
            if f.get("acodec") not in (None, "none"):
                audio_url = f["url"]
        if video_url is None and requested:
            video_url = requested[0]["url"]

    if video_url is None:
        raise RuntimeError("Could not resolve a playable stream for this URL.")

    return ResolvedStream(
        video_url=video_url,
        audio_url=audio_url,
        title=info.get("title", url),
        is_live=is_live,
    )
