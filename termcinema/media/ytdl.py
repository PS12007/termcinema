"""yt-dlp integration: resolve pages to streams, search, playlists, subs.

Streams are never downloaded to disk. yt-dlp gives us direct media URLs
(plus the HTTP headers the site insists on -- skipping those is what
causes 403 errors) and ffmpeg streams them with range requests, so
seeking anywhere in a long video is instant.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from typing import Optional

from .inputs import Input, Media
from . import subtitles


class YtdlError(RuntimeError):
    pass


def _ytdl():
    try:
        import yt_dlp  # noqa: F401
    except ImportError as e:
        raise YtdlError("yt-dlp is not installed. Run:  pip install -U yt-dlp") from e
    return yt_dlp


@dataclass
class SearchResult:
    id: str
    title: str
    url: str
    channel: str = ""
    duration: float = 0.0
    views: int = 0
    thumbnail: str = ""
    is_live: bool = False
    upload_date: str = ""


def _opts(**extra) -> dict:
    o = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "logger": _QuietLogger(),
    }
    o.update(extra)
    return o


class _QuietLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def format_selector(max_height: int, audio_only: bool = False) -> str:
    if audio_only:
        return "bestaudio/best"
    h = f"[height<={max_height}]" if max_height else ""
    # Prefer H.264/VP9 over AV1 (much cheaper to decode in software), then
    # anything that fits, then anything at all.
    return (f"bv*{h}[vcodec!^=av01]+ba/bv*{h}+ba/b{h}/"
            f"bv*[vcodec!^=av01]+ba/bv*+ba/b")


def search(query: str, limit: int = 15) -> list:
    yt_dlp = _ytdl()
    with yt_dlp.YoutubeDL(_opts(extract_flat="in_playlist")) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        except Exception as e:  # pragma: no cover - network
            raise YtdlError(_clean_error(e)) from e
    out = []
    for e in info.get("entries") or []:
        if not e:
            continue
        vid = e.get("id", "")
        url = e.get("url") or f"https://www.youtube.com/watch?v={vid}"
        if not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={vid}"
        thumb = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg" if vid else ""
        if not thumb and e.get("thumbnails"):
            thumb = e["thumbnails"][-1].get("url", "")
        out.append(SearchResult(
            id=vid, title=e.get("title") or url, url=url,
            channel=e.get("channel") or e.get("uploader") or "",
            duration=float(e.get("duration") or 0.0),
            views=int(e.get("view_count") or 0), thumbnail=thumb,
            is_live=e.get("live_status") == "is_live",
        ))
    return out


def expand_playlist(url: str) -> "list | None":
    """If `url` is a playlist/channel page, return [(title, url)], else None."""
    yt_dlp = _ytdl()
    with yt_dlp.YoutubeDL(_opts(extract_flat="in_playlist", noplaylist=True)) as ydl:
        try:
            info = ydl.extract_info(url, download=False, process=False)
        except Exception as e:  # pragma: no cover - network
            raise YtdlError(_clean_error(e)) from e
    if info.get("_type") not in ("playlist", "multi_video"):
        return None
    entries = []
    for e in info.get("entries") or []:
        if not e:
            continue
        u = e.get("url") or e.get("webpage_url")
        if u and not u.startswith("http") and e.get("ie_key", "").lower().startswith("youtube"):
            u = f"https://www.youtube.com/watch?v={u}"
        if u:
            entries.append((e.get("title") or u, u))
    return entries


def resolve(url: str, max_height: int = 720, audio_only: bool = False, sub_lang: Optional[str] = None) -> Media:
    yt_dlp = _ytdl()
    opts = _opts(format=format_selector(max_height, audio_only), noplaylist=True)
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:  # pragma: no cover - network
            raise YtdlError(_clean_error(e)) from e
        if info.get("_type") == "playlist":
            entries = [e for e in info.get("entries") or [] if e]
            if not entries:
                raise YtdlError("Playlist is empty.")
            info = entries[0]

    video = audio = None
    requested = info.get("requested_formats")
    if requested:
        for f in requested:
            inp = Input(f["url"], headers=dict(f.get("http_headers") or info.get("http_headers") or {}))
            has_v = f.get("vcodec") not in (None, "none")
            has_a = f.get("acodec") not in (None, "none")
            if has_v and video is None:
                video = inp
            if has_a and audio is None:
                audio = inp
    elif info.get("url"):
        inp = Input(info["url"], headers=dict(info.get("http_headers") or {}))
        if info.get("vcodec") not in ("none",) and not audio_only:
            video = inp
        if info.get("acodec") not in ("none",):
            audio = inp
        if info.get("vcodec") is None and info.get("acodec") is None:
            video = audio = inp
    if video is None and audio is None:
        raise YtdlError("No playable stream found for this URL.")
    if audio_only:
        video = None

    is_live = bool(info.get("is_live"))
    for inp in (video, audio):
        if inp is not None and is_live:
            inp.seekable = False

    media = Media(
        title=info.get("title") or url,
        video=video,
        audio=audio,
        duration=float(info.get("duration") or 0.0),
        fps=float(info.get("fps") or 0.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        is_live=is_live,
        uploader=info.get("channel") or info.get("uploader") or "",
        webpage_url=info.get("webpage_url") or url,
        thumbnail=info.get("thumbnail") or "",
        kind="stream",
        resume_key=info.get("webpage_url") or url,
        chapters=[(float(c.get("start_time") or 0), c.get("title") or "") for c in info.get("chapters") or []],
    )
    sub_url = _pick_subtitle(info, sub_lang) if sub_lang else None
    if sub_url:
        headers = dict(info.get("http_headers") or {})
        media.subtitle_loader = lambda: subtitles.parse(fetch_text(sub_url, headers))
    return media


def _pick_subtitle(info: dict, lang: str) -> Optional[str]:
    lang = (lang or "en").lower()
    for key in ("subtitles", "automatic_captions"):
        tracks = info.get(key) or {}
        candidates = [k for k in tracks if k.lower() == lang] or \
                     [k for k in tracks if k.lower().startswith(lang + "-") or k.lower().split("-")[0] == lang]
        for k in candidates:
            for f in tracks[k]:
                if f.get("ext") == "vtt" and f.get("url"):
                    return f["url"]
            for f in tracks[k]:
                if f.get("ext") == "srt" and f.get("url"):
                    return f["url"]
    return None


def fetch_bytes(url: str, headers: Optional[dict] = None, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_text(url: str, headers: Optional[dict] = None) -> str:
    return fetch_bytes(url, headers).decode("utf-8", "replace")


def _clean_error(e: Exception) -> str:
    msg = str(e)
    for prefix in ("ERROR: ", "DownloadError: "):
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
    if "Sign in to confirm" in msg:
        msg += "\n(YouTube is rate-limiting this network; try again later or update yt-dlp: pip install -U yt-dlp)"
    elif "HTTP Error 403" in msg or "Requested format is not available" in msg:
        msg += "\n(Updating yt-dlp usually fixes this: pip install -U yt-dlp)"
    return msg
