"""SRT / WebVTT subtitle parsing and lookup."""
from __future__ import annotations

import bisect
import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(s)
    h, mnt, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000.0


def parse(text: str) -> list:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->", 1)
                try:
                    start, end = _ts(a), _ts(b)
                except ValueError:
                    break
                body = "\n".join(lines[i + 1:])
                body = html.unescape(_TAG.sub("", body)).strip()
                if body:
                    cues.append(Cue(start, end, body))
                break
    cues.sort(key=lambda c: c.start)
    # YouTube auto-captions repeat the previous line in each cue ("rolling"
    # captions); keep only the newest line so the overlay doesn't stutter.
    cleaned = []
    for c in cues:
        lines = c.text.split("\n")
        if cleaned and len(lines) > 1 and lines[0] == cleaned[-1].text.split("\n")[-1]:
            c = Cue(c.start, c.end, "\n".join(lines[1:]))
        if cleaned and c.text == cleaned[-1].text and c.start - cleaned[-1].end < 0.2:
            prev = cleaned[-1]
            cleaned[-1] = Cue(prev.start, max(prev.end, c.end), prev.text)
            continue
        cleaned.append(c)
    return cleaned


def load(path: str) -> list:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return parse(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return []


class Track:
    def __init__(self, cues: list):
        self.cues = cues
        self._starts = [c.start for c in cues]

    def at(self, t: float) -> str:
        i = bisect.bisect_right(self._starts, t) - 1
        # Overlapping cues: look back a few for any still active.
        texts = []
        for j in range(max(i - 3, 0), i + 1):
            c = self.cues[j]
            if c.start <= t < c.end:
                texts.append(c.text)
        return "\n".join(texts[-2:])


def find_sidecar(video_path: str) -> "str | None":
    p = Path(video_path)
    for ext in (".srt", ".vtt", ".en.srt", ".en.vtt"):
        cand = p.with_suffix(ext)
        if cand.exists():
            return str(cand)
    return None
