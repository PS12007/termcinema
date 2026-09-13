"""User config (TOML), watch history and resume positions."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    tomllib = None


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "termcinema"
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "termcinema"
    return Path.home() / ".config" / "termcinema"


CONFIG_PATH = config_dir() / "config.toml"
STATE_PATH = config_dir() / "state.json"


@dataclass
class Config:
    mode: str = "auto"
    color: str = "auto"
    effect: str = "none"
    fit: str = "fit"
    quality: int = 720
    volume: int = 100
    speed: float = 1.0
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    sharpen: float = 0.0
    tolerance: int = 3
    max_fps: float = 60.0
    hwaccel: str = "auto"      # auto | on | off
    osd: str = "auto"          # auto | always | off
    subs: bool = True
    sub_lang: str = "en"
    resume: bool = True
    audio: bool = True
    dither: bool = False
    visualizer: str = "spectrum"


def load() -> Config:
    cfg = Config()
    if tomllib is None or not CONFIG_PATH.exists():
        return cfg
    try:
        data = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return cfg
    types = {f.name: f.type for f in fields(Config)}
    for k, v in data.items():
        if k in types:
            try:
                cur = getattr(cfg, k)
                setattr(cfg, k, type(cur)(v))
            except (TypeError, ValueError):
                pass
    return cfg


def save(cfg: Config) -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TermCinema settings -- CLI flags override these.", ""]
    for k, v in asdict(cfg).items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CONFIG_PATH


class State:
    """Watch history + resume positions, persisted as JSON."""

    def __init__(self):
        self.data = {"resume": {}, "history": []}
        try:
            if STATE_PATH.exists():
                self.data.update(json.loads(STATE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass

    def save(self) -> None:
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except Exception:
            pass

    def resume_position(self, key: str) -> float:
        entry = self.data["resume"].get(key)
        return float(entry["pos"]) if entry else 0.0

    def set_resume(self, key: str, pos: float, duration: float) -> None:
        if not key:
            return
        if duration and (pos < 20 or pos > duration - 20):
            self.data["resume"].pop(key, None)
        else:
            self.data["resume"][key] = {"pos": round(pos, 1), "t": int(time.time())}
        if len(self.data["resume"]) > 300:
            oldest = sorted(self.data["resume"].items(), key=lambda kv: kv[1].get("t", 0))
            for k, _ in oldest[:50]:
                self.data["resume"].pop(k, None)

    def add_history(self, title: str, source: str) -> None:
        hist = [h for h in self.data["history"] if h.get("source") != source]
        hist.insert(0, {"title": title, "source": source, "t": int(time.time())})
        self.data["history"] = hist[:40]

    @property
    def history(self) -> list:
        return self.data["history"]
