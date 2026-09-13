"""Player configuration: CLI defaults, loaded from and saveable to
~/.config/termcinema/config.toml. Uses the stdlib `tomllib` for reading
(Python 3.11+) so no extra dependency is required just to load config;
writing is done with a tiny hand-rolled TOML serializer since we only
ever write flat key/value pairs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None


CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "termcinema"
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    mode: str = "halfblock"
    color: str = "auto"          # auto | truecolor | 256 | 16 | mono
    theme: str = "normal"
    fps: float = 0.0             # 0 = use source fps
    audio: bool = True
    hw_accel: bool = True
    gpu: bool = True             # use CUDA for resize/color-convert/sobel when available
    edge_aware: bool = True
    dither: bool = True          # shape dithering (braille/quarterblock)
    color_dither: bool = False   # ordered dithering before 256/16-color quantization
    volume: int = 100
    quality: str = "best"        # for URL/YouTube playback


def load() -> Config:
    cfg = Config()
    if tomllib and CONFIG_PATH.exists():
        try:
            data = tomllib.loads(CONFIG_PATH.read_text())
            valid = {f.name for f in fields(Config)}
            for k, v in data.items():
                if k in valid:
                    setattr(cfg, k, v)
        except Exception:
            pass
    return cfg


def save(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, v in asdict(cfg).items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    CONFIG_PATH.write_text("\n".join(lines) + "\n")
