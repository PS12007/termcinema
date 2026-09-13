"""Renderer registry."""
from __future__ import annotations

from . import pixels, text

_TEXT = {
    "blocks": text.Blocks,
    "octant": text.Octant,
    "sextant": text.Sextant,
    "quadrant": text.Quadrant,
    "halfblock": text.HalfBlock,
    "braille": text.Braille,
    "ascii": text.Ascii,
    "ascii-ramp": text.AsciiRamp,
}
_PIXEL = {
    "kitty": pixels.Kitty,
    "iterm": pixels.ITerm,
    "sixel": pixels.Sixel,
}
ALL_MODES = list(_PIXEL) + list(_TEXT)
TEXT_MODES = list(_TEXT)
PIXEL_MODES = list(_PIXEL)

_cache: dict = {}


def get(name: str):
    if name not in _cache:
        cls = _TEXT.get(name) or _PIXEL.get(name)
        if cls is None:
            raise KeyError(name)
        _cache[name] = cls()
    return _cache[name]


def describe(name: str) -> str:
    cls = _TEXT.get(name) or _PIXEL.get(name)
    return cls.description if cls else ""


def available(caps, include_unsupported: bool = False) -> list:
    """Modes worth cycling through on this terminal, best first."""
    modes = []
    for m in PIXEL_MODES:
        if include_unsupported or getattr(caps, m, False):
            modes.append(m)
    modes += TEXT_MODES
    return modes


def auto_mode(caps) -> str:
    return caps.best_pixel_mode or "blocks"
