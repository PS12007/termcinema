from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..terminal import ColorSupport
from . import ascii_mode, blocks, braille, halfblock


@dataclass(frozen=True)
class Mode:
    name: str
    rows_per_cell: int   # source pixel rows consumed per terminal row
    cols_per_cell: int   # source pixel cols consumed per terminal col
    fn: Callable[..., str]
    kwargs: dict


def _ascii(ramp):
    return lambda frame, support, **kw: ascii_mode.render(frame, support, ramp=ramp, **kw)


MODES: dict[str, Mode] = {
    "ascii": Mode("ascii", 1, 1, _ascii("classic"), {}),
    "ascii-extended": Mode("ascii-extended", 1, 1, _ascii("extended"), {}),
    "ascii-unicode": Mode("ascii-unicode", 1, 1, _ascii("unicode"), {}),
    "halfblock": Mode("halfblock", 2, 1, halfblock.render, {}),
    "quarterblock": Mode("quarterblock", 2, 2, blocks.render, {}),
    "braille": Mode("braille", 4, 2, braille.render, {}),
}

DEFAULT_MODE = "halfblock"


def render_frame(mode: str, frame_rgb: np.ndarray, support: ColorSupport, **overrides) -> str:
    m = MODES[mode]
    kwargs = dict(m.kwargs)
    kwargs.update(overrides)
    return m.fn(frame_rgb, support, **kwargs)


def next_mode(mode: str) -> str:
    names = list(MODES.keys())
    i = names.index(mode)
    return names[(i + 1) % len(names)]
