import numpy as np
import pytest

from termcinema.terminal import ColorSupport
from termcinema import renderer, colorspace, scaler


def synthetic_frame(h=64, w=64):
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = (xx / w * 255).astype(np.uint8)
    frame[..., 1] = (yy / h * 255).astype(np.uint8)
    frame[..., 2] = 100
    return frame


@pytest.mark.parametrize("mode", list(renderer.MODES.keys()))
@pytest.mark.parametrize("support", list(ColorSupport))
def test_all_modes_render_without_error(mode, support):
    frame = synthetic_frame()
    out = renderer.render_frame(mode, frame, support)
    assert isinstance(out, str)
    assert len(out) > 0
    # Every mode should emit at least as many newlines as rows-1.
    assert "\n" in out or frame.shape[0] // renderer.MODES[mode].rows_per_cell <= 1


def test_halfblock_row_count():
    frame = synthetic_frame(h=40, w=20)
    out = renderer.render_frame("halfblock", frame, ColorSupport.TRUECOLOR)
    assert len(out.split("\n")) == 20  # 40 pixel rows / 2 rows_per_cell


def test_braille_row_col_reduction():
    frame = synthetic_frame(h=32, w=32)
    out = renderer.render_frame("braille", frame, ColorSupport.TRUECOLOR)
    rows = out.split("\n")
    assert len(rows) == 8  # 32 / 4 rows_per_cell


def test_mono_has_no_escape_codes():
    frame = synthetic_frame()
    out = renderer.render_frame("ascii", frame, ColorSupport.MONO, colorize=False, edge_aware=False)
    assert "\x1b[" not in out


def test_truecolor_has_escape_codes():
    frame = synthetic_frame()
    out = renderer.render_frame("halfblock", frame, ColorSupport.TRUECOLOR)
    assert "\x1b[38;2;" in out or "\x1b[48;2;" in out


def test_next_mode_cycles():
    names = list(renderer.MODES.keys())
    for i, name in enumerate(names):
        assert renderer.next_mode(name) == names[(i + 1) % len(names)]


def test_edge_aware_detects_a_hard_edge():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, 10:] = 255
    out = renderer.render_frame("ascii", frame, ColorSupport.MONO, colorize=False, edge_aware=True)
    assert "|" in out
