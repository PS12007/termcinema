import numpy as np

from termcinema.terminal import ColorSupport
from termcinema.renderer import base


def test_row_runs_detects_single_run():
    keys = np.array([5, 5, 5, 5])
    starts = base._row_runs(keys)
    assert list(starts) == [0]


def test_row_runs_detects_multiple_runs():
    keys = np.array([1, 1, 2, 2, 2, 3])
    starts = base._row_runs(keys)
    assert list(starts) == [0, 2, 5]


def test_truecolor_cache_reuses_entries():
    base._TC_FG_CACHE.clear()
    c1 = base._tc_fg((255 << 16) | (0 << 8) | 0)
    assert len(base._TC_FG_CACHE) == 1
    c2 = base._tc_fg((255 << 16) | (0 << 8) | 0)
    assert c1 == c2
    assert len(base._TC_FG_CACHE) == 1  # no duplicate entry for the same key


def test_256_code_table_matches_index():
    assert base._FG256_CODES[42] == "38;5;42"
    assert base._BG256_CODES[100] == "48;5;100"


def test_assemble_frame_flat_region_is_compact():
    # A perfectly flat frame should produce exactly one color-run per row
    # -- i.e. one escape sequence, not one per cell.
    h, w = 4, 50
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    glyphs = np.full((h, w), "#")
    out = base.assemble_frame(glyphs, frame, None, ColorSupport.TRUECOLOR)
    for line in out.split("\n"):
        assert line.count("\x1b[38;2;") == 1


def test_assemble_frame_alternating_colors_produces_multiple_runs():
    h, w = 1, 10
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0, ::2] = [255, 0, 0]
    frame[0, 1::2] = [0, 255, 0]
    glyphs = np.full((h, w), "#")
    out = base.assemble_frame(glyphs, frame, None, ColorSupport.TRUECOLOR)
    assert out.count("\x1b[38;2;") == 10  # every cell alternates, no merging
