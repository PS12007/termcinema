import numpy as np
import pytest

from termcinema.render.encode import CONTINUATION, CellEncoder, CellFrame
from termcinema.term.caps import ColorDepth

from vt import Screen


def random_frame(rows, cols, seed=0, palette=None):
    rng = np.random.default_rng(seed)
    cps = rng.choice(np.array([0x20, 0x2580, 0x2598, 0x41, 0x1CD00, 0x28FF], np.int32), size=(rows, cols))
    if palette is None:
        fg = rng.integers(0, 256, (rows, cols, 3), dtype=np.uint8)
        bg = rng.integers(0, 256, (rows, cols, 3), dtype=np.uint8)
    else:
        fg = palette[rng.integers(0, len(palette), (rows, cols))]
        bg = palette[rng.integers(0, len(palette), (rows, cols))]
    return CellFrame(cps, fg, bg)


def test_full_frame_reproduces_exactly():
    f = random_frame(12, 40)
    scr = Screen(12, 40)
    scr.feed(CellEncoder().encode(f, 0, 0, ColorDepth.TRUECOLOR))
    assert (scr.cps == f.cps).all()
    assert (scr.fg == f.fg).all()
    assert (scr.bg == f.bg).all()


def test_runs_of_equal_color_share_one_escape():
    cps = np.full((1, 50), 0x2580, np.int32)
    fg = np.full((1, 50, 3), 10, np.uint8)
    bg = np.full((1, 50, 3), 200, np.uint8)
    data = CellEncoder().encode(CellFrame(cps, fg, bg), 0, 0, ColorDepth.TRUECOLOR)
    assert data.count(b"38;2;") == 1
    assert data.count(b"48;2;") == 1


def test_unchanged_frame_emits_nothing():
    f = random_frame(10, 30)
    enc = CellEncoder()
    enc.encode(f, 0, 0, ColorDepth.TRUECOLOR)
    assert enc.encode(f.copy(), 0, 0, ColorDepth.TRUECOLOR) == b""


def test_incremental_updates_track_the_screen():
    enc = CellEncoder()
    scr = Screen(16, 48)
    f = random_frame(16, 48, seed=1)
    scr.feed(enc.encode(f, 0, 0, ColorDepth.TRUECOLOR))
    rng = np.random.default_rng(5)
    for step in range(6):
        g = f.copy()
        mask = rng.random((16, 48)) < 0.1
        g.cps[mask] = 0x2586
        g.fg[mask] = rng.integers(0, 256, (mask.sum(), 3), dtype=np.uint8)
        data = enc.encode(g, 0, 0, ColorDepth.TRUECOLOR)
        assert len(data) < 16 * 48 * 20  # far less than a full redraw
        scr.feed(data)
        assert (scr.cps == g.cps).all()
        assert (scr.fg == g.fg).all()
        assert (scr.bg == g.bg).all()
        f = g


def test_tolerance_bounds_error():
    enc = CellEncoder()
    scr = Screen(8, 20)
    f = random_frame(8, 20, seed=3)
    scr.feed(enc.encode(f, 0, 0, ColorDepth.TRUECOLOR, tolerance=4))
    g = f.copy()
    g.fg = np.clip(g.fg.astype(int) + 3, 0, 255).astype(np.uint8)   # small drift: skipped
    assert enc.encode(g, 0, 0, ColorDepth.TRUECOLOR, tolerance=4) == b""
    h = g.copy()
    h.fg = np.clip(h.fg.astype(int) + 3, 0, 255).astype(np.uint8)   # accumulated drift: redrawn
    scr.feed(enc.encode(h, 0, 0, ColorDepth.TRUECOLOR, tolerance=4))
    assert np.abs(scr.fg - h.fg).max() <= 4


def test_offset_and_wide_characters():
    cps = np.array([[0x41, 0x4E2D, CONTINUATION, 0x42]], np.int32)
    fg = np.full((1, 4, 3), 255, np.uint8)
    bg = np.zeros((1, 4, 3), np.uint8)
    data = CellEncoder().encode(CellFrame(cps, fg, bg), 3, 5, ColorDepth.TRUECOLOR)
    assert data.startswith(b"\x1b[0m\x1b[4;6H")
    text = data.decode("utf-8")
    assert "A中" in text and text.index("B") > text.index("中")


@pytest.mark.parametrize("depth", [ColorDepth.ANSI256, ColorDepth.ANSI16])
def test_palette_modes_emit_valid_indices(depth):
    f = random_frame(6, 20, seed=7)
    scr = Screen(6, 20)
    scr.feed(CellEncoder().encode(f, 0, 0, depth))
    assert (scr.cps == f.cps).all()
    if depth == ColorDepth.ANSI256:
        assert ((scr.fg_idx >= 16) & (scr.fg_idx <= 255)).all()
    else:
        assert np.isin(scr.fg_idx, list(range(30, 38)) + list(range(90, 98))).all()


def test_mono_has_no_colors():
    f = random_frame(4, 10)
    data = CellEncoder().encode(f, 0, 0, ColorDepth.MONO)
    assert b"38;" not in data and b"48;" not in data


def test_invalidate_region_forces_redraw():
    enc = CellEncoder()
    f = random_frame(6, 12)
    enc.encode(f, 0, 0, ColorDepth.TRUECOLOR)
    enc.invalidate_region(2, 4)
    scr = Screen(6, 12)
    scr.feed(enc.encode(f, 0, 0, ColorDepth.TRUECOLOR))
    assert (scr.cps[2:4] == f.cps[2:4]).all()
    assert (scr.cps[:2] == 32).all()
