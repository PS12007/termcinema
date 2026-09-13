import numpy as np
import pytest

from termcinema import render
from termcinema.render import glyphs, kernels, palette, pixels, text

from sixel_decode import decode as sixel_decode


def _edge_image(h, w):
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img[xx > yy * (w / h) * 0.8] = (240, 60, 30)
    img[..., 2] = np.maximum(img[..., 2], (yy * 255 // max(h - 1, 1)).astype(np.uint8))
    return img


def test_glyph_tables_are_complete_and_unique():
    assert len(set(glyphs.OCTANT_CODEPOINTS)) == 256
    assert len(set(glyphs.SEXTANT_CODEPOINTS)) == 64
    assert glyphs.OCTANT_CODEPOINTS[0b00001111] == 0x2580     # top half
    assert glyphs.OCTANT_CODEPOINTS[0b01010101] == 0x258C     # left half
    assert glyphs.SEXTANT_CODEPOINTS[0b000011] == 0x1FB02     # BLOCK SEXTANT-12
    assert glyphs.BRAILLE_CODEPOINTS[255] == 0x28FF


@pytest.mark.parametrize("name", render.TEXT_MODES)
def test_text_renderers_shape_and_types(name):
    r = render.get(name)
    rows, cols = 7, 11
    px = _edge_image(rows * r.gh, cols * r.gw)
    cf = r.render(px)
    assert cf.cps.shape == (rows, cols)
    assert cf.fg.shape == (rows, cols, 3) and cf.fg.dtype == np.uint8
    assert cf.cps.dtype == np.int32
    assert (cf.cps > 0).all()


@pytest.mark.parametrize("name", ["blocks", "sextant", "quadrant", "octant"])
def test_flat_image_becomes_plain_background(name):
    r = render.get(name)
    px = np.full((4 * r.gh, 6 * r.gw, 3), (12, 200, 99), np.uint8)
    cf = r.render(px)
    assert (cf.cps == 0x20).all()
    assert (np.abs(cf.bg.astype(int) - (12, 200, 99)) <= 1).all()


def test_block_fit_reproduces_a_two_color_cell_exactly():
    r = render.get("blocks")
    cell = np.zeros((4, 2, 3), np.uint8)
    cell[:2] = (255, 0, 0)       # top half red
    cell[2:] = (0, 0, 255)       # bottom half blue
    cf = r.render(cell)
    assert cf.cps[0, 0] == 0x2580
    assert tuple(cf.fg[0, 0]) == (255, 0, 0)
    assert tuple(cf.bg[0, 0]) == (0, 0, 255)


def test_octant_picks_exact_pattern():
    r = render.get("octant")
    cell = np.zeros((4, 2, 3), np.uint8)
    cell[0, 0] = cell[1, 1] = (255, 255, 255)   # sub-pixels 0 and 3
    cf = r.render(cell)
    assert cf.cps[0, 0] == glyphs.OCTANT_CODEPOINTS[0b1001]
    assert tuple(cf.fg[0, 0]) == (255, 255, 255)
    assert tuple(cf.bg[0, 0]) == (0, 0, 0)


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name", ["octant", "braille"])
def test_numba_matches_numpy(name):
    r = render.get(name)
    px = _edge_image(20 * 4, 30 * 2)
    a, b = r.render(px), r.render_numpy(px)
    assert (a.cps == b.cps).all()
    assert np.abs(a.fg.astype(int) - b.fg.astype(int)).max() <= 1


def test_sixel_roundtrip():
    img = _edge_image(66, 90)
    out = pixels.Sixel().render(img, 0, 0, 9, 3)
    idx = sixel_decode(out)
    assert idx.shape == (66, 90)
    assert (idx == pixels._quantize_np(img, pixels._BAYER8)).all()


def test_sixel_numpy_fallback_roundtrip():
    img = _edge_image(30, 40)
    q = pixels._quantize_np(img, pixels._BAYER8)
    body = pixels._sixel_body_np(q)
    data = b'\x1bP0;1;0q"1;1;40;30' + body + b"\x1b\\"
    assert (sixel_decode(data) == q).all()


def test_palette_quantization_prefers_close_colors():
    rgb = np.array([[[255, 0, 0], [0, 0, 0], [255, 255, 255], [128, 128, 128]]], np.uint8)
    idx = palette.quantize(rgb, 256)
    pal = palette.PALETTE_256[idx[0]]
    assert np.abs(pal.astype(int) - rgb[0].astype(int)).max(1).tolist()[:3] == [0, 0, 0]
    assert abs(int(pal[3, 0]) - 128) <= 10
