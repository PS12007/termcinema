import numpy as np

from termcinema import colorspace


def test_grayscale_removes_hue():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 200  # pure-ish red
    out = colorspace.apply_theme(frame, colorspace.Theme.GRAYSCALE)
    assert np.all(out[..., 0] == out[..., 1])
    assert np.all(out[..., 1] == out[..., 2])


def test_normal_theme_is_identity():
    frame = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    out = colorspace.apply_theme(frame, colorspace.Theme.NORMAL)
    assert np.array_equal(frame, out)


def test_sepia_shifts_toward_warm_tones():
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = colorspace.apply_theme(frame, colorspace.Theme.SEPIA).astype(int)
    # Warm highlight should have more red than blue.
    assert out[0, 0, 0] >= out[0, 0, 2]


def test_quantize_256_stays_in_range():
    frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    idx = colorspace.quantize(frame, bits256=True)
    assert idx.min() >= 0 and idx.max() <= 255


def test_quantize_16_stays_in_range():
    frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    idx = colorspace.quantize(frame, bits256=False)
    assert idx.min() >= 0 and idx.max() <= 15


def test_quantize_black_and_white_are_distinct():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [255, 255, 255]
    idx = colorspace.quantize(frame, bits256=True)
    assert idx[0, 0] != idx[1, 1]


def test_quantize_with_color_dither_stays_in_range():
    frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    idx = colorspace.quantize(frame, bits256=True, dither=True)
    assert idx.min() >= 0 and idx.max() <= 255
    idx16 = colorspace.quantize(frame, bits256=False, dither=True)
    assert idx16.min() >= 0 and idx16.max() <= 15


def test_redmean_palette_prefers_closer_color():
    # A pure red pixel should map to a palette entry that is red-ish,
    # not e.g. blue, confirming the perceptual LUT isn't scrambled.
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]
    idx = colorspace.quantize(frame, bits256=True)[0, 0]
    r, g, b = colorspace.PALETTE_256[idx]
    assert int(r) > int(g) and int(r) > int(b)
