from termcinema import scaler


def test_grid_size_fits_within_bounds():
    cols, rows = scaler.compute_grid_size(1920, 1080, 100, 40, font_aspect=2.0)
    assert 1 <= cols <= 100
    assert 1 <= rows <= 40


def test_grid_size_preserves_aspect_roughly():
    # A 2:1 (w:h) source, with a 2:1 font aspect, should map to
    # roughly square-looking output in "physical" terms, i.e.
    # cols / (rows * font_aspect) ~= src_w / src_h.
    src_w, src_h = 200, 100
    font_aspect = 2.0
    cols, rows = scaler.compute_grid_size(src_w, src_h, 500, 500, font_aspect)
    physical_ratio = cols / (rows * font_aspect)
    src_ratio = src_w / src_h
    assert abs(physical_ratio - src_ratio) / src_ratio < 0.1


def test_target_pixel_size_scales_by_packing():
    cols, rows = 40, 20
    pw, ph = scaler.target_pixel_size(cols, rows, cols_per_cell=2, rows_per_cell=4)
    assert pw == 80
    assert ph == 80


def test_narrow_terminal_constrains_cols():
    cols, rows = scaler.compute_grid_size(1920, 1080, 20, 1000, font_aspect=2.0)
    assert cols <= 20
