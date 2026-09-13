"""Resize video frames to fit the terminal grid without distortion.

Terminal character cells are taller than they are wide (roughly 2:1 in
most monospace fonts), so naively resizing a frame to `columns x rows`
squashes the image horizontally. We correct for that here, and also
support GPU-accelerated resizing via OpenCV's CUDA module when it is
available, falling back to CPU transparently.
"""
from __future__ import annotations

import numpy as np
import cv2

_CUDA_OK = False
try:  # pragma: no cover - depends on the machine's OpenCV build
    _CUDA_OK = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    _CUDA_OK = False

_CUDA_STREAM = None
if _CUDA_OK:
    try:
        _CUDA_STREAM = cv2.cuda.Stream()
    except Exception:
        _CUDA_STREAM = None


def gpu_available() -> bool:
    return _CUDA_OK


def gpu_status() -> dict:
    """Report what's actually GPU-accelerated on this machine, for
    --info / the settings wizard / the status bar."""
    info = {"cuda_available": _CUDA_OK, "device": None, "resize": False,
            "color_convert": False, "sobel": False}
    if not _CUDA_OK:
        return info
    try:
        dev = cv2.cuda.DeviceInfo(cv2.cuda.getDevice())
        info["device"] = dev.name()
    except Exception:
        info["device"] = "CUDA device"
    info["resize"] = True
    info["color_convert"] = True
    try:
        cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0)
        info["sobel"] = True
    except Exception:
        info["sobel"] = False
    return info


def compute_grid_size(
    frame_w: int,
    frame_h: int,
    max_cols: int,
    max_rows: int,
    font_aspect: float,
) -> tuple[int, int]:
    """Compute (cols, rows) of terminal cells that best fit the frame
    while preserving the source's on-screen aspect ratio.

    `font_aspect` is the physical height:width ratio of one terminal
    glyph cell (e.g. ~2.0). It's the only thing that affects the aspect
    math -- how many source pixels a renderer later packs into that one
    cell (rows_per_cell/cols_per_cell) is a resolution/detail choice
    handled separately by `target_pixel_size`, not an aspect concern.
    """
    src_aspect = frame_w / frame_h  # width:height of the source

    cols = max_cols
    rows = int(round(cols / src_aspect / font_aspect))
    if rows > max_rows:
        rows = max_rows
        cols = int(round(rows * src_aspect * font_aspect))
    cols = max(1, min(cols, max_cols))
    rows = max(1, min(rows, max_rows))
    return cols, rows


def target_pixel_size(cols: int, rows: int, cols_per_cell: int, rows_per_cell: int) -> tuple[int, int]:
    """Pixel dimensions to resize the source frame to before handing it
    to a renderer, given how many source pixels that renderer packs
    into each terminal cell."""
    return cols * cols_per_cell, rows * rows_per_cell


def resize_frame(frame_bgr: np.ndarray, target_w: int, target_h: int, use_gpu: bool = True) -> np.ndarray:
    """Resize a BGR frame to exactly (target_w, target_h) pixels.

    Uses area-averaging interpolation when downscaling (the accurate
    choice: it averages every source pixel that lands in each output
    cell, instead of discarding most of them like nearest/bilinear
    would) and linear interpolation when upscaling.
    """
    interp = cv2.INTER_AREA if target_w < frame_bgr.shape[1] else cv2.INTER_LINEAR
    if use_gpu and _CUDA_OK:
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(frame_bgr)
            resized = cv2.cuda.resize(gpu_mat, (target_w, target_h), interpolation=interp)
            return resized.download()
        except Exception:
            pass  # fall through to CPU
    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=interp)


def bgr_to_rgb(frame_bgr: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """Color-convert BGR -> RGB, GPU-accelerated for large frames when
    available. For most frame sizes a NumPy channel-reverse view is
    actually just as fast (it's O(1), no copy) -- this exists mainly so
    4K+ sources get a real accelerated conversion path end-to-end
    alongside the GPU resize."""
    if use_gpu and _CUDA_OK:
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(frame_bgr)
            converted = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2RGB)
            return converted.download()
        except Exception:
            pass
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def sobel_gpu(gray: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """GPU Sobel gradients (gx, gy) as float32, or None if unavailable
    so callers can fall back to cv2.Sobel on CPU."""
    if not _CUDA_OK:
        return None
    try:
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(gray)
        fx = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)
        gx = fx.apply(gpu_mat).download()
        gy = fy.apply(gpu_mat).download()
        return gx, gy
    except Exception:
        return None
