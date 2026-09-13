from __future__ import annotations

import time

from . import renderer, scaler, terminal
from .video import VideoSource


def run_benchmark(path: str, mode: str = "halfblock", frames: int = 150, hw_accel: bool = True, use_gpu: bool = True) -> None:
    video = VideoSource(path, hw_accel=hw_accel, use_gpu=use_gpu)
    caps = terminal.detect_caps()
    m = renderer.MODES[mode]
    cols, rows = scaler.compute_grid_size(video.info.width, video.info.height, caps.columns, caps.rows, caps.font_aspect)
    pw, ph = scaler.target_pixel_size(cols, rows, m.cols_per_cell, m.rows_per_cell)

    decode_times = []
    render_times = []
    n = 0
    t_total0 = time.perf_counter()
    while n < frames:
        t0 = time.perf_counter()
        frame = video.read_rgb()
        if frame is None:
            break
        decode_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        resized = scaler.resize_frame(frame[:, :, ::-1], pw, ph, use_gpu=use_gpu)[:, :, ::-1]
        renderer.render_frame(mode, resized, caps.color)
        render_times.append(time.perf_counter() - t1)
        n += 1
    total = time.perf_counter() - t_total0
    video.release()

    if n == 0:
        print("No frames decoded.")
        return

    avg_decode = sum(decode_times) / n
    avg_render = sum(render_times) / n
    gpu = scaler.gpu_status()
    print(f"TermCinema benchmark: {path}")
    print(f"  Source: {video.info.width}x{video.info.height} @ {video.info.fps:.2f} fps")
    print(f"  Terminal grid: {cols}x{rows} cells  (mode={mode}, color={caps.color.name})")
    print(f"  Frames measured: {n}")
    print(f"  Avg decode time:  {avg_decode * 1000:.2f} ms/frame  ({1/avg_decode:.1f} fps)")
    print(f"  Avg render time:  {avg_render * 1000:.2f} ms/frame  ({1/avg_render:.1f} fps)")
    print(f"  Combined:         {(avg_decode + avg_render) * 1000:.2f} ms/frame  ({1/(avg_decode+avg_render):.1f} fps)")
    print(f"  Wall time for {n} frames: {total:.2f}s  ({n/total:.1f} fps sustained)")
    if gpu["cuda_available"]:
        print(f"  GPU: {gpu['device']}  (resize={gpu['resize']}, color_convert={gpu['color_convert']}, sobel={gpu['sobel']})")
    else:
        print("  GPU: not available (running on CPU)")
