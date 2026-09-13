"""Local video/GIF decoding via OpenCV (which itself shells out to
FFmpeg for demuxing/decoding), with hardware acceleration requested
when the underlying FFmpeg build supports it.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from . import scaler


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


class VideoSource:
    def __init__(self, path_or_url: str, hw_accel: bool = True, use_gpu: bool = True):
        self.path = path_or_url
        self.use_gpu = use_gpu
        cap = None
        if hw_accel:
            try:
                cap = cv2.VideoCapture(path_or_url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
                ])
            except Exception:
                cap = None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(path_or_url)
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {path_or_url}")
        self.cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0 or fps > 240:
            fps = 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (count / fps) if (count and fps) else 0.0

        self.info = VideoInfo(width=width, height=height, fps=fps,
                               frame_count=count, duration=duration)

    def read_rgb(self) -> Optional[np.ndarray]:
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None
        return scaler.bgr_to_rgb(frame_bgr, use_gpu=self.use_gpu)

    def seek_seconds(self, t: float) -> None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000.0)

    def seek_frame(self, n: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(n, 0))

    def current_pos_seconds(self) -> float:
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def release(self) -> None:
        self.cap.release()

    def is_seekable(self) -> bool:
        return self.info.frame_count > 0
