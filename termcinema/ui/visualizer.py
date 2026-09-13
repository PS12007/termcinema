"""Audio visualizers for audio-only media.

Visualizers draw ordinary RGB frames, so they go through the same
renderers as video: spectrum bars look great in octant mode and even
better as sixel/kitty pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

STYLES = ("spectrum", "mirror", "wave", "radial")


def _gradient(n: int, t: float) -> np.ndarray:
    """Hue sweep palette that slowly drifts over time. (n, 3) float32."""
    hue = ((np.linspace(0, 150, n) + t * 12) % 180).astype(np.uint8)
    hsv = np.stack([hue, np.full(n, 220, np.uint8), np.full(n, 255, np.uint8)], -1)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(np.float32)


class Visualizer:
    def __init__(self):
        self.style = STYLES[0]
        self._levels = None
        self._peaks = None
        self.background = None  # optional RGB image (e.g. blurred thumbnail)

    def next_style(self) -> str:
        self.style = STYLES[(STYLES.index(self.style) + 1) % len(STYLES)]
        return self.style

    def _spectrum(self, samples: np.ndarray, bands: int) -> np.ndarray:
        n = samples.size
        win = np.hanning(n).astype(np.float32)
        mag = np.abs(np.fft.rfft(samples * win))[1:]
        freqs = np.fft.rfftfreq(n, 1 / 48000.0)[1:]
        edges = np.geomspace(35, 16000, bands + 1)
        idx = np.searchsorted(freqs, edges)
        out = np.zeros(bands, np.float32)
        for i in range(bands):
            a, b = idx[i], max(idx[i + 1], idx[i] + 1)
            out[i] = mag[a:b].mean() if b <= mag.size else 0.0
        db = 20 * np.log10(out + 1e-6)
        lvl = np.clip((db + 10) / 55.0, 0, 1)
        # tilt: highs carry less energy, lift them a bit
        lvl *= np.linspace(0.85, 1.25, bands)
        return np.clip(lvl, 0, 1)

    def _smooth(self, lvl: np.ndarray) -> np.ndarray:
        if self._levels is None or self._levels.shape != lvl.shape:
            self._levels = lvl.copy()
            self._peaks = lvl.copy()
        up = lvl > self._levels
        self._levels = np.where(up, self._levels * 0.35 + lvl * 0.65, self._levels * 0.82 + lvl * 0.18)
        self._peaks = np.maximum(self._peaks - 0.012, self._levels)
        return self._levels

    def _canvas(self, w: int, h: int, t: float) -> np.ndarray:
        if self.background is not None:
            bg = cv2.resize(self.background, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) * 0.33
        else:
            yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            top = np.array([10, 8, 24], np.float32)
            bot = np.array([2, 2, 6], np.float32)
            bg = np.broadcast_to(top * (1 - yy) + bot * yy, (h, w, 3)).copy()
        return bg

    def frame(self, samples: np.ndarray, w: int, h: int, t: float) -> np.ndarray:
        w, h = max(w, 16), max(h, 16)
        img = self._canvas(w, h, t)
        if self.style == "wave":
            self._wave(img, samples, t)
        elif self.style == "radial":
            self._radial(img, samples, t)
        else:
            self._bars(img, samples, t, mirror=self.style == "mirror")
        return np.clip(img, 0, 255).astype(np.uint8)

    def _bars(self, img, samples, t, mirror=False):
        h, w = img.shape[:2]
        bands = int(np.clip(w // 6, 12, 96))
        lvl = self._smooth(self._spectrum(samples, bands))
        colors = _gradient(bands, t)
        slot = w / bands
        bar_w = max(1, int(slot * 0.72))
        base = h // 2 if mirror else int(h * 0.86)
        max_h = (h // 2 - 2) if mirror else int(h * 0.80)
        ys = np.arange(h)[:, None]
        xs = np.arange(w)[None, :]
        band_of_x = np.minimum((xs / slot).astype(np.int32), bands - 1)
        in_bar = (xs - band_of_x * slot) < bar_w
        heights = (lvl * max_h).astype(np.int32)[band_of_x]
        col = np.broadcast_to(colors[band_of_x[0]][None, :, :], (h, w, 3))  # (h, w, 3)
        up = (ys <= base) & (ys > base - heights) & in_bar
        # brighten toward the bar tips
        frac = np.clip((base - ys) / np.maximum(heights, 1), 0, 1)
        shade = (0.55 + 0.45 * frac)[..., None]
        img[up] = (col * shade)[up]
        if mirror:
            down = (ys >= base) & (ys < base + heights) & in_bar
            img[down] = (col * 0.75)[down]
        else:
            refl = (ys > base) & (ys < base + heights * 0.35) & in_bar
            img[refl] = img[refl] * 0.4 + (col * 0.35)[refl]
            peaks = (self._peaks * max_h).astype(np.int32)[band_of_x]
            pk = (ys == base - peaks - 2) & in_bar & ((self._peaks > 0.03)[band_of_x])
            img[pk] = 255

    def _wave(self, img, samples, t):
        h, w = img.shape[:2]
        s = samples[-min(samples.size, w * 4):]
        xs = np.linspace(0, s.size - 1, w).astype(np.int32)
        ys = (h / 2 - s[xs] * h * 0.42).astype(np.int32)
        pts = np.stack([np.arange(w), np.clip(ys, 0, h - 1)], -1).reshape(-1, 1, 2)
        glow = np.zeros((h, w), np.uint8)
        cv2.polylines(glow, [pts], False, 255, max(1, h // 60), cv2.LINE_AA)
        blur = cv2.GaussianBlur(glow, (0, 0), max(1.5, h / 60))
        color = _gradient(w, t)[None, :, :] / 255.0
        img += color * glow[..., None] + color * blur[..., None] * 1.6

    def _radial(self, img, samples, t):
        h, w = img.shape[:2]
        bands = 64
        lvl = self._smooth(self._spectrum(samples, bands))
        cx, cy = w / 2, h / 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)
        r = np.hypot((xx - cx) / (w / 2), (yy - cy) / (h / 2))
        b = np.minimum((ang * bands).astype(np.int32), bands - 1)
        # symmetric
        b = np.where(b >= bands // 2, bands - 1 - b, b) * 2 % bands
        r0 = 0.28 + 0.04 * float(lvl[:4].mean())
        reach = r0 + lvl[b] * 0.62
        mask = (r > r0) & (r < reach)
        colors = _gradient(bands, t)
        img[mask] = colors[b[mask]] * (1.1 - (r[mask] - r0) / 0.9)[:, None]
        ring = np.abs(r - r0) < 0.012
        img[ring] = 230
