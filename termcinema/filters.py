"""Picture adjustments and visual effects, applied at render resolution.

Everything operates on (H, W, 3) uint8 RGB arrays with vectorized NumPy /
OpenCV calls, so even stacked effects cost a millisecond or two.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Adjust:
    brightness: float = 0.0   # -1 .. 1
    contrast: float = 1.0     # 0 .. 3
    saturation: float = 1.0   # 0 .. 3
    gamma: float = 1.0        # 0.2 .. 3
    sharpen: float = 0.0      # 0 .. 2

    def is_identity(self) -> bool:
        return (abs(self.brightness) < 1e-3 and abs(self.contrast - 1) < 1e-3 and
                abs(self.saturation - 1) < 1e-3 and abs(self.gamma - 1) < 1e-3 and self.sharpen < 1e-3)

    def describe(self) -> str:
        return (f"bright {self.brightness:+.2f}  contrast {self.contrast:.2f}  "
                f"sat {self.saturation:.2f}  gamma {self.gamma:.2f}  sharpen {self.sharpen:.2f}")


_gamma_cache: dict = {}


def _gamma_lut(g: float) -> np.ndarray:
    key = round(g, 3)
    lut = _gamma_cache.get(key)
    if lut is None:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / key)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        _gamma_cache[key] = lut
    return lut


def apply_adjust(img: np.ndarray, a: Adjust) -> np.ndarray:
    if a.is_identity():
        return img
    out = img
    if a.sharpen > 1e-3:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.0 + a.sharpen, blur, -a.sharpen, 0)
    if abs(a.contrast - 1) > 1e-3 or abs(a.brightness) > 1e-3 or abs(a.saturation - 1) > 1e-3:
        f = out.astype(np.float32)
        if abs(a.saturation - 1) > 1e-3:
            gray = f @ np.array([0.299, 0.587, 0.114], np.float32)
            f = gray[..., None] + (f - gray[..., None]) * a.saturation
        if abs(a.contrast - 1) > 1e-3:
            f = (f - 128.0) * a.contrast + 128.0
        if abs(a.brightness) > 1e-3:
            f = f + a.brightness * 255.0
        out = np.clip(f, 0, 255).astype(np.uint8)
    if abs(a.gamma - 1) > 1e-3:
        out = cv2.LUT(out, _gamma_lut(a.gamma))
    return out


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _luma(img):
    return (img.astype(np.float32) @ _LUMA) / 255.0


def _duotone(shadow, highlight, mid=None):
    s = np.array(shadow, np.float32)
    h = np.array(highlight, np.float32)
    m = np.array(mid, np.float32) if mid is not None else None

    def fx(img, t):
        l = _luma(img)[..., None]
        if m is None:
            out = s + (h - s) * l
        else:
            out = np.where(l < 0.5, s + (m - s) * (l * 2), m + (h - m) * (l * 2 - 1))
        return np.clip(out, 0, 255).astype(np.uint8)
    return fx


def _grayscale(img, t):
    l = (_luma(img) * 255).astype(np.uint8)
    return np.repeat(l[..., None], 3, axis=2)


def _invert(img, t):
    return 255 - img


def _posterize(img, t):
    return ((img // 64) * 85).astype(np.uint8)


def _thermal(img, t):
    l = (_luma(img) * 255).astype(np.uint8)
    return cv2.applyColorMap(l, cv2.COLORMAP_INFERNO)[:, :, ::-1]


def _nightvision(img, t):
    l = _luma(img)
    h, w = l.shape
    rng = np.random.default_rng(int(t * 30))
    noise = rng.normal(0, 0.06, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vign = 1.0 - 0.7 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    v = np.clip((l * 1.6 + noise) * np.clip(vign, 0, 1), 0, 1)
    out = np.stack([v * 60, v * 255, v * 80], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _neon(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.clip(cv2.magnitude(gx, gy) / 255.0, 0, 1)
    glow = cv2.GaussianBlur(mag, (0, 0), 2.0)
    e = np.clip(mag * 1.5 + glow * 1.2, 0, 1)[..., None]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    base = img.astype(np.float32) * 0.12
    return np.clip(base + vivid * e, 0, 255).astype(np.uint8)


def _crt(img, t):
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    scan = np.ones(h, np.float32)
    scan[1::2] = 0.62
    f *= scan[:, None, None]
    # slight chroma bleed
    f[:, 1:, 0] = f[:, 1:, 0] * 0.7 + f[:, :-1, 0] * 0.3
    f[:, :-1, 2] = f[:, :-1, 2] * 0.7 + f[:, 1:, 2] * 0.3
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vign = 1.0 - 0.35 * (((xx - w / 2) / (w / 2)) ** 4 + ((yy - h / 2) / (h / 2)) ** 4)
    f *= vign[..., None]
    return np.clip(f * 1.15, 0, 255).astype(np.uint8)


def _vhs(img, t):
    h, w = img.shape[:2]
    shift = max(1, w // 160)
    out = img.copy()
    out[:, shift:, 0] = img[:, :-shift, 0]
    out[:, :-shift, 2] = img[:, shift:, 2]
    rng = np.random.default_rng(int(t * 24))
    f = out.astype(np.float32)
    f += rng.normal(0, 7, (h, 1, 1)).astype(np.float32)
    # a rolling tracking band
    band = int((t * 0.25 % 1.0) * h)
    f[band:band + max(1, h // 30)] *= 1.35
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = (hsv[..., 1] * 0.75).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _pixelate(img, t):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 6), max(1, h // 6)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _vivid(img, t):
    return apply_adjust(img, Adjust(contrast=1.12, saturation=1.45, sharpen=0.4))


def _cartoon(img, t):
    smooth = cv2.bilateralFilter(img, 7, 60, 7)
    q = ((smooth.astype(np.int32) // 40) * 40 + 20).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
    return (q * (edges[..., None] > 0)).astype(np.uint8)


EFFECTS = {
    "none": None,
    "vivid": _vivid,
    "grayscale": _grayscale,
    "sepia": _duotone((38, 24, 12), (255, 232, 185), (160, 110, 60)),
    "matrix": _duotone((0, 8, 2), (170, 255, 170), (0, 170, 40)),
    "amber": _duotone((18, 6, 0), (255, 196, 60), (220, 120, 0)),
    "cyberpunk": _duotone((20, 0, 40), (0, 255, 240), (255, 0, 160)),
    "synthwave": _duotone((25, 5, 50), (255, 220, 120), (240, 40, 140)),
    "nord": _duotone((46, 52, 64), (236, 239, 244), (136, 192, 208)),
    "dracula": _duotone((40, 42, 54), (248, 248, 242), (189, 147, 249)),
    "gruvbox": _duotone((29, 32, 33), (251, 241, 199), (215, 153, 33)),
    "catppuccin": _duotone((30, 30, 46), (245, 224, 220), (203, 166, 247)),
    "thermal": _thermal,
    "nightvision": _nightvision,
    "neon": _neon,
    "crt": _crt,
    "vhs": _vhs,
    "cartoon": _cartoon,
    "posterize": _posterize,
    "pixelate": _pixelate,
    "invert": _invert,
}


def apply_effect(img: np.ndarray, name: str, t: float = 0.0) -> np.ndarray:
    fx = EFFECTS.get(name)
    if fx is None:
        return img
    return fx(img, t)
