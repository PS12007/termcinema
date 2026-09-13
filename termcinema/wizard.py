"""Interactive settings prompt shown before playback starts (unless
skipped with --yes). Plain input()-based menus so it works in any
terminal without extra dependencies -- press Enter to accept the
highlighted default at every step.
"""
from __future__ import annotations

import sys

from . import colorspace, renderer, scaler
from .config import Config
from .terminal import ColorSupport, TerminalCaps

_COLOR_CHOICES = [
    ("auto", "Auto-detect (recommended)"),
    ("truecolor", "TrueColor -- 16.7M colors, most accurate"),
    ("256", "256-color -- widely supported, slight banding"),
    ("16", "16-color -- max compatibility, noticeable banding"),
    ("mono", "No color -- plain glyphs only"),
]

_MODE_DESCRIPTIONS = {
    "ascii": "Classic ASCII ramp, edge-aware outlines",
    "ascii-extended": "Block-shading ASCII ramp (░▒▓█)",
    "ascii-unicode": "Dense Unicode ramp",
    "halfblock": "▀ half-blocks -- best quality/speed balance (recommended)",
    "quarterblock": "2x2 quadrant blocks -- sharper shapes, more color detail",
    "braille": "Braille dot-matrix -- highest spatial resolution",
}

_THEME_DESCRIPTIONS = {
    "normal": "True source colors",
    "grayscale": "Black & white",
    "monochrome": "Hard black/white threshold",
    "sepia": "Warm vintage tone",
    "green": "Retro green phosphor terminal",
    "amber": "Amber CRT",
    "matrix": "Matrix-style green",
    "dracula": "Dracula palette duotone",
    "nord": "Nord palette duotone",
    "gruvbox": "Gruvbox palette duotone",
    "catppuccin": "Catppuccin palette duotone",
    "tokyonight": "Tokyo Night palette duotone",
}

_PRESETS = {
    "1": ("fast", "Fast -- lower detail, maximum performance"),
    "2": ("balanced", "Balanced -- edge-aware + dithering, good performance"),
    "3": ("best", "Best quality -- TrueColor, all detail features on"),
}


def _ask(prompt: str, choices: list, default_idx: int = 0) -> str:
    """choices: list of (value, description). Returns the chosen value."""
    print(f"\n{prompt}")
    for i, (val, desc) in enumerate(choices, 1):
        marker = " (default)" if i - 1 == default_idx else ""
        print(f"  {i}. {desc}{marker}")
    raw = input(f"Choose [1-{len(choices)}] (Enter for default): ").strip()
    if not raw:
        return choices[default_idx][0]
    try:
        i = int(raw)
        if 1 <= i <= len(choices):
            return choices[i - 1][0]
    except ValueError:
        pass
    print("  (unrecognized input, using default)")
    return choices[default_idx][0]


def _ask_yesno(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def run_wizard(cfg: Config, caps: TerminalCaps, is_url: bool = False) -> Config:
    print("=" * 60)
    print("  TermCinema -- Playback Settings")
    print("=" * 60)
    print(f"Detected terminal: {caps.columns}x{caps.rows} cells, "
          f"color={caps.color.name}, unicode={'yes' if caps.unicode_ok else 'no'}")
    gpu = scaler.gpu_status()
    if gpu["cuda_available"]:
        print(f"GPU acceleration: available ({gpu['device']})")
    else:
        print("GPU acceleration: not available on this machine (using CPU)")
    print("(Press Enter at any prompt to accept the recommended default.)")

    preset_choices = [(v, d) for v, d in _PRESETS.values()]
    preset = _ask("Quality preset:", preset_choices, default_idx=1)
    if preset == "fast":
        cfg.mode, cfg.edge_aware, cfg.dither, cfg.color_dither = "halfblock", False, False, False
        cfg.color = "256"
    elif preset == "best":
        cfg.mode, cfg.edge_aware, cfg.dither, cfg.color_dither = "braille", True, True, True
        cfg.color = "truecolor"
    else:
        cfg.edge_aware, cfg.dither, cfg.color_dither = True, True, True

    mode_choices = [(name, _MODE_DESCRIPTIONS.get(name, name)) for name in renderer.MODES]
    default_idx = list(renderer.MODES.keys()).index(cfg.mode)
    cfg.mode = _ask("Rendering mode:", mode_choices, default_idx=default_idx)

    color_default_idx = 0 if preset != "best" else 1
    cfg.color = _ask("Color depth:", _COLOR_CHOICES, default_idx=color_default_idx)

    theme_choices = [(name, _THEME_DESCRIPTIONS.get(name, name)) for name in colorspace.Theme]
    cfg.theme = _ask("Color theme:", theme_choices, default_idx=0)

    if gpu["cuda_available"]:
        cfg.gpu = _ask_yesno("Use GPU acceleration?", default=True)
    else:
        cfg.gpu = False

    cfg.audio = _ask_yesno("Enable audio?", default=cfg.audio)

    if is_url:
        q_choices = [
            ("best", "Best available quality"),
            ("worst", "Lowest quality (fastest to start)"),
            ("audio", "Audio only"),
        ]
        cfg.quality = _ask("Stream quality:", q_choices, default_idx=0)

    print("\nStarting playback with these settings. (Use --yes next time to skip this menu.)")
    print("=" * 60)
    return cfg
