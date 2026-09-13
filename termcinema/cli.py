from __future__ import annotations

import argparse
import sys

from . import config as config_mod
from . import renderer
from .benchmark import run_benchmark
from .player import Player
from .terminal import detect_caps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="termcinema",
        description="Play videos, GIFs, and YouTube links as colored text in your terminal.",
    )
    p.add_argument("source", nargs="?", help="Path to a local video/GIF, or a URL (YouTube, etc.)")
    p.add_argument("--mode", choices=list(renderer.MODES.keys()), help="Rendering mode")
    p.add_argument("--color", choices=["auto", "truecolor", "256", "16", "mono"], help="Color depth")
    p.add_argument("--theme", help="Color theme (normal, grayscale, sepia, green, amber, matrix, dracula, nord, gruvbox, catppuccin, tokyonight, monochrome)")
    p.add_argument("--fps", type=float, help="Target playback FPS (default: source FPS)")
    p.add_argument("--width", type=int, help="Force terminal columns to use")
    p.add_argument("--height", type=int, help="Force terminal rows to use")
    p.add_argument("--no-audio", action="store_true", help="Disable audio playback")
    p.add_argument("--no-hw-accel", action="store_true", help="Disable hardware-accelerated decoding")
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU (CUDA) acceleration even if available")
    p.add_argument("--no-edge", action="store_true", help="Disable edge-aware ASCII glyph selection")
    p.add_argument("--no-dither", action="store_true", help="Disable shape dithering (braille/quarterblock modes)")
    p.add_argument("--color-dither", action="store_true", help="Enable ordered dithering before 256/16-color quantization (reduces banding)")
    p.add_argument("--quality", default=None, help="yt-dlp format selector shortcut: best|worst|audio")
    p.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (no real-time playback)")
    p.add_argument("--frames", type=int, default=150, help="Frames to measure in --benchmark mode")
    p.add_argument("--info", action="store_true", help="Print terminal capability detection and exit")
    p.add_argument("--list-modes", action="store_true", help="List available rendering modes and exit")
    p.add_argument("--save-config", action="store_true", help="Save the resolved settings as the new default config")
    p.add_argument("-y", "--yes", action="store_true",
                    help="Skip the interactive settings wizard and play immediately with flags/config defaults")
    p.add_argument("--no-wizard", action="store_true", help="Alias for --yes")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_modes:
        for name, m in renderer.MODES.items():
            print(f"  {name:15s} {m.rows_per_cell}x{m.cols_per_cell} src px/cell")
        return 0

    if args.info:
        caps = detect_caps()
        print(f"Terminal size:   {caps.columns}x{caps.rows}")
        print(f"Color support:   {caps.color.name}")
        print(f"Unicode support: {caps.unicode_ok}")
        from . import scaler
        gpu = scaler.gpu_status()
        if gpu["cuda_available"]:
            print(f"GPU:             {gpu['device']} (resize={gpu['resize']}, color_convert={gpu['color_convert']}, sobel={gpu['sobel']})")
        else:
            print("GPU:             not available (CPU only)")
        return 0

    if not args.source:
        build_parser().print_help()
        return 1

    cfg = config_mod.load()
    if args.mode:
        cfg.mode = args.mode
    if args.color:
        cfg.color = args.color
    if args.theme:
        cfg.theme = args.theme
    if args.fps is not None:
        cfg.fps = args.fps
    if args.no_audio:
        cfg.audio = False
    if args.no_hw_accel:
        cfg.hw_accel = False
    if args.no_gpu:
        cfg.gpu = False
    if args.no_edge:
        cfg.edge_aware = False
    if args.no_dither:
        cfg.dither = False
    if args.color_dither:
        cfg.color_dither = True
    if args.quality:
        cfg.quality = args.quality

    valid_themes = {t.value for t in __import__("termcinema.colorspace", fromlist=["Theme"]).Theme}
    if cfg.theme not in valid_themes:
        print(f"Unknown theme '{cfg.theme}'. Valid options: {', '.join(sorted(valid_themes))}", file=sys.stderr)
        return 1

    if args.benchmark:
        run_benchmark(args.source, mode=cfg.mode, frames=args.frames, hw_accel=cfg.hw_accel, use_gpu=cfg.gpu)
        return 0

    skip_wizard = args.yes or args.no_wizard
    if not skip_wizard and sys.stdin.isatty() and sys.stdout.isatty():
        from . import wizard
        from . import youtube
        cfg = wizard.run_wizard(cfg, detect_caps(), is_url=youtube.is_url(args.source))

    if args.save_config:
        config_mod.save(cfg)
        print(f"Saved config to {config_mod.CONFIG_PATH}")

    player = Player(args.source, cfg, forced_width=args.width, forced_height=args.height)
    try:
        player.run()
    except KeyboardInterrupt:
        pass
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
