# TermCinema

Play local videos, GIFs, and YouTube links as colored text, right in your terminal.

```
termcinema movie.mp4
termcinema "https://youtube.com/watch?v=..."
termcinema --mode braille --color truecolor clip.gif
```

Run it with no flags and it now **asks you what settings to use** before
playing — quality preset, rendering mode, color depth, theme, GPU on/off,
audio — instead of just guessing and going. Pass `-y`/`--yes` to skip
straight to playback with your flags/config defaults (handy for scripts).

## Features

- **Interactive settings wizard** on startup: pick a quality preset
  (Fast / Balanced / Best), rendering mode, color depth, theme, GPU
  usage, and audio, with sensible defaults on every prompt (just hit
  Enter). Skip it anytime with `-y`.
- **6 rendering modes**, from classic ASCII to a custom quarter-block
  renderer and a full braille dot-matrix mode (8x the spatial resolution
  of one-pixel-per-character rendering):
  - `ascii`, `ascii-extended`, `ascii-unicode` — luminance-ramp text art,
    with **edge-aware glyph selection**: pixels on a strong gradient get
    a directional character (`| / - \`) instead of a flat density char,
    which visibly sharpens outlines.
  - `halfblock` — `▀` per cell with independent fg/bg colors (2 source
    pixel rows per cell). The best quality/performance/compatibility
    trade-off, and the default.
  - `quarterblock` — the 16 Unicode quadrant-block glyphs, 2x2 source
    pixels per cell, blended two-tone (fg = lit quadrants, bg = unlit).
  - `braille` — Unicode Braille cells, 2x4 source pixels per cell, with
    optional Bayer ordered dithering for smoother gradients.
- **Perceptually-accurate color**: TrueColor by default when supported;
  the 256/16-color fallback palettes are matched using a "redmean"
  perceptually-weighted distance (not plain Euclidean RGB), and an
  optional ordered dither (`--color-dither`) smooths out banding on
  reduced palettes.
- **True color pipeline**: auto-detects TrueColor / 256-color / 16-color
  / no-color terminals (`COLORTERM`, `TERM`), and every renderer adapts
  automatically. Colors are **run-length encoded per row** so flat
  regions cost one ANSI escape code no matter how wide they are — this
  is the main reason it stays fast.
- **GPU acceleration (CUDA)**: frame resizing, BGR→RGB color conversion,
  and Sobel edge detection all run on the GPU through OpenCV's CUDA
  module when available (`--info` shows what's detected), with an
  automatic, silent fallback to CPU everywhere — nothing to configure,
  and `--no-gpu` to force CPU-only if you want it.
- **Color themes**: grayscale, monochrome, sepia, retro green phosphor,
  amber CRT, matrix, plus Dracula/Nord/Gruvbox/Catppuccin/Tokyo Night
  duotones. Cycle live with `C`.
- **Correct aspect ratio**: compensates for terminal glyphs being taller
  than they are wide, so nothing looks squashed or stretched.
- **Audio playback**, synced to wall-clock/frame-timestamps (not just
  free-running), via `ffplay`. Pause/resume/seek/volume/mute all work.
- **Adaptive frame pacing**: if rendering falls behind, frames are
  dropped (not slowed down) to keep audio and video in sync.
- **YouTube / URL playback** via `yt-dlp`, streamed directly — no need
  to download the whole video first.
- **Interactive controls**, a **live status bar**, a **benchmark mode**
  (now GPU-aware), and a **TOML config file** for your preferred
  defaults.

## Install

```bash
pip install .                 # core: local video/GIF playback
pip install ".[youtube]"      # + YouTube/URL playback (yt-dlp)
pip install ".[all]"          # + CPU/mem stats
```

You also need **FFmpeg** on your `PATH` (used for audio playback and as
OpenCV's video backend):

```bash
brew install ffmpeg           # macOS
sudo apt install ffmpeg       # Ubuntu/Debian
# Windows: https://ffmpeg.org/download.html
```

GPU acceleration is automatic if your OpenCV build has CUDA support
(the standard `pip install opencv-python` wheels do not — you'd need a
CUDA-enabled build). Check what's detected with `termcinema --info`;
everything works fine on CPU either way.

## Usage

```bash
termcinema movie.mp4                        # asks settings, then plays
termcinema movie.mp4 -y                      # skip the wizard, play now
termcinema clip.gif --mode braille -y
termcinema video.mkv --color truecolor --theme green -y
termcinema "https://youtube.com/watch?v=dQw4w9WgXcQ"
termcinema movie.mp4 --benchmark --frames 300
termcinema --info          # show detected terminal + GPU capabilities
termcinema --list-modes    # show all rendering modes
```

### Controls

| Key         | Action                     |
|-------------|----------------------------|
| `Space`     | Pause / resume             |
| `← / →`     | Seek -5s / +5s              |
| `↑ / ↓`     | Volume up / down            |
| `M`         | Mute                        |
| `R`         | Cycle rendering mode        |
| `C`         | Cycle color theme           |
| `*` / `/`   | Speed up / slow down        |
| `0`         | Reset speed to 1.0x         |
| `S`         | Save a screenshot           |
| `Q`         | Quit                        |

### Config file

Save your preferred settings once:

```bash
termcinema movie.mp4 --mode halfblock --color truecolor --theme green --save-config
```

This writes `~/.config/termcinema/config.toml`, which is loaded
automatically on every future run (CLI flags always override it).

## Architecture

```
termcinema/
  terminal.py      Terminal capability detection (color depth, size, unicode)
  colorspace.py     Color themes + redmean-weighted 256/16-color ANSI quantization
  scaler.py         Aspect-ratio-correct resizing, GPU (CUDA) resize/color/sobel paths
  video.py          OpenCV/FFmpeg video decoding, GPU color conversion
  audio.py          ffplay-based audio playback, wall-clock sync
  youtube.py        yt-dlp URL resolution (true streaming, no pre-download)
  input_handler.py  Cross-platform non-blocking keyboard input
  wizard.py         Interactive pre-playback settings prompt
  player.py         Threaded decode/render/audio pipeline, adaptive pacing
  ui.py             Status bar rendering
  config.py         TOML config load/save
  benchmark.py       Throughput measurement (decode/render fps, GPU status)
  cli.py            Argument parsing / entry point
  renderer/
    base.py          Fast ANSI assembly: RLE + precomputed/memoized color codes
    ascii_mode.py     Luminance-ramp + edge-aware glyph selection (GPU Sobel)
    halfblock.py      ▀ half-block renderer (2 px/cell vertical)
    blocks.py          Quarter-block renderer (2x2 px/cell)
    braille.py          Braille dot-matrix renderer (2x4 px/cell)
```

### How rendering works

1. **Decode**: OpenCV pulls a frame (FFmpeg backend, hardware
   acceleration requested when available), then BGR→RGB conversion runs
   on the GPU when available.
2. **Theme**: an optional color transform (grayscale/sepia/etc) is
   applied — fully vectorized NumPy, no Python-level pixel loops.
3. **Scale**: the frame is resized to `(cols * cols_per_cell) x (rows *
   rows_per_cell)` pixels — the exact resolution the chosen renderer
   needs — with aspect ratio corrected for the terminal's glyph shape
   (`compute_grid_size` in `scaler.py`), not the source frame's raw
   pixel dimensions. Downscaling uses area-averaging interpolation
   (every source pixel contributes, not just the nearest one) for
   accurate color reproduction; resize runs on GPU when available.
4. **Render**: the mode-specific renderer picks a glyph per cell
   (luminance ramp, edge direction, dot pattern, or quadrant pattern)
   and computes its color(s), all vectorized with NumPy. Edge detection
   for the ASCII modes uses a GPU Sobel filter when available.
5. **Assemble**: `renderer/base.py` turns the glyph/color grid into an
   ANSI string. Consecutive same-colored cells within each row are
   merged into a single escape code via a vectorized `np.diff`-based
   run-length pass (no per-cell Python loop for the common case), and
   256-/16-color escape fragments are looked up from a precomputed
   table instead of being formatted fresh — only TrueColor's 16M
   possible values need any formatting at all, and those are cached
   after their first use per session.
6. **Sync & pace**: the player thread compares each frame's timestamp
   to wall-clock-since-start (scaled by playback speed) and sleeps or
   drops frames as needed, rather than assuming decode + render always
   takes exactly `1/fps` seconds.

### Color accuracy

- TrueColor mode carries the full 24-bit source color with no
  quantization at all — the most accurate any renderer can be.
- When TrueColor isn't available, the 256/16-color palette match uses
  a **redmean-weighted distance** (the same perceptual approximation
  ImageMagick uses for color reduction) instead of plain Euclidean RGB
  distance, which noticeably improves how close the reduced palette
  looks to the source, especially in skin tones and warm colors.
- `--color-dither` applies a 4x4 Bayer ordered dither before palette
  quantization, trading a little visible texture for much smoother
  gradients (skies, skin) instead of hard color bands.
- Downscaling always uses `INTER_AREA` interpolation, which averages
  every source pixel that lands in each output cell rather than
  sampling a single nearest pixel — this alone measurably improves
  color accuracy versus naive resizing, especially at high
  source-resolution-to-terminal-size ratios (e.g. 4K into 100 columns).

### Performance

On the built-in benchmark (`termcinema clip.mp4 --benchmark`), a
320x240 clip rendered into a large ~184x69 TrueColor terminal grid
measured 200+ fps combined decode+render for `halfblock` and 100+ fps
for the highest-resolution `braille` mode — comfortably above
real-time for any actual video frame rate, even without a GPU. The
techniques that make this possible, in order of impact:

1. Per-row ANSI run-length encoding with vectorized run-boundary
   detection (`renderer/base.py`) — the biggest win, since escape-code
   string formatting is far more expensive than the color math itself.
2. Precomputed 256-/16-color escape-code tables and a memoized
   TrueColor code cache — most modes spend zero time formatting color
   strings after the first few frames of a scene.
3. Vectorized color math — every color transform, quantization, and
   glyph-selection step operates on whole NumPy arrays, not per-pixel
   Python loops.
4. Precomputed color-quantization lookup tables for 256/16-color modes
   (`colorspace.py`) — nearest-palette-color search is done once at
   startup for all 32³ quantized RGB buckets, not per pixel per frame.
5. Optional GPU (CUDA) offload for resize, color conversion, and edge
   detection, auto-detected and used transparently when available.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Covers the renderer, color transforms/quantization, and aspect-ratio
scaling math — the parts that are pure functions and don't need a real
terminal or video file.

## Known limitations

- `ffplay`-based audio has no live in-place volume control; changing
  volume restarts audio at the current position (a few ms of glitch).
- Live audio pause/resume similarly restarts the ffplay process rather
  than truly pausing it in place.
- GPU acceleration requires an OpenCV build with CUDA support; the
  stock `opencv-python` wheels from PyPI do not include it. It's
  auto-detected (`termcinema --info` shows the result) and silently
  falls back to CPU if unavailable, for resize, color conversion, and
  edge detection alike — nothing to configure either way.
- The settings wizard only appears when stdin/stdout are an interactive
  terminal (not when piped, scripted, or run with `-y`/`--yes`), so
  automation always gets the fast, prompt-free path.
