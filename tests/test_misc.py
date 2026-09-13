import numpy as np

from termcinema import filters
from termcinema.media import audio, subtitles
from termcinema.media.inputs import Input, summarize
from termcinema.term.caps import ColorDepth
from termcinema.term.keys import Key, Mouse, Parser, Reply
from termcinema.ui import layout
from termcinema.ui.canvas import Canvas, fmt_time, parse_time, text_width, truncate
from termcinema.render.encode import CellFrame


# -- input parsing -------------------------------------------------------------

def test_parser_keys_and_sequences():
    p = Parser()
    evs = p.feed("q \x1b[A\x1b[1;2C\x1b[5~\r\x7f")
    assert [e.name for e in evs] == ["q", "space", "up", "shift+right", "pgup", "enter", "backspace"]


def test_parser_mouse_and_replies():
    p = Parser()
    evs = p.feed("\x1b[<0;10;5M\x1b[<64;3;3M\x1b[?62;4;22c\x1b[6;20;10t\x1b_Gi=31;OK\x1b\\")
    assert evs[0] == Mouse(9, 4, 0, "press")
    assert evs[1].kind == "wheel_up"
    assert isinstance(evs[2], Reply) and evs[2].seq.endswith("c")
    assert isinstance(evs[3], Reply) and "6;20;10t" in evs[3].seq
    assert isinstance(evs[4], Reply) and "OK" in evs[4].seq


def test_parser_split_sequence_and_lone_escape():
    p = Parser()
    assert p.feed("\x1b[") == []
    assert p.feed("B") == [Key("down")]
    assert p.feed("\x1b") == []
    assert p.flush() == [Key("esc")]


# -- subtitles -----------------------------------------------------------------

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello <i>there</i>

2
00:00:04,000 --> 00:00:06,000
Second &amp; line
"""

VTT = """WEBVTT

00:01.000 --> 00:02.000 align:start
one

00:02.000 --> 00:03.000
one
two
"""


def test_srt_parse_and_lookup():
    cues = subtitles.parse(SRT)
    assert [c.text for c in cues] == ["Hello there", "Second & line"]
    tr = subtitles.Track(cues)
    assert tr.at(2.0) == "Hello there"
    assert tr.at(3.7) == ""
    assert tr.at(5.0) == "Second & line"


def test_vtt_rolling_captions_are_collapsed():
    cues = subtitles.parse(VTT)
    assert [c.text for c in cues] == ["one", "two"]


# -- layout ----------------------------------------------------------------------

def test_fit_preserves_aspect():
    L = layout.compute(200, 50, 16 / 9, 2.0, "fit")
    assert L.h == 50 and abs(L.w / (L.h * 2.0) - 16 / 9) < 0.05
    assert L.x0 == (200 - L.w) // 2


def test_fill_crops_source():
    L = layout.compute(100, 50, 16 / 9, 2.0, "fill")
    assert (L.w, L.h) == (100, 50)
    cx, cy, cw, ch = L.crop
    assert cw < 1.0 and ch == 1.0


def test_zoom_crops_center():
    L = layout.compute(100, 40, 16 / 9, 2.0, "fit", zoom=2.0)
    cx, cy, cw, ch = L.crop
    assert abs(cw - 0.5) < 1e-6 and abs(cx - 0.25) < 1e-6


# -- canvas & helpers ------------------------------------------------------------

def test_text_width_and_truncate():
    assert text_width("abc") == 3
    assert text_width("中文") == 4
    assert truncate("hello world", 6) == "hello…"


def test_time_helpers():
    assert fmt_time(3725) == "1:02:05"
    assert fmt_time(65) == "01:05"
    assert parse_time("1:30") == 90
    assert parse_time("1:02:03") == 3723


def test_canvas_composes_translucent_panel():
    frame = CellFrame(np.full((3, 5), 0x2580, np.int32), np.full((3, 5, 3), 200, np.uint8),
                      np.full((3, 5, 3), 100, np.uint8))
    cv = Canvas(3, 5)
    cv.shade(1, 0, 1, 5, (0, 0, 0), 0.5)
    cv.text(1, 1, "hi")
    cv.compose(frame)
    assert frame.cps[0, 0] == 0x2580 and frame.cps[1, 1] == ord("h")
    assert frame.bg[1, 0, 0] == 50 and frame.fg[1, 0, 0] == 100    # glyph kept, dimmed
    assert frame.bg[0, 0, 0] == 100


# -- media plumbing --------------------------------------------------------------

def test_input_args_include_headers_and_seek():
    inp = Input("https://example.com/v.mp4", headers={"User-Agent": "UA", "Referer": "https://x"})
    args = inp.ffmpeg_args(12.5)
    assert args[args.index("-user_agent") + 1] == "UA"
    assert "Referer: https://x" in args[args.index("-headers") + 1]
    assert args[args.index("-ss") + 1] == "12.500"
    assert args[-2:] == ["-i", "https://example.com/v.mp4"]


def test_probe_summary_handles_rotation_and_sar():
    data = {"format": {"duration": "10.0", "format_name": "mov,mp4"},
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001",
                         "side_data_list": [{"rotation": -90}]},
                        {"codec_type": "audio", "codec_name": "aac"}]}
    s = summarize(data)
    assert (s["width"], s["height"]) == (1080, 1920)
    assert abs(s["fps"] - 29.97) < 0.01 and s["has_audio"]


def test_atempo_chain_covers_extremes():
    assert audio.atempo_chain(1.0) == ""
    assert audio.atempo_chain(4.0) == "atempo=2.0,atempo=2.0000" or audio.atempo_chain(4.0).count("atempo") == 2
    assert audio.atempo_chain(0.25).count("atempo=0.5") == 2


def test_wall_clock_pause_and_speed():
    c = audio.WallClock(10.0, 2.0)
    c.set_paused(True)
    t = c.time()
    assert c.time() == t
    c.reset(5.0)
    assert abs(c.time() - 5.0) < 1e-6


# -- filters ---------------------------------------------------------------------

def test_every_effect_runs():
    img = np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    for name in filters.EFFECTS:
        out = filters.apply_effect(img, name, 1.0)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_adjust_identity_is_noop():
    img = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert filters.apply_adjust(img, filters.Adjust()) is img
    brighter = filters.apply_adjust(img, filters.Adjust(brightness=0.2))
    assert brighter.astype(int).sum() > img.astype(int).sum()


def test_color_depth_parse():
    assert ColorDepth.parse("256") == ColorDepth.ANSI256
    assert ColorDepth.parse("truecolor").label == "truecolor"
