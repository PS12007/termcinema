"""Minimal sixel decoder used by the tests to verify the encoder round-trips."""
import re

import numpy as np


def decode(data: bytes):
    s = data.decode("ascii", "replace")
    start = s.index("q") + 1
    body = s[start:s.index("\x1b\\")]
    m = re.match(r'"1;1;(\d+);(\d+)', body)
    w, h = int(m.group(1)), int(m.group(2))
    body = body[m.end():]
    idx = np.full((h, w), -1, np.int32)
    x = y = 0
    color = 0
    i = 0
    while i < len(body):
        c = body[i]
        if c == "#":
            m = re.match(r"#(\d+)(;2;\d+;\d+;\d+)?", body[i:])
            if not m.group(2):
                color = int(m.group(1))
            i += m.end()
            continue
        if c == "$":
            x = 0
        elif c == "-":
            x = 0
            y += 6
        elif c == "!":
            m = re.match(r"!(\d+)(.)", body[i:])
            n, ch = int(m.group(1)), m.group(2)
            v = ord(ch) - 63
            for _ in range(n):
                for r in range(6):
                    if v & (1 << r) and y + r < h:
                        idx[y + r, x] = color
                x += 1
            i += m.end()
            continue
        elif "?" <= c <= "~":
            v = ord(c) - 63
            for r in range(6):
                if v & (1 << r) and y + r < h:
                    idx[y + r, x] = color
            x += 1
        i += 1
    return idx
