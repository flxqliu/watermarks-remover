#!/usr/bin/env python3
"""Render the app icon into an .iconset directory. Stdlib only.

The icon is drawn once at 2x the largest size and box-downsampled to every
size macOS asks for; the downsample is what antialiases the edges, so there is
no separate supersampling pass. Kept in Python (rather than checked-in PNGs)
so the icon stays diffable and the repo stays free of binaries.

Usage:
    python3 make_icon.py --out build/AppIcon.iconset
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

# Palette. Indigo -> violet, the same family the README badges sit in.
TOP = (0x5B, 0x69, 0xF5)
BOTTOM = (0x7C, 0x3A, 0xED)
SHEET = (0xFF, 0xFF, 0xFF)
LINE = (0xC7, 0xD2, 0xFE)
MARK = (0xFB, 0xBF, 0x24)
SWEEP = (0xFF, 0xFF, 0xFF)

RENDER_SIZE = 2048
ICONSET_SIZES = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]


def rounded_rect(x, y, left, top, right, bottom, radius):
    """True when (x, y) is inside the rounded rectangle."""
    if x < left or x > right or y < top or y > bottom:
        return False
    cx = min(max(x, left + radius), right - radius)
    cy = min(max(y, top + radius), bottom - radius)
    dx = x - cx
    dy = y - cy
    return dx * dx + dy * dy <= radius * radius


def blend(base, color, alpha):
    return tuple(round(b + (c - b) * alpha) for b, c in zip(base, color, strict=True))


def render(size: int) -> bytearray:
    """Draw the icon at `size` px and return RGBA rows."""
    pixels = bytearray(size * size * 4)

    # Body: 0.04..0.96 leaves the margin macOS icons are drawn with.
    body = (0.055, 0.055, 0.945, 0.945)
    body_radius = 0.21
    sheet = (0.285, 0.235, 0.715, 0.775)
    # Diagonal sweep, expressed as distance to the line through two points.
    sweep_a = (0.14, 0.86)
    sweep_b = (0.90, 0.16)
    dx = sweep_b[0] - sweep_a[0]
    dy = sweep_b[1] - sweep_a[1]
    length = (dx * dx + dy * dy) ** 0.5
    nx, ny = -dy / length, dx / length

    lines = [
        (0.345, 0.345, 0.655, 0.373),
        (0.345, 0.415, 0.610, 0.443),
        (0.345, 0.485, 0.640, 0.513),
    ]
    marks = [0.345 + i * 0.052 for i in range(5)]

    for py in range(size):
        y = (py + 0.5) / size
        row = py * size * 4
        gradient = blend(TOP, BOTTOM, min(max((y - body[1]) / (body[3] - body[1]), 0.0), 1.0))
        for px in range(size):
            x = (px + 0.5) / size
            index = row + px * 4

            if not rounded_rect(x, y, *body, body_radius):
                continue  # stays transparent

            color = gradient

            # Subtle top-left sheen.
            sheen = max(0.0, 0.55 - (x + y))
            if sheen > 0:
                color = blend(color, (0xFF, 0xFF, 0xFF), sheen * 0.18)

            if rounded_rect(x, y, *sheet, 0.045):
                color = SHEET
                for lx0, ly0, lx1, ly1 in lines:
                    if rounded_rect(x, y, lx0, ly0, lx1, ly1, 0.014):
                        color = LINE
                if 0.555 <= y <= 0.585:
                    for mx in marks:
                        if rounded_rect(x, y, mx, 0.555, mx + 0.030, 0.585, 0.012):
                            color = MARK

            # The sweep crosses everything: it is the "removal" gesture.
            distance = abs((x - sweep_a[0]) * nx + (y - sweep_a[1]) * ny)
            if distance < 0.052:
                edge = 1.0 - (distance / 0.052) ** 2
                color = blend(color, SWEEP, 0.30 + 0.45 * edge)

            pixels[index] = color[0]
            pixels[index + 1] = color[1]
            pixels[index + 2] = color[2]
            pixels[index + 3] = 255

    return pixels


def downsample(pixels: bytearray, size: int, target: int) -> bytearray:
    """Box-filter `pixels` (size x size RGBA) down to target x target.

    Alpha-weighted, so the transparent margin outside the icon body does not
    bleed dark pixels into the rounded corners.
    """
    if target == size:
        return pixels
    if target > size:
        raise ValueError(f"cannot upscale {size} to {target}")
    out = bytearray(target * target * 4)
    for ty in range(target):
        y0 = ty * size // target
        y1 = max(y0 + 1, (ty + 1) * size // target)
        for tx in range(target):
            x0 = tx * size // target
            x1 = max(x0 + 1, (tx + 1) * size // target)
            r = g = b = a = 0
            for sy in range(y0, y1):
                base = (sy * size + x0) * 4
                for offset in range(0, (x1 - x0) * 4, 4):
                    index = base + offset
                    alpha = pixels[index + 3]
                    r += pixels[index] * alpha
                    g += pixels[index + 1] * alpha
                    b += pixels[index + 2] * alpha
                    a += alpha
            index = (ty * target + tx) * 4
            if a:
                out[index] = r // a
                out[index + 1] = g // a
                out[index + 2] = b // a
            out[index + 3] = a // ((y1 - y0) * (x1 - x0))
    return out


def write_png(path: Path, pixels: bytearray, size: int) -> None:
    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw.append(0)  # filter type 0
        raw.extend(pixels[y * stride : (y + 1) * stride])

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="iconset directory to write")
    parser.add_argument(
        "--render-size", type=int, default=RENDER_SIZE, help="internal render resolution"
    )
    args = parser.parse_args()

    largest = max(size for _, size in ICONSET_SIZES)
    if args.render_size < largest:
        parser.error(f"--render-size must be at least {largest}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    master = render(args.render_size)
    for name, size in ICONSET_SIZES:
        write_png(out / name, downsample(master, args.render_size, size), size)
    print(f"wrote {len(ICONSET_SIZES)} icons to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
