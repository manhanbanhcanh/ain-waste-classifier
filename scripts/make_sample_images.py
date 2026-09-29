#!/usr/bin/env python3
"""Generate ``data/sample/`` — original synthetic offline smoke-test images.

These are **not** photographs of trash. Each image is a small, deterministic,
generated PNG (a colored rectangle with a simple per-class texture pattern)
encoded directly with the standard-library :mod:`zlib` module. No Pillow, no
network access, and no web-scraped images are involved.

Run with no arguments to (re)write the committed ``data/sample/`` tree:

    python scripts/make_sample_images.py

The output is deterministic for a fixed ``--seed`` (default ``42``): running
this script twice produces byte-identical PNGs, which is how the kit verifies
that the committed sample images actually match this generator.
"""
from __future__ import annotations

import argparse
import random
import struct
import zlib
from pathlib import Path

WIDTH = 32
HEIGHT = 32
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DIR = DATA_DIR / "sample"

# Base RGB color per class — chosen to be visually distinct, not realistic.
BASE_COLORS: dict[str, tuple[int, int, int]] = {
    "cardboard": (150, 111, 51),
    "glass": (120, 200, 190),
    "metal": (170, 170, 180),
    "paper": (235, 230, 210),
    "plastic": (70, 120, 220),
    "trash": (90, 80, 70),
}

CLASSES = tuple(BASE_COLORS.keys())


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _encode_png(pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """Encode an RGB pixel grid (rows of (r, g, b) tuples) as PNG bytes."""
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type 0 (None) for every scanline
        for r, g, b in row:
            raw.extend((r, g, b))
    compressed = zlib.compress(bytes(raw), level=9)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _make_pixels(
    color: tuple[int, int, int],
    rng: random.Random,
    *,
    texture: str = "grid",
    dark: bool = False,
    blur: bool = False,
) -> list[list[tuple[int, int, int]]]:
    """Build a small textured rectangle so classes are not just flat colors."""
    r0, g0, b0 = color
    pixels: list[list[tuple[int, int, int]]] = []
    for y in range(HEIGHT):
        row: list[tuple[int, int, int]] = []
        for x in range(WIDTH):
            # Deterministic per-pixel jitter (a stand-in for material texture).
            jitter = rng.randint(-12, 12)
            if texture == "grid" and (x % 6 == 0 or y % 6 == 0):
                jitter -= 20
            elif texture == "specks" and rng.random() < 0.08:
                jitter += 40
            elif texture == "stripes" and (x // 4) % 2 == 0:
                jitter -= 10
            r, g, b = r0 + jitter, g0 + jitter, b0 + jitter
            row.append((_clamp(r), _clamp(g), _clamp(b)))
        pixels.append(row)

    if blur:
        # Cheap box blur: average each pixel with its 4-neighborhood.
        blurred: list[list[tuple[int, int, int]]] = []
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                neighbors = [pixels[y][x]]
                if x > 0:
                    neighbors.append(pixels[y][x - 1])
                if x < WIDTH - 1:
                    neighbors.append(pixels[y][x + 1])
                if y > 0:
                    neighbors.append(pixels[y - 1][x])
                if y < HEIGHT - 1:
                    neighbors.append(pixels[y + 1][x])
                avg = tuple(_clamp(sum(c) / len(neighbors)) for c in zip(*neighbors))
                row.append(avg)
            blurred.append(row)
        pixels = blurred

    if dark:
        pixels = [
            [(_clamp(r * 0.35), _clamp(g * 0.35), _clamp(b * 0.35)) for r, g, b in row]
            for row in pixels
        ]

    return pixels


def build_sample_set(seed: int) -> dict[Path, bytes]:
    """Return {relative_path: png_bytes} for the whole data/sample/ tree."""
    textures = {
        "cardboard": "stripes",
        "glass": "grid",
        "metal": "specks",
        "paper": "grid",
        "plastic": "specks",
        "trash": "stripes",
    }
    files: dict[Path, bytes] = {}
    for class_index, cls in enumerate(CLASSES):
        # Use the class's fixed position instead of hash(cls): Python's
        # string hash is randomized per-process unless PYTHONHASHSEED is
        # pinned, which would make this generator non-deterministic.
        rng = random.Random(seed + class_index)
        color = BASE_COLORS[cls]
        normal = _make_pixels(color, rng, texture=textures[cls])
        files[Path(cls) / f"{cls}_01.png"] = _encode_png(normal)

    # A couple of deliberately hard/blurred/dark variants for the required
    # "hard image" and "blur/dark image" demos, without inflating the count
    # past the <=20 budget.
    rng = random.Random(seed + 1)
    cardboard_blur = _make_pixels(
        BASE_COLORS["cardboard"], rng, texture=textures["cardboard"], blur=True
    )
    files[Path("cardboard") / "cardboard_02_hard_blur.png"] = _encode_png(cardboard_blur)

    rng = random.Random(seed + 2)
    glass_dark = _make_pixels(
        BASE_COLORS["glass"], rng, texture=textures["glass"], dark=True
    )
    files[Path("glass") / "glass_02_hard_dark.png"] = _encode_png(glass_dark)

    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42).")
    parser.add_argument(
        "--out",
        type=Path,
        default=SAMPLE_DIR,
        help="Output directory for the sample tree (default: data/sample).",
    )
    args = parser.parse_args()

    files = build_sample_set(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for rel_path, png_bytes in files.items():
        full_path = args.out / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(png_bytes)
    total = len(files)
    print(f"Wrote {total} sample images (<=20 budget) under {args.out} (seed={args.seed}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
