#!/usr/bin/env python3
"""
Darken keypoints in draw.io SVGs for print visibility.

- Extracts embedded base64 PNGs from the SVG content attribute
- Exactly matches keypoint colors and maps them to darker equivalents
- Only keypoint color matching — no gray pixel pass
- Re-encodes PNGs and rebuilds the SVGs
- Writes output to viz/kps/

Usage:
    python scripts/viz/darken_for_print.py
"""

import os
import re
import sys
import io
import base64
import numpy as np
from PIL import Image
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

SRC_DIR = _PROJECT_ROOT / "viz"
DST_DIR = SRC_DIR / "kps"
os.makedirs(DST_DIR, exist_ok=True)

SVG_FILES = ["iSLiM-Ours.drawio.svg", "iSLiM-PS-KPS.drawio.svg"]


def darken_pixels(img: Image.Image) -> Image.Image:
    rgba = np.array(img)
    if rgba.shape[-1] == 4:
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        valid = alpha > 0
    else:
        rgb = rgba
        valid = np.ones(rgb.shape[:2], dtype=bool)

    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    changed = np.zeros(rgb.shape[:2], dtype=bool)

    TARGET_COLORS = [
        ((33, 150, 243), (5, 40, 120)),
        ((76, 175, 80), (15, 60, 20)),
        ((244, 67, 54), (140, 10, 10)),
        ((255, 152, 0), (140, 30, 0)),
        ((255, 82, 82), (140, 10, 10)),
        ((79, 195, 247), (0, 60, 120)),
        ((206, 147, 216), (50, 5, 100)),
    ]
    TOLERANCE = 45

    tR_map = np.zeros_like(R)
    tG_map = np.zeros_like(G)
    tB_map = np.zeros_like(B)
    origR_map = np.zeros_like(R)
    origG_map = np.zeros_like(G)
    origB_map = np.zeros_like(B)
    kp_mask = np.zeros(rgb.shape[:2], dtype=bool)

    for (sr, sg, sb), (tr, tg, tb) in TARGET_COLORS:
        dist_sq = (R - sr) ** 2 + (G - sg) ** 2 + (B - sb) ** 2
        mask = (dist_sq < TOLERANCE ** 2) & valid
        kp_mask = kp_mask | mask
        tR_map[mask], tG_map[mask], tB_map[mask] = tr, tg, tb
        origR_map[mask], origG_map[mask], origB_map[mask] = sr, sg, sb

    if kp_mask.any():
        d_sq = (R - origR_map) ** 2 + (G - origG_map) ** 2 + (B - origB_map) ** 2
        ramp = np.clip(1.0 - np.sqrt(np.maximum(d_sq, 0)) / TOLERANCE, 0.0, 1.0)
        rr = ramp[kp_mask]
        R[kp_mask] = R[kp_mask] * (1 - rr) + tR_map[kp_mask] * rr
        G[kp_mask] = G[kp_mask] * (1 - rr) + tG_map[kp_mask] * rr
        B[kp_mask] = B[kp_mask] * (1 - rr) + tB_map[kp_mask] * rr
        changed = changed | kp_mask

    if not changed.any():
        return img

    rgb_out = rgb.copy()
    rgb_out[changed, 0] = np.clip(R[changed], 0, 255).astype(np.uint8)
    rgb_out[changed, 1] = np.clip(G[changed], 0, 255).astype(np.uint8)
    rgb_out[changed, 2] = np.clip(B[changed], 0, 255).astype(np.uint8)

    if rgba.shape[-1] == 4:
        rgba_out = np.dstack([rgb_out, alpha])
        return Image.fromarray(rgba_out, "RGBA")
    return Image.fromarray(rgb_out, "RGB")


def process_svg(input_path: str, output_path: str):
    print(f"\n{'=' * 60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'=' * 60}")

    with open(input_path, "r") as f:
        content = f.read()

    pattern = r'image=data:image/png,([a-zA-Z0-9+/=]+)'
    matches = list(re.finditer(pattern, content))

    print(f"  Found {len(matches)} embedded PNGs")

    seen = {}
    replacements = {}
    for m in matches:
        b64 = m.group(1)
        if b64 not in seen:
            seen[b64] = 1
        else:
            seen[b64] += 1

    print(f"  Unique PNGs: {len(seen)}")
    for b64, count in seen.items():
        print(f"    {len(b64)} chars (appears {count}x)")

    for idx, (b64, count) in enumerate(seen.items()):
        print(f"\n  [Unique {idx + 1}/{len(seen)}] Decoding...")
        try:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            print(f"    Size: {img.size}  Mode: {img.mode}")
        except Exception as e:
            print(f"    ERROR: {e}")
            replacements[b64] = b64
            continue

        print(f"    Darkening pixels...")
        img_dark = darken_pixels(img)

        buf = io.BytesIO()
        img_dark.save(buf, format="PNG", optimize=True)
        new_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        replacements[b64] = new_b64
        print(f"    Done (size: {len(b64)} -> {len(new_b64)})")

    print(f"\n  Applying replacements to SVG...")
    for old_b64, new_b64 in replacements.items():
        content = content.replace(old_b64, new_b64)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("Darkening keypoints for print visibility")
    print("=" * 60)

    for svg_file in SVG_FILES:
        input_path = os.path.join(SRC_DIR, svg_file)
        if not os.path.exists(input_path):
            print(f"  SKIP: {input_path} not found")
            continue
        output_path = os.path.join(DST_DIR, svg_file)
        process_svg(input_path, output_path)

    print(f"\n{'=' * 60}")
    print(f"Done. Output in {DST_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
