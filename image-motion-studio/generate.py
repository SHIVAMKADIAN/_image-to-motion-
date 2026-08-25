#!/usr/bin/env python3
"""
Image Motion Studio — CLI
Usage:
    python generate.py <image_path> [output_dir]

Output:
    <name>_2.0s.mp4  — living cinematic clip with full optical & biological dynamics
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# Make sure backend modules are importable
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR / "backend"
for p in (str(CURRENT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.depth import estimate_depth
from backend.renderer import run_pipeline

# ─────────────────────────────────────────────
# DEFAULT SETTINGS (Living Cinematic Engine)
# ─────────────────────────────────────────────
DEFAULTS = dict(
    duration            = 2.0,
    fps                 = 30,
    resolution          = "1080p",
    aspect_ratio        = "original",
    edge_fill           = "inpaint",

    # Camera Motion
    push_in             = 0.0,
    h_drift             = 0.0,
    v_drift             = 0.0,
    handheld            = 6.5,
    zoom_out            = 5.0,

    # Parallax / Depth
    depth_strength      = 15.0,
    foreground_separation = 10.0,

    # Bio-Motion & Physiological Dynamics
    breathing           = 10.0,
    watcher_sway        = 10.0,
    blink               = False,
    micro_saccades      = 2.5,
    edge_flutter        = 1.0,
    heartbeat_pulse     = 2.5,

    # Atmosphere & Optical Physics
    dust_particles      = 1.0,
    light_shift         = 2.0,
    film_grain          = 3.0,
    rack_focus          = 2.0,
    specular_shimmer    = 2.0,
    motion_blur         = 1.0,
    camera_shake        = 2.0,
)


def find_ffmpeg():
    """Return path to ffmpeg binary."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    return ffmpeg


def progress_bar(label: str):
    def cb(stage: str, pct: int):
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  {label} [{bar}] {pct:3d}%  {stage:<30}", end="", flush=True)
        if pct >= 100:
            print()
    return cb


def main():
    parser = argparse.ArgumentParser(
        description="Image Motion Studio CLI — generates living cinematic clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py photo.jpg
  python generate.py photo.jpg ./my_outputs
        """,
    )
    parser.add_argument("image", help="Path to input image (JPG/PNG/WEBP)")
    parser.add_argument("output_dir", nargs="?", default="outputs",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    print(f"\n🎬 Image Motion Studio — Living Cinematic Engine")
    print(f"   Image     : {image_path.name}")
    print(f"   Output dir: {output_dir}")
    print()

    # ── Step 1: Depth Estimation ──────────────────────────────────────────────
    print("1/2  Estimating depth (Depth Anything V2 Small, MPS)...")
    depth_path = str(temp_dir / f"{stem}_depth.png")
    depth_map = estimate_depth(str(image_path), depth_path)
    print(f"     ✅ Depth map: {depth_map.shape[1]}x{depth_map.shape[0]}")

    # ── Step 2: Render clip ─────────────────────────────────────────
    print(f"\n2/2  Rendering {DEFAULTS['duration']} sec video clip...")
    cb = progress_bar("     Render")

    output_file = run_pipeline(
        image_path        = str(image_path),
        depth_map         = depth_map,
        output_dir        = str(output_dir),
        temp_dir          = str(temp_dir),
        **DEFAULTS,
        progress_callback = cb,
    )

    final_path = output_dir / f"{stem}_{DEFAULTS['duration']}s.mp4"
    if Path(output_file).exists():
        shutil.move(output_file, final_path)

    print(f"     ✅ {final_path.name}  ({final_path.stat().st_size // 1024} KB)")

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════╗
║  ✅  Done!                       ║
╠══════════════════════════════════╣
║  📹  {final_path.name}
║  🗂   {str(output_dir)}
╚══════════════════════════════════╝
""")

    # Clean up temp
    shutil.rmtree(str(temp_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
