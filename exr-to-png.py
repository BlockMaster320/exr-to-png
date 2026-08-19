from pathlib import Path
import time
import re
import argparse
import numpy as np
import OpenImageIO as oiio
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# USER SETTINGS
# =========================================================

INPUT_FOLDER = r"Z:\zakazky\2026_03_16_COSMO_Planetarium\07_EXPORT\test\exr-to-png\exr"
OUTPUT_FOLDER = r"Z:\zakazky\2026_03_16_COSMO_Planetarium\07_EXPORT\test\exr-to-png\png"

RECURSIVE = True   # True = search subfolders too
APPLY_SRGB_TRANSFER = True
# Set to True if your EXR files are linear and you want normal display-ready PNGs.
# Set to False if your EXRs are already display-referred.
# IMPORTANT: this is a simple linear->sRGB transfer, not a full ACEScg transform.

# -------- Default frame range settings --------
# Inclusive range
DEFAULT_FRAME_START = None
DEFAULT_FRAME_END = None

# -------- Default grain settings --------
# Approximation of AE Add Grain look:
DEFAULT_AE_GRAIN_SIZE = 1.0
DEFAULT_AE_GRAIN_SCALE = 2.0
MONO_GRAIN = True         # True = monochrome/luminance grain
SEED = None               # None = random every file; set an int for repeatability

# Strength mapping:
# AE scale 0.4 is approximated here as 0.004 amplitude.

# -------- PNG settings --------
# PNG does NOT support LZW. PNG uses DEFLATE/zlib compression.
PNG_COMPRESSION_LEVEL = 0   # 0 fastest, 9 smallest file
PNG_FILTER = 8              # 8 = PNG_FILTER_NONE (usually fastest)
DEFAULT_PNG_BIT_DEPTH = 16  # 8 or 16

# -------- Multi-threading settings --------
MAX_WORKERS = 2

# =========================================================
# MULTI-THREADING
# =========================================================

def process_file_worker(args):
    (
        i,
        exr_path,
        input_root,
        output_root,
        grain_size,
        grain_scale,
        bit_depth,
        seed
    ) = args

    start = time.perf_counter()

    out_path = process_one_file(
        exr_path=exr_path,
        input_root=input_root,
        output_root=output_root,
        grain_size=grain_size,
        grain_scale=grain_scale,
        bit_depth=bit_depth,
        seed=seed
    )

    elapsed = time.perf_counter() - start

    return i, exr_path, out_path, elapsed

# =========================================================
# HELPERS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EXR files to PNG with optional frame filtering and grain settings."
    )

    parser.add_argument(
        "--frame-start",
        type=int,
        default=DEFAULT_FRAME_START,
        help="Inclusive start frame number to process (default: no lower limit)."
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=DEFAULT_FRAME_END,
        help="Inclusive end frame number to process (default: no upper limit)."
    )
    parser.add_argument(
        "--grain-size",
        type=float,
        default=DEFAULT_AE_GRAIN_SIZE,
        help=f"Grain size parameter (default: {DEFAULT_AE_GRAIN_SIZE})."
    )
    parser.add_argument(
        "--grain-scale",
        type=float,
        default=DEFAULT_AE_GRAIN_SCALE,
        help=f"Grain scale/strength parameter (default: {DEFAULT_AE_GRAIN_SCALE})."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of images to process simultaneously (default: 2)."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(INPUT_FOLDER),
        help=f'Input EXR folder (default: "{INPUT_FOLDER}").'
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_FOLDER),
        help=f'Output PNG folder (default: "{OUTPUT_FOLDER}").'
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=(8, 16),
        default=DEFAULT_PNG_BIT_DEPTH,
        help=f"PNG bit depth: 8 or 16 (default: {DEFAULT_PNG_BIT_DEPTH})."
    )

    return parser.parse_args()

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055
    )

def resize_float_channel(channel: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(
        channel,
        (out_w, out_h),
        interpolation=cv2.INTER_LINEAR
    )

def build_grain(height: int, width: int,
                grain_size: float,
                grain_strength: float,
                mono: bool = True,
                seed=None) -> np.ndarray:

    grain_size = max(0.01, min(1.0, grain_size))

    small_h = max(16, int(height * grain_size))
    small_w = max(16, int(width * grain_size))

    rng = np.random.default_rng(seed)
    channels = 1 if mono else 3

    # Fast path: at grain_size == 1.0 the noise is already full resolution.
    # Avoid allocating a second 8K grain image and resizing 8192x8192 -> 8192x8192.
    if small_h == height and small_w == width:
        up = rng.standard_normal(
            (height, width, channels),
            dtype=np.float32
        )
    else:
        small = rng.standard_normal(
            (small_h, small_w, channels),
            dtype=np.float32
        )

        up = np.empty((height, width, channels), dtype=np.float32)

        for c in range(channels):
            up[:, :, c] = resize_float_channel(
                small[:, :, c], width, height
            )

    std = float(up.std())
    if std > 1e-8:
        up /= std

    up *= grain_strength

    return up

def apply_grain(rgb: np.ndarray, grain: np.ndarray) -> None:
    luma = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    )

    np.clip(luma, 0.0, 1.0, out=luma)

    # Turn luma itself into the weight to avoid another H×W allocation.
    luma *= -0.65
    luma += 1.0

    rgb += grain * luma[:, :, None]

def extract_frame_number(path: Path):
    """
    Extract the last numeric group from the filename stem.

    Examples:
        render_00125.exr -> 125
        shot_20_fisheye_00450.exr -> 450
    """
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])

def list_exr_files(folder: Path, recursive: bool, frame_start=None, frame_end=None):
    if recursive:
        files = list(folder.rglob("*.exr")) + list(folder.rglob("*.EXR"))
    else:
        files = list(folder.glob("*.exr")) + list(folder.glob("*.EXR"))

    # remove duplicates if any
    files = sorted(set(files))

    if frame_start is None and frame_end is None:
        return files

    filtered = []
    for path in files:
        frame = extract_frame_number(path)

        if frame is None:
            print(f"Skipping file without frame number: {path.name}")
            continue

        if frame_start is not None and frame < frame_start:
            continue

        if frame_end is not None and frame > frame_end:
            continue

        filtered.append(path)

    return filtered

def make_output_path(exr_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = exr_path.relative_to(input_root).with_suffix(".png")
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def process_one_file(
    exr_path: Path,
    input_root: Path,
    output_root: Path,
    grain_size: float,
    grain_scale: float,
    bit_depth: int,
    seed=None
) -> Path:
    out_path = make_output_path(exr_path, input_root, output_root)

    # Read EXR
    buf = oiio.ImageBuf(str(exr_path))
    pixels = buf.get_pixels(oiio.FLOAT)

    if pixels is None or pixels.size == 0:
        raise RuntimeError(f"Could not read pixels from: {exr_path}")

    height, width, channels = pixels.shape

    if channels < 3:
        raise RuntimeError(f"Image has fewer than 3 channels: {exr_path}")

    # Use RGB, preserve alpha if present
    rgb = pixels[:, :, :3].astype(np.float32, copy=True)
    alpha = pixels[:, :, 3:4].astype(np.float32, copy=True) if channels >= 4 else None

    # Clean up weird values
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert to display-referred if requested
    if APPLY_SRGB_TRANSFER:
        rgb = linear_to_srgb(rgb)

    grain_strength = 0.01 * grain_scale

    # Grain
    grain = build_grain(
        height=height,
        width=width,
        grain_size=grain_size,
        grain_strength=grain_strength,
        mono=MONO_GRAIN,
        seed=seed
    )

    apply_grain(rgb, grain)

    # Clamp for PNG
    np.clip(rgb, 0.0, 1.0, out=rgb)

    if alpha is not None:
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=0.0)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        out_float = np.concatenate([rgb, alpha], axis=2)
    else:
        out_float = rgb

    # Quantize to requested PNG bit depth.
    if bit_depth == 8:
        out_pixels = np.rint(out_float * 255.0).astype(np.uint8)
    else:
        out_pixels = np.rint(out_float * 65535.0).astype(np.uint16)

    # Make output ImageBuf from numpy array
    out_buf = oiio.ImageBuf(out_pixels)

    # PNG write settings
    spec = out_buf.specmod()
    spec["png:compressionLevel"] = int(PNG_COMPRESSION_LEVEL)
    spec["png:filter"] = int(PNG_FILTER)
    spec["oiio:ColorSpace"] = "sRGB" if APPLY_SRGB_TRANSFER else "Linear"

    ok = out_buf.write(str(out_path))
    if not ok:
        raise RuntimeError(f"Write failed for {out_path}: {out_buf.geterror()}")

    return out_path

def main():
    args = parse_args()

    input_root = args.input
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    if args.workers < 1:
        print("Error: --workers must be at least 1")
        return

    if args.frame_start is not None and args.frame_end is not None:
        if args.frame_start > args.frame_end:
            print("Error: --frame-start cannot be greater than --frame-end")
            return

    if not input_root.exists():
        print(f"Input folder does not exist: {input_root}")
        return

    exr_files = list_exr_files(
        input_root,
        RECURSIVE,
        frame_start=args.frame_start,
        frame_end=args.frame_end
    )

    if not exr_files:
        print("No EXR files found.")
        return

    print(f"Found {len(exr_files)} EXR file(s).")
    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"Frame start : {args.frame_start}")
    print(f"Frame end   : {args.frame_end}")
    print(f"Grain size  : {args.grain_size}")
    print(f"Grain scale : {args.grain_scale}")
    print(f"Bit depth   : {args.bit_depth}")
    print(f"Workers     : {args.workers}")
    print()

    times = []
    total_start = time.perf_counter()

    # Execute workers
    jobs = []
    for i, exr_path in enumerate(exr_files, start=1):
        file_seed = None if SEED is None else (SEED + i)

        jobs.append(
            (
                i,
                exr_path,
                input_root,
                output_root,
                args.grain_size,
                args.grain_scale,
                args.bit_depth,
                file_seed
            )
        )

    times = []
    total_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_file_worker, job)
            for job in jobs
        ]

        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                i, exr_path, out_path, elapsed = future.result()

                times.append(elapsed)

                avg = sum(times) / len(times)

                # With 2 workers this is only an approximate ETA.
                remaining_images = len(exr_files) - completed
                remaining = avg * remaining_images / args.workers

                print(
                    f"[{completed}/{len(exr_files)}] "
                    f"frame/job {i}: "
                    f"{exr_path.name} -> {out_path.name} | "
                    f"{elapsed:.2f}s | "
                    f"avg {avg:.2f}s/img | "
                    f"ETA {remaining/60:.1f} min"
                )

            except Exception as e:
                print(f"[{completed}/{len(exr_files)}] ERROR")
                print(f"    {e}")

    total_elapsed = time.perf_counter() - total_start

    print("\nDone.")
    print(f"Processed: {len(times)} / {len(exr_files)}")

    if times:
        print(f"Average processing time: {sum(times)/len(times):.2f}s/image")
        print(f"Fastest image          : {min(times):.2f}s")
        print(f"Slowest image          : {max(times):.2f}s")

    print(f"Wall-clock time        : {total_elapsed/60:.2f} min")
    if times:
        print(f"Effective time/frame   : {total_elapsed/len(times):.2f}s")
        print(f"Throughput             : {len(times)/total_elapsed:.3f} frames/s")

if __name__ == "__main__":
    main()
