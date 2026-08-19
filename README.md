# EXR to PNG Converter

A Python script for batch-converting EXR images to PNG, with optional frame-range filtering, grain, multiprocessing, and 8-bit or 16-bit PNG output.

## Requirements

- Python 3
- Dependencies listed in `requirements.txt`

## Installation

Open PowerShell in the folder containing the script and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Optionally, create and activate a virtual environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Basic usage:

```powershell
python exr_to_png_optimized.py --input "C:\path\to\exr" --output "C:\path\to\png"
```

Example using 16-bit output, 4 workers, and only frames 50 through 120:

```powershell
python exr_to_png_optimized.py `
    --input "C:\path\to\exr" `
    --output "C:\path\to\png" `
    --bit-depth 16 `
    --workers 4 `
    --frame-start 50 `
    --frame-end 120
```

Example with grain settings:

```powershell
python exr_to_png_optimized.py `
    --input "C:\path\to\exr" `
    --output "C:\path\to\png" `
    --grain-size 1.0 `
    --grain-scale 2.0
```

## Command-line options

- `--input PATH` — input folder containing EXR files
- `--output PATH` — output folder for PNG files
- `--frame-start N` — first frame to process, inclusive
- `--frame-end N` — last frame to process, inclusive
- `--grain-size VALUE` — grain size
- `--grain-scale VALUE` — grain strength/scale
- `--workers N` — number of images processed simultaneously
- `--bit-depth {8,16}` — PNG output bit depth

The script searches subdirectories recursively by default and preserves the input directory structure in the output folder.

Frame filtering uses the last numeric group in each filename. For example, `shot_fisheye_00450.exr` is treated as frame `450`.

## Notes

The script applies a simple linear-to-sRGB transfer by default before writing the PNG. This is not a full ACES/ACEScg color-space conversion.

PNG compression, recursive scanning, sRGB conversion, monochrome grain, and default grain settings are currently configured as constants near the top of the script.
