# oximedia-cli

Command-line interface for the OxiMedia multimedia framework.

## Overview

`oximedia-cli` provides a command-line tool for working with media files using only royalty-free codecs.

## Installation

```bash
cargo install --path oximedia-cli
```

Or build from source:

```bash
cargo build --release -p oximedia-cli
```

## Commands

### Probe

Analyze media files and display format information:

```bash
# Basic probe
oximedia probe -i video.mkv

# Detailed output / machine-readable JSON
oximedia probe -i video.mkv --detail
oximedia probe -i video.mkv --format json

# Show stream information, content hash, quality snapshot
oximedia probe -i video.mkv --streams
oximedia probe -i video.mkv --hash
oximedia probe -i video.mkv --quality-snapshot
```

### Info

Display supported formats and codecs:

```bash
oximedia info
```

Output:
```
Supported Containers:
  ✓ Matroska (.mkv)
  ✓ WebM (.webm)
  ✓ Ogg (.ogg, .opus, .oga)
  ✓ FLAC (.flac)
  ✓ WAV (.wav)

Supported Video Codecs (Green List):
  ✓ AV1 (Primary codec, best compression)
  ✓ VP9 (Excellent quality/size ratio)
  ✓ VP8 (Legacy support)
  ✓ Theora (Legacy support)

Supported Audio Codecs (Green List):
  ✓ Opus (Primary codec, versatile)
  ✓ Vorbis (High quality)
  ✓ FLAC (Lossless)
  ✓ PCM (Uncompressed)

Rejected Codecs (Patent-Encumbered):
  ✗ H.264/AVC
  ✗ H.265/HEVC
  ✗ AAC
  ✗ AC-3/E-AC-3
  ✗ DTS
```

### Transcode

Remux, trim, select streams, and re-encode within the supported codec
matrix. Real re-encode targets today: **video** mjpeg, apv, mpeg2, ffv1,
prores, rawvideo (from Y4M/raw-decodable sources); **audio** flac, pcm,
alac, opus (from WAV/FLAC-decodable sources). AV1/VP9/VP8 and
vorbis/aac/mp3 parse but fail with a precise unsupported-encode error —
full patent-free video re-encode is a 0.2.x roadmap item. Omitting
`--codec`/`--audio-codec` performs a stream copy.

```bash
# Stream-copy remux (no re-encode)
oximedia transcode -i input.mkv -o output.mkv

# Audio re-encode: WAV -> FLAC (sample-exact, verified in tests)
oximedia transcode -i input.wav -o output.flac

# Video re-encode: Y4M -> MPEG-2 elementary stream
oximedia transcode -i input.y4m -o output.m2v --codec mpeg2

# Seek and duration trim (works on the stream-copy path too)
oximedia transcode -i input.mkv -o output.mkv --ss 00:01:00 -t 30

# Stream selection (FFmpeg-style -map selectors)
oximedia transcode -i input.mkv -o output.mkv --map 0:a

# Frame-rate conversion on the frame-level path
oximedia transcode -i input.y4m -o output.m2v --codec mpeg2 -r 25

# EBU R128 loudness normalization during transcode
oximedia transcode -i input.wav -o output.flac --normalize-audio
```

### Extract

Extract frames from video:

```bash
# Extract all frames as PNG
oximedia extract video.mkv frames_%04d.png

# Extract first 100 frames
oximedia extract video.mkv frames_%04d.png -n 100

# Extract every 30th frame (1 fps from 30fps video)
oximedia extract video.mkv frames_%04d.png --every 30

# Extract as JPEG with quality
oximedia extract video.mkv frames_%04d.jpg --format jpg --quality 85

# Start from specific time
oximedia extract video.mkv frames_%04d.png --ss 00:05:00
```

### Batch

Process multiple files:

```bash
# Batch transcode with config file
oximedia batch input_dir/ output_dir/ config.toml

# Parallel processing with 4 jobs
oximedia batch input_dir/ output_dir/ config.toml -j 4

# Dry run (show what would be done)
oximedia batch input_dir/ output_dir/ config.toml --dry-run

# Continue on errors
oximedia batch input_dir/ output_dir/ config.toml --continue-on-error
```

## FFmpeg-Compatible Options

The CLI supports FFmpeg-style options for familiarity:

| OxiMedia | FFmpeg | Description |
|----------|--------|-------------|
| `-i` | `-i` | Input file |
| `-o` | (positional) | Output file |
| `--codec` | `-c:v` | Video codec |
| `--audio-codec` | `-c:a` | Audio codec |
| `--bitrate` | `-b:v` | Video bitrate |
| `--audio-bitrate` | `-b:a` | Audio bitrate (not implemented yet: warns and proceeds) |
| `--video-filter` | `-vf` | Video filter (`scale=W:H` supported; others error clearly) |
| `--audio-filter` | `-af` | Audio filter (`volume=N`/`volume=NdB` supported; others error clearly) |
| `--map` | `-map` | Stream selection (`0:v`, `0:a:1`, `-0:s`, ...; repeatable) |
| `--ss` | `-ss` | Start time (seek) |
| `-t` | `-t` | Duration |
| `-r` | `-r` | Output frame rate (frame-level encode path) |
| `-y` | `-y` | Overwrite output |

## Companion Binaries

In addition to the primary `oximedia` multitool, the crate ships two
companion binaries:

- `oximedia-ff` — drop-in FFmpeg replacement that translates argv through
  the `oximedia-compat-ffmpeg` layer and executes the resulting jobs.
  Supports `--dry-run`, `--plan`, `--explain`, and `--json` (emits the
  structured `TranslateResult` to stdout for tooling/CI consumption).
- `oximedia-cv2` — opt-in OpenCV-style helper layered on
  `oximedia-compat-cv2`. Use `oximedia-cv2 --list-functions` /
  `--list-constants` to enumerate the supported surface; the constants
  list is reflected at build time from the underlying
  `oximedia-compat-cv2` crate so it never drifts.

## Global Options

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Increase log verbosity (can stack: -v, -vv, -vvv) |
| `-q`, `--quiet` | Suppress logs and status/banner stdout (transcode/extract/image paths); command results, `--json` output, and errors still print |
| `--no-color` | Disable colored output (also honours `NO_COLOR`, `CLICOLOR=0`, `TERM=dumb`) |
| `--json` | Structured JSON output on stdout |
| `--ndjson` | One NDJSON record per item (conflicts with `--json`) |
| `--log-format json` | Structured JSON log output |
| `--progress <plain\|json>` | Progress reporting format |

## Examples

```bash
# Probe a file
oximedia probe -i video.mkv

# Remux Matroska -> Matroska with only the audio streams kept
oximedia transcode -i video.mkv -o audio_only.mkv --map 0:a

# Encode WAV to lossless FLAC with EBU R128 normalization
oximedia transcode -i mix.wav -o mix.flac --normalize-audio

# Extract every 30th frame as JPEG
oximedia extract video.mkv thumb_%03d.jpg --every 30 --format jpg

# Batch process all files matched by config patterns, 8 jobs in parallel
oximedia batch videos/ output/ convert.toml -j 8
```

## Config File Format (TOML)

For batch processing. The schema is flat (no sections); `patterns` is
required, everything else optional:

```toml
# Which files in the input directory to process (required)
patterns = ["*.mkv", "*.webm"]
# Optional excludes
exclude = ["*.tmp.mkv"]

# Codec/quality settings (all optional; omit codecs for stream copy)
video_codec = "mpeg2"
audio_codec = "flac"
video_bitrate = "2M"
scale = "1920:-1"
preset = "medium"
two_pass = false
crf = 6
threads = 0

# Output naming and traversal
output_extension = "mkv"
overwrite = false
recursive = true
```

## Exit Codes

Derived from `src/exit_codes.rs` (`OxiExitCode`); these are the only codes
the binary returns:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Unclassified runtime error |
| 2 | Command-line usage error (clap) |
| 3 | IO / file-system error (e.g. missing input file) |
| 4 | Input validation error |

## Policy

- Only supports patent-free codecs (Green List)
- Rejects patent-encumbered codecs with clear error messages
- Apache 2.0 license

## License

Apache-2.0

Version: 0.2.0 — 2026-07-15
