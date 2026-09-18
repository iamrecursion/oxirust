#!/usr/bin/env bash
# cv2-pipeline.sh - Pure-Rust OpenCV cv2 image processing chain.
#
# Demonstrates the `oximedia-cv2` binary (a drop-in for the OpenCV cv2 Python
# API). Subcommands take POSITIONAL <input> <output> arguments and follow
# OpenCV naming.
#
# Pipeline:
#   1. imread        -> normalise the source into a Mat, write back as PNG.
#   2. cvt-color     -> BGR -> grayscale (channel order matches OpenCV).
#   3. gaussian-blur -> separable convolution with sigma 1.4.
#   4. canny         -> edge detection at thresholds 100 / 200.
#   5. probe         -> emit metadata (rows / cols / channels / dtype).
#
# Usage:
#   ./cv2-pipeline.sh <input_image> [output_dir]
#
# Environment overrides:
#   KSIZE      Gaussian blur kernel size (default: 5).
#   SIGMA      Gaussian sigma (default: 1.4).
#   THRESH1    Canny low threshold (default: 100).
#   THRESH2    Canny high threshold (default: 200).
#
# Requires: oximedia-cv2 on PATH.

set -euo pipefail

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input_image> [output_dir]" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found" >&2
    exit 3
fi

OUTPUT_DIR="${2:-./cv2-out}"
KSIZE="${KSIZE:-5}"
SIGMA="${SIGMA:-1.4}"
THRESH1="${THRESH1:-100}"
THRESH2="${THRESH2:-200}"

mkdir -p "$OUTPUT_DIR"

NORMALIZED="${OUTPUT_DIR}/01_normalized.png"
GRAY="${OUTPUT_DIR}/02_gray.png"
BLURRED="${OUTPUT_DIR}/03_blurred.png"
EDGES="${OUTPUT_DIR}/04_edges.png"

echo "-> imread (source -> PNG round-trip)"
oximedia-cv2 imread "$INPUT" "$NORMALIZED" --flags color

echo "-> cvt-color BGR -> Gray"
oximedia-cv2 cvt-color "$NORMALIZED" "$GRAY" --code bgr2gray

echo "-> gaussian-blur ksize=${KSIZE} sigma=${SIGMA}"
oximedia-cv2 gaussian-blur "$GRAY" "$BLURRED" --ksize "$KSIZE" --sigma "$SIGMA"

echo "-> canny t1=${THRESH1} t2=${THRESH2}"
oximedia-cv2 canny "$BLURRED" "$EDGES" --threshold1 "$THRESH1" --threshold2 "$THRESH2"

echo "-> probe (final edge map metadata)"
oximedia-cv2 probe "$EDGES"

echo
echo "OK: cv2 pipeline outputs in ${OUTPUT_DIR}"
ls -la "$OUTPUT_DIR"
