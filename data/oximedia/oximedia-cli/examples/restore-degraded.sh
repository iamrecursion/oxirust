#!/usr/bin/env bash
# restore-degraded.sh - Restore a degraded asset: analyze -> denoise ->
# stabilize -> upscale.
#
# Pipeline (each step writes an intermediate file in a tempdir):
#
#   1. `restore analyze`     -> JSON degradation report.
#   2. `denoise`             -> reduce noise / grain.
#   3. `stabilize`           -> remove camera shake.
#   4. `scaling upscale`     -> Lanczos upscale to target resolution.
#
# `restore video --mode full` is an all-in-one alternative wired through
# `oximedia-restore`. We use the explicit chain here so each stage's flags
# (mode / strength / motion model / algorithm) are visible.
#
# Usage:
#   ./restore-degraded.sh <input> <output> [--width 1920 --height 1080]
#
# Environment overrides:
#   DENOISE_MODE     fast | balanced | quality | grain-aware (default: balanced).
#   DENOISE_STRENGTH 0.0 - 1.0 (default: 0.5).
#   STAB_MOTION      translation | affine | perspective | 3d (default: affine).
#   STAB_QUALITY     fast | balanced | maximum (default: balanced).
#   UPSCALE_ALG      bilinear | bicubic | lanczos (default: lanczos).
#
# Requires: oximedia on PATH.

set -euo pipefail

INPUT="${1:-}"
OUTPUT="${2:-}"
shift 2 || true

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input> <output> [--width W --height H]" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found" >&2
    exit 3
fi

WIDTH=1920
HEIGHT=1080
while [ $# -gt 0 ]; do
    case "$1" in
        --width)  WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

DENOISE_MODE="${DENOISE_MODE:-balanced}"
DENOISE_STRENGTH="${DENOISE_STRENGTH:-0.5}"
STAB_MOTION="${STAB_MOTION:-affine}"
STAB_QUALITY="${STAB_QUALITY:-balanced}"
UPSCALE_ALG="${UPSCALE_ALG:-lanczos}"

WORK="$(mktemp -d -t oximedia-restore-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

DENOISED="${WORK}/denoised.mkv"
STABILIZED="${WORK}/stabilized.mkv"

echo "-> restore analyze (degradation report)"
oximedia --json restore analyze \
    -i "$INPUT" \
    --analysis-type auto \
    > "${WORK}/analysis.json" || true
cat "${WORK}/analysis.json" 2>/dev/null || true

echo "-> denoise (mode=${DENOISE_MODE} strength=${DENOISE_STRENGTH})"
oximedia denoise \
    -i "$INPUT" \
    -o "$DENOISED" \
    --mode "$DENOISE_MODE" \
    --strength "$DENOISE_STRENGTH"

echo "-> stabilize (motion=${STAB_MOTION} quality=${STAB_QUALITY})"
oximedia stabilize \
    -i "$DENOISED" \
    -o "$STABILIZED" \
    --mode "$STAB_MOTION" \
    --quality "$STAB_QUALITY" \
    --zoom

echo "-> scaling upscale (${WIDTH}x${HEIGHT}, ${UPSCALE_ALG})"
oximedia scaling upscale \
    -i "$STABILIZED" \
    -o "$OUTPUT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --algorithm "$UPSCALE_ALG"

echo "OK: restored asset written to ${OUTPUT}"
