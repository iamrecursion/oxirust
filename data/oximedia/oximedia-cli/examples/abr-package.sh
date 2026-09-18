#!/usr/bin/env bash
# abr-package.sh - Multi-rendition adaptive-bitrate packaging (HLS + DASH).
#
# `oximedia package` performs the rendition ladder generation in one shot
# when given `--ladders 1080p,720p,480p`. We invoke it twice: once for HLS
# (fMP4) and once for DASH, producing a complete CDN-ready output tree.
#
# Usage:
#   ./abr-package.sh <input> [output_dir]
#
# Environment overrides:
#   LADDERS    Comma-separated ladder rungs (default: 1080p,720p,480p).
#   SEGMENT    Segment duration in seconds (default: 6).
#   ENCRYPT    none | aes128 | sample-aes | cenc (default: none).
#   LL_HLS     "1" to enable LL-HLS chunked output.
#
# Requires: oximedia on PATH.

set -euo pipefail

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input> [output_dir]" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found" >&2
    exit 3
fi

OUTPUT_DIR="${2:-./abr-out}"
LADDERS="${LADDERS:-1080p,720p,480p}"
SEGMENT="${SEGMENT:-6}"
ENCRYPT="${ENCRYPT:-none}"
LL_HLS="${LL_HLS:-0}"

mkdir -p "${OUTPUT_DIR}/hls"
mkdir -p "${OUTPUT_DIR}/dash"

LL_FLAG=()
if [ "$LL_HLS" = "1" ]; then
    LL_FLAG=(--low-latency)
fi

echo "-> Packaging HLS (fMP4) ladder=${LADDERS} segments=${SEGMENT}s encrypt=${ENCRYPT}..."
oximedia package \
    -i "$INPUT" \
    -o "${OUTPUT_DIR}/hls" \
    --format hls-fmp4 \
    --ladders "$LADDERS" \
    --segments "$SEGMENT" \
    --encrypt "$ENCRYPT" \
    "${LL_FLAG[@]}"

echo "-> Packaging DASH ladder=${LADDERS} segments=${SEGMENT}s encrypt=${ENCRYPT}..."
oximedia package \
    -i "$INPUT" \
    -o "${OUTPUT_DIR}/dash" \
    --format dash \
    --ladders "$LADDERS" \
    --segments "$SEGMENT" \
    --encrypt "$ENCRYPT"

echo "OK: ABR package written to ${OUTPUT_DIR}"
echo "    HLS:  ${OUTPUT_DIR}/hls"
echo "    DASH: ${OUTPUT_DIR}/dash"
