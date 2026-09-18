#!/usr/bin/env bash
# quality-check.sh - Transcode then compare to the source with VMAF/SSIM/PSNR.
#
# Encodes a copy of the input at a chosen codec/CRF, then runs full-reference
# quality metrics between the original and the encode. Outputs a JSON report
# that downstream tooling (CI gates, per-rendition QC) can parse.
#
# Usage:
#   ./quality-check.sh <input> [report.json]
#
# Environment overrides:
#   CODEC         av1 | vp9 (default: vp9). Patent-free codecs only.
#   CRF           Quality (lower = better). Default: 30 for vp9, 32 for av1.
#   PRESET        ultrafast..veryslow (default: medium).
#   METRICS       Comma list passed to `quality compare`.
#                 Default: psnr,ssim,ms-ssim,vmaf
#
# Requires: oximedia on PATH.

set -euo pipefail

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input> [report.json]" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found" >&2
    exit 3
fi

REPORT="${2:-./quality-report.json}"
CODEC="${CODEC:-vp9}"
PRESET="${PRESET:-medium}"
METRICS="${METRICS:-psnr,ssim,ms-ssim,vmaf}"

# Default CRF differs by codec family.
if [ -z "${CRF:-}" ]; then
    case "$CODEC" in
        av1) CRF=32 ;;
        vp9) CRF=30 ;;
        *)   CRF=30 ;;
    esac
fi

# Workspace (cleaned up at exit).
WORK="$(mktemp -d -t oximedia-qc-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
ENCODED="${WORK}/encoded.mkv"

echo "-> Encoding reference -> distorted (${CODEC} CRF ${CRF}, preset ${PRESET})..."
oximedia transcode \
    -i "$INPUT" \
    -o "$ENCODED" \
    --codec "$CODEC" \
    --crf "$CRF" \
    --preset "$PRESET" \
    --overwrite

echo "-> Comparing reference vs distorted with metrics: ${METRICS}..."
oximedia quality compare \
    --reference "$INPUT" \
    --distorted "$ENCODED" \
    --metrics "$METRICS" \
    --output-format json \
    > "$REPORT"

echo "OK: quality report written to ${REPORT}"
echo "    head: $(head -c 200 "$REPORT")"
