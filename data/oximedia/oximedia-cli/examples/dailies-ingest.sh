#!/usr/bin/env bash
# dailies-ingest.sh - Camera-to-MAM ingest pipeline.
#
# Probes a raw camera file, dumps metadata, builds a thumbnail and a sprite
# sheet, generates a low-bitrate proxy, and registers the asset in a JSON
# MAM catalog. Useful at the end of a shoot day for daily review packages.
#
# Usage:
#   ./dailies-ingest.sh <input.mxf> [output_dir]
#
# Environment overrides:
#   OUTPUT_DIR        Override default ./dailies output location.
#   PROXY_QUALITY     low | medium | high (default: medium).
#   PROXY_RESOLUTION  quarter | half | full (default: quarter).
#   PROXY_CODEC       vp9 | av1 (default: vp9).
#   CATALOG           Path to MAM JSON catalog (default: <output_dir>/catalog.json).
#
# Requires:
#   oximedia on PATH (cargo install --path /path/to/oximedia/oximedia-cli or
#   a symlink into target/release/oximedia).

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

OUTPUT_DIR="${2:-${OUTPUT_DIR:-./dailies}}"
PROXY_QUALITY="${PROXY_QUALITY:-medium}"
PROXY_RESOLUTION="${PROXY_RESOLUTION:-quarter}"
PROXY_CODEC="${PROXY_CODEC:-vp9}"
CATALOG="${CATALOG:-${OUTPUT_DIR}/catalog.json}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/proxies"

base="$(basename "$INPUT")"
base="${base%.*}"

echo "-> Probing ${INPUT}..."
oximedia probe -i "$INPUT" --format json --streams --metadata \
    > "${OUTPUT_DIR}/${base}.probe.json"

echo "-> Extracting metadata..."
# `metadata` uses the global --json flag; --show prints the current sidecar.
oximedia --json metadata -i "$INPUT" --show \
    > "${OUTPUT_DIR}/${base}.metadata.json"

echo "-> Generating thumbnail..."
oximedia thumbnail -i "$INPUT" -o "${OUTPUT_DIR}/${base}.thumb.jpg" \
    --mode single --format jpeg --quality 90

echo "-> Generating sprite sheet (4x4 grid)..."
oximedia sprite -i "$INPUT" -o "${OUTPUT_DIR}/${base}.sprite.jpg" \
    --rows 4 --cols 4 --format jpeg --quality 85 --vtt --manifest

echo "-> Creating ${PROXY_RESOLUTION}/${PROXY_QUALITY} proxy (${PROXY_CODEC})..."
oximedia proxy generate \
    -i "$INPUT" \
    -o "${OUTPUT_DIR}/proxies" \
    --resolution "$PROXY_RESOLUTION" \
    --quality "$PROXY_QUALITY" \
    --codec "$PROXY_CODEC" \
    --db "${OUTPUT_DIR}/proxies/proxy.db.json"

echo "-> Registering in MAM catalog (${CATALOG})..."
oximedia mam ingest \
    -i "$INPUT" \
    --catalog "$CATALOG" \
    --extract-metadata \
    --tags "dailies,unverified"

echo "OK: dailies for ${base} written to ${OUTPUT_DIR}"
