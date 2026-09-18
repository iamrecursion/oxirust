#!/usr/bin/env bash
# forensics-investigate.sh - Tamper / dedup / watermark investigation chain.
#
# Runs three orthogonal forensic passes on a single asset:
#
#   1. `forensics --all` for ELA, noise, compression, splicing, metadata,
#      tampering analysis.
#   2. `dedup hash --perceptual` for content + perceptual fingerprints.
#   3. `watermark detect` to surface any embedded watermark payload.
#   4. (optional) `watermark verify` against an EXPECTED_WATERMARK message.
#
# Usage:
#   ./forensics-investigate.sh <input> [report_dir]
#
# Environment overrides:
#   EXPECTED_WATERMARK   If set, runs `watermark verify --expected ...` too.
#   HASH_ALG             blake3 | sha256 | sha512 | xxhash (default: blake3).
#   FORENSICS_TESTS      Comma list (overrides --all) e.g. "ela,noise,metadata".
#
# Requires: oximedia on PATH.

set -euo pipefail

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input> [report_dir]" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found" >&2
    exit 3
fi

REPORT_DIR="${2:-./forensics-report}"
HASH_ALG="${HASH_ALG:-blake3}"
mkdir -p "$REPORT_DIR"

base="$(basename "$INPUT")"
base="${base%.*}"

echo "-> forensics: tamper / integrity / provenance"
if [ -n "${FORENSICS_TESTS:-}" ]; then
    oximedia forensics \
        -i "$INPUT" \
        --tests "$FORENSICS_TESTS" \
        --output-format json \
        --report "${REPORT_DIR}/${base}.forensics.json" \
        > "${REPORT_DIR}/${base}.forensics.summary.json"
else
    oximedia forensics \
        -i "$INPUT" \
        --all \
        --output-format json \
        --report "${REPORT_DIR}/${base}.forensics.json" \
        > "${REPORT_DIR}/${base}.forensics.summary.json"
fi

echo "-> dedup hash (${HASH_ALG} + perceptual)"
oximedia --json dedup hash \
    -i "$INPUT" \
    --algorithm "$HASH_ALG" \
    --perceptual \
    > "${REPORT_DIR}/${base}.hash.json"

echo "-> watermark detect"
oximedia --json watermark detect \
    -i "$INPUT" \
    > "${REPORT_DIR}/${base}.watermark.json" || {
        echo "    note: watermark detect returned non-zero (no watermark found?)"
}

if [ -n "${EXPECTED_WATERMARK:-}" ]; then
    echo "-> watermark verify (expected: ${EXPECTED_WATERMARK})"
    if oximedia --json watermark verify \
        -i "$INPUT" \
        --expected "$EXPECTED_WATERMARK" \
        > "${REPORT_DIR}/${base}.watermark.verify.json"; then
        echo "    VERIFIED"
    else
        echo "    MISMATCH (see ${REPORT_DIR}/${base}.watermark.verify.json)" >&2
    fi
fi

echo "OK: forensic reports written to ${REPORT_DIR}"
ls -la "$REPORT_DIR"
