#!/usr/bin/env bash
# loudness-normalize.sh - Batch EBU R128 loudness normalization with verify.
#
# For every audio/video file in a directory:
#   1. Pre-analyze with `loudness analyze` to record the original LUFS/LRA/TP.
#   2. Normalize via `normalize process` to a target integrated loudness.
#   3. Verify the result with `loudness check --strict` (exits 1 on failure).
#
# Usage:
#   ./loudness-normalize.sh <input_dir> [output_dir]
#
# Environment overrides:
#   STANDARD       ebu | atsc | spotify | youtube | apple | netflix | tidal |
#                  amazon | bbc | podcast | ... (default: ebu).
#   TARGET_LUFS    Integrated target (default: -23.0 for EBU R128).
#   TRUE_PEAK      dBTP ceiling (default: -1.0).
#   EXTENSIONS     Space-separated list (default: "wav flac mp3 aac m4a mka").
#
# Requires: oximedia on PATH.

set -euo pipefail

INPUT_DIR="${1:-}"
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: $0 <input_dir> [output_dir]" >&2
    exit 2
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: $INPUT_DIR is not a directory" >&2
    exit 3
fi

OUTPUT_DIR="${2:-./normalized}"
STANDARD="${STANDARD:-ebu}"
TARGET_LUFS="${TARGET_LUFS:--23.0}"
TRUE_PEAK="${TRUE_PEAK:--1.0}"
EXTENSIONS="${EXTENSIONS:-wav flac mp3 aac m4a mka}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/reports"

processed=0
failed=0

# Build a regex of accepted extensions for `find -E`-portable matching.
build_find_args() {
    local args=()
    for ext in $EXTENSIONS; do
        args+=( -iname "*.${ext}" -o )
    done
    # drop trailing -o
    if [ ${#args[@]} -gt 0 ]; then
        unset 'args[${#args[@]}-1]'
    fi
    printf '%s\n' "${args[@]}"
}

mapfile -t FIND_ARGS < <(build_find_args)

# Iterate matching files (NUL-safe).
while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    name="${base%.*}"
    ext="${base##*.}"
    out="${OUTPUT_DIR}/${name}.norm.${ext}"
    pre_report="${OUTPUT_DIR}/reports/${name}.pre.json"
    post_report="${OUTPUT_DIR}/reports/${name}.post.json"

    echo "==> ${base}"

    echo "    -> pre-analysis (${STANDARD})"
    oximedia loudness analyze -i "$f" --target "$STANDARD" \
        --output-format json > "$pre_report"

    echo "    -> normalizing to ${TARGET_LUFS} LUFS / ${TRUE_PEAK} dBTP"
    oximedia normalize process \
        -i "$f" -o "$out" \
        --target "$TARGET_LUFS" \
        --true-peak "$TRUE_PEAK" \
        --output-format json \
        > "${OUTPUT_DIR}/reports/${name}.process.json"

    echo "    -> post-verify (--strict)"
    if oximedia loudness check -i "$out" --standard "$STANDARD" --strict \
        > "$post_report" 2>&1; then
        processed=$((processed + 1))
        echo "    OK"
    else
        failed=$((failed + 1))
        echo "    FAIL: post-normalize compliance check failed (see ${post_report})" >&2
    fi
done < <(find "$INPUT_DIR" -type f \( "${FIND_ARGS[@]}" \) -print0)

echo
echo "Summary: ${processed} processed, ${failed} failed"
echo "Outputs: ${OUTPUT_DIR}"
exit "$failed"
