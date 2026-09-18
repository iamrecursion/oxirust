#!/usr/bin/env bash
# Enforces the gzip -9 size budgets for the oximedia-web wasm modules.
#
# Env vars:
#   SIZE_GATE_SOFT=1       treat soft-budget breaches as non-fatal (hard
#                           ceilings still fail the gate)
#   SIZE_GATE_EMPTY_OK=1   exit 0 with a notice if dist/ does not exist at
#                           all yet (pre-build CI-of-one state)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"

MODULES="scopes color scale quality"

wasm_soft() {
    case "$1" in
        scopes) echo 153600 ;;
        color) echo 204800 ;;
        scale) echo 122880 ;;
        quality) echo 204800 ;;
        *) echo "0" ;;
    esac
}

wasm_hard() {
    case "$1" in
        scopes) echo 204800 ;;
        color) echo 256000 ;;
        scale) echo 163840 ;;
        quality) echo 256000 ;;
        *) echo "0" ;;
    esac
}

GLUE_SOFT=15360
GLUE_HARD=25600
TOTAL_SOFT=512000
TOTAL_HARD=614400

if [ ! -d "$DIST" ]; then
    if [ "${SIZE_GATE_EMPTY_OK:-0}" = "1" ]; then
        echo "notice: dist/ does not exist yet (nothing built) — SIZE_GATE_EMPTY_OK=1, passing"
        exit 0
    fi
    echo "error: dist/ does not exist — run web/scripts/build.sh first" >&2
    exit 1
fi

raw_size() {
    wc -c < "$1" | tr -d '[:space:]'
}

gzip_size() {
    gzip -9 -c "$1" | wc -c | tr -d '[:space:]'
}

report_offender() {
    module_dir="$1"
    if command -v twiggy >/dev/null 2>&1; then
        wasm_file="$DIST/wasm/$module_dir"/*_bg.wasm
        for f in $wasm_file; do
            if [ -f "$f" ]; then
                echo "    --- twiggy top -n 20 ($f) ---"
                twiggy top -n 20 "$f" || true
            fi
        done
    fi
}

printf '%-16s %10s %10s %10s/%-10s %s\n' "module" "raw" "gzip" "soft" "hard" "status"
printf '%-16s %10s %10s %10s %10s %s\n' "----------------" "----------" "----------" "----------" "----------" "------"

fail=0
soft_fail=0
missing=0
total_gzip=0

for m in $MODULES; do
    module_dir="$DIST/wasm/$m"
    wasm_file=""
    if [ -d "$module_dir" ]; then
        for f in "$module_dir"/*_bg.wasm; do
            if [ -f "$f" ]; then
                wasm_file="$f"
            fi
        done
        glue_file=""
        for f in "$module_dir"/oximedia_web_"$m".js; do
            if [ -f "$f" ]; then
                glue_file="$f"
            fi
        done
    fi

    if [ -z "$wasm_file" ]; then
        echo "error: missing artifact for module '$m' (expected $module_dir/oximedia_web_${m}_bg.wasm) — run web/scripts/build.sh first" >&2
        missing=1
        continue
    fi

    raw="$(raw_size "$wasm_file")"
    gz="$(gzip_size "$wasm_file")"
    soft="$(wasm_soft "$m")"
    hard="$(wasm_hard "$m")"
    total_gzip=$((total_gzip + gz))

    status="OK"
    if [ "$gz" -gt "$hard" ]; then
        status="HARD-BREACH"
        fail=1
        report_offender "$m"
    elif [ "$gz" -gt "$soft" ]; then
        status="soft-breach"
        soft_fail=1
    fi
    printf '%-16s %10s %10s %10s/%-10s %s\n' "$m.wasm" "$raw" "$gz" "$soft" "$hard" "$status"

    glue_raw=0
    glue_gz=0
    if [ -n "$glue_file" ]; then
        glue_raw=$((glue_raw + $(raw_size "$glue_file")))
        glue_gz=$((glue_gz + $(gzip_size "$glue_file")))
    fi
    wrapper_file="$DIST/$m.js"
    if [ -f "$wrapper_file" ]; then
        glue_raw=$((glue_raw + $(raw_size "$wrapper_file")))
        glue_gz=$((glue_gz + $(gzip_size "$wrapper_file")))
    fi
    total_gzip=$((total_gzip + glue_gz))

    glue_status="OK"
    if [ "$glue_gz" -gt "$GLUE_HARD" ]; then
        glue_status="HARD-BREACH"
        fail=1
    elif [ "$glue_gz" -gt "$GLUE_SOFT" ]; then
        glue_status="soft-breach"
        soft_fail=1
    fi
    printf '%-16s %10s %10s %10s/%-10s %s\n' "$m.glue" "$glue_raw" "$glue_gz" "$GLUE_SOFT" "$GLUE_HARD" "$glue_status"
done

if [ "$missing" -eq 1 ]; then
    exit 1
fi

total_status="OK"
if [ "$total_gzip" -gt "$TOTAL_HARD" ]; then
    total_status="HARD-BREACH"
    fail=1
elif [ "$total_gzip" -gt "$TOTAL_SOFT" ]; then
    total_status="soft-breach"
    soft_fail=1
fi
printf '%-16s %10s %10s %10s/%-10s %s\n' "TOTAL" "-" "$total_gzip" "$TOTAL_SOFT" "$TOTAL_HARD" "$total_status"

if [ "$fail" -eq 1 ]; then
    echo "error: size gate failed (hard ceiling breach)" >&2
    exit 1
fi

if [ "$soft_fail" -eq 1 ] && [ "${SIZE_GATE_SOFT:-0}" != "1" ]; then
    echo "error: size gate failed (soft budget breach; set SIZE_GATE_SOFT=1 to allow)" >&2
    exit 1
fi

echo "size gate passed"
