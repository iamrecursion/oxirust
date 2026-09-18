#!/usr/bin/env bash
# Builds the oximedia-web wasm modules with wasm-pack and stages the
# generated artifacts (plus the hand-written JS wrappers) under dist/.
#
# Usage:
#   ./scripts/build.sh                # build all modules
#   ./scripts/build.sh scopes color   # build a subset
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ALL_MODULES=(scopes color scale quality)

if [ "$#" -gt 0 ]; then
    MODULES=("$@")
else
    MODULES=("${ALL_MODULES[@]}")
fi

for m in "${MODULES[@]}"; do
    valid=0
    for candidate in "${ALL_MODULES[@]}"; do
        if [ "$m" = "$candidate" ]; then
            valid=1
            break
        fi
    done
    if [ "$valid" -ne 1 ]; then
        echo "error: unknown module '$m' (expected one of: ${ALL_MODULES[*]})" >&2
        exit 1
    fi
done

for m in "${MODULES[@]}"; do
    crate_dir="crates/oximedia-web-$m"
    out_dir="$ROOT/dist/wasm/$m"
    echo "==> building oximedia-web-$m"
    (
        cd "$ROOT"
        wasm-pack build "$crate_dir" \
            --release \
            --target web \
            --out-dir "$out_dir" \
            --out-name "oximedia_web_$m"
    )

    # wasm-pack emits its own package.json / .gitignore into the out-dir;
    # dist/ is gitignored wholesale by web/, and we don't want an npm
    # package manifest fighting with the top-level web/package.json.
    rm -f "$out_dir/package.json" "$out_dir/.gitignore"
done

mkdir -p "$ROOT/dist"

shopt -s nullglob
js_files=("$ROOT"/js/*.js "$ROOT"/js/*.d.ts)
shopt -u nullglob

if [ "${#js_files[@]}" -eq 0 ]; then
    echo "warning: no files under web/js/*.js or web/js/*.d.ts yet — skipping wrapper copy" >&2
else
    cp "${js_files[@]}" "$ROOT/dist/"
fi

echo "==> build complete: ${MODULES[*]}"
