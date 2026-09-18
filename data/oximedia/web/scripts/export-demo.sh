#!/usr/bin/env bash
# Exports the OxiScope demo as a flattened, self-contained static site
# suitable for hosting under any URL prefix (e.g. /oxiscope/):
#
#   <target-dir>/index.html ...   web/demo/* browser files, flattened
#   <target-dir>/dist/            web/dist/* (JS wrappers + wasm/ subtree)
#   <target-dir>/bench/           web/bench/{index.html,bench.js,lib/}
#                                 (browser-served parts only — no run.sh,
#                                 no results/)
#
# In-repo the demo lives at web/demo/ and reaches its modules via '../dist';
# the flattened copy sits next to its own dist/, so every '../dist'
# reference in the top-level copied files is rewritten to './dist', and the
# footer's '../bench/' link to './bench/'. bench/bench.js keeps its
# '../dist' imports untouched: from <target-dir>/bench/ they already
# resolve to <target-dir>/dist/. Source files under web/ are never
# modified — only the copies are rewritten.
#
# Usage:
#   ./scripts/export-demo.sh <target-dir>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <target-dir>" >&2
    exit 1
fi

TARGET="$1"

DEMO="$ROOT/demo"
DIST="$ROOT/dist"
BENCH="$ROOT/bench"

for d in "$DEMO" "$DIST" "$BENCH"; do
    if [ ! -d "$d" ]; then
        echo "error: $d does not exist — run web/scripts/build.sh first" >&2
        exit 1
    fi
done

MODULES="scopes color scale quality"
for m in $MODULES; do
    if [ ! -f "$DIST/wasm/$m/oximedia_web_${m}_bg.wasm" ]; then
        echo "error: missing $DIST/wasm/$m/oximedia_web_${m}_bg.wasm — run web/scripts/build.sh first" >&2
        exit 1
    fi
done

mkdir -p "$TARGET"

# The dist/ and bench/ subtrees are wholly derived — recreate them from
# scratch so repeated exports never leave stale files behind.
rm -rf "$TARGET/dist" "$TARGET/bench"

echo "==> copying web/demo/ (flattened, browser files only)"
shopt -s nullglob
demo_files=("$DEMO"/*.html "$DEMO"/*.js "$DEMO"/*.css)
shopt -u nullglob
if [ "${#demo_files[@]}" -eq 0 ]; then
    echo "error: no browser files (*.html, *.js, *.css) under $DEMO" >&2
    exit 1
fi
cp "${demo_files[@]}" "$TARGET/"

echo "==> copying web/dist/"
mkdir -p "$TARGET/dist"
cp -R "$DIST"/. "$TARGET/dist/"

echo "==> copying web/bench/ (browser-served parts only)"
mkdir -p "$TARGET/bench"
cp "$BENCH/index.html" "$BENCH/bench.js" "$TARGET/bench/"
cp -R "$BENCH/lib" "$TARGET/bench/lib"

echo "==> rewriting '../dist' -> './dist' and '../bench/' -> './bench/' in the flattened copies"
rewrites=0
for f in "$TARGET"/*.html "$TARGET"/*.js "$TARGET"/*.css; do
    [ -f "$f" ] || continue
    hits="$(grep -c -e '\.\./dist' -e '\.\./bench/' "$f" || true)"
    if [ "$hits" -gt 0 ]; then
        grep -n -e '\.\./dist' -e '\.\./bench/' "$f" \
            | sed -e "s|^|    $(basename "$f"):|"
        tmp="$f.rewrite.tmp"
        sed -e 's|\.\./dist|./dist|g' -e 's|\.\./bench/|./bench/|g' "$f" > "$tmp"
        mv "$tmp" "$f"
        rewrites=$((rewrites + hits))
    fi
done
echo "    rewrote $rewrites reference(s)"

# Assert nothing at the flattened top level still points one level up.
if grep -n '\.\./dist' "$TARGET"/*.html "$TARGET"/*.js "$TARGET"/*.css; then
    echo "error: '../dist' references remain in the flattened copy" >&2
    exit 1
fi

file_count=0
total_bytes=0
while IFS= read -r f; do
    size="$(wc -c < "$f" | tr -d '[:space:]')"
    file_count=$((file_count + 1))
    total_bytes=$((total_bytes + size))
done < <(find "$TARGET" -type f)

echo "==> exported $file_count files, $total_bytes bytes total -> $TARGET"
