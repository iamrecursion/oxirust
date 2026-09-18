#!/usr/bin/env bash
# Guards the wasm32 dependency tree of the four oximedia-web modules
# against unreviewed additions. Every crate name reachable on the
# wasm32-unknown-unknown target must appear in web/allowed-deps.txt;
# adding a new one requires a size-justified edit to that file.
#
# A short list of crates is hard-forbidden regardless of allowed-deps.txt
# (heavyweight runtimes/serialization that would blow the size budget, and
# any non-web oximedia-* crate that would drag in native-only deps).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALLOWED_FILE="$ROOT/allowed-deps.txt"

if [ ! -f "$ALLOWED_FILE" ]; then
    echo "error: $ALLOWED_FILE not found" >&2
    exit 1
fi

TREE_OUTPUT="$(cargo tree \
    --manifest-path "$ROOT/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --prefix none \
    --no-dedupe \
    -e normal \
    -p oximedia-web-scopes \
    -p oximedia-web-color \
    -p oximedia-web-scale \
    -p oximedia-web-quality)"

FOUND_DEPS="$(printf '%s\n' "$TREE_OUTPUT" | awk '{print $1}' | sort -u)"

# --- hard-forbidden list, independent of allowed-deps.txt ---
HARD_FORBIDDEN="tokio rayon reqwest sqlx serde_json serde rand getrandom"

hard_hit=0
for dep in $FOUND_DEPS; do
    for forbidden in $HARD_FORBIDDEN; do
        if [ "$dep" = "$forbidden" ]; then
            echo "error: hard-forbidden dependency '$dep' present in wasm32 dep tree" >&2
            hard_hit=1
        fi
    done
    case "$dep" in
        oximedia-web-*) ;;
        oximedia-*)
            echo "error: hard-forbidden dependency '$dep' — only oximedia-web-* crates may appear in the wasm32 dep tree" >&2
            hard_hit=1
            ;;
    esac
done

if [ "$hard_hit" -eq 1 ]; then
    exit 1
fi

# --- allowed-deps.txt gate ---
NEW_DEPS="$(comm -13 <(sort -u "$ALLOWED_FILE") <(printf '%s\n' "$FOUND_DEPS"))"

if [ -n "$NEW_DEPS" ]; then
    echo "error: new dependencies not present in web/allowed-deps.txt:" >&2
    printf '%s\n' "$NEW_DEPS" | sed 's/^/    /' >&2
    echo "adding requires editing web/allowed-deps.txt with a size justification" >&2
    exit 1
fi

echo "dep gate passed ($(printf '%s\n' "$FOUND_DEPS" | wc -l | tr -d '[:space:]') crates in wasm32 dep tree)"
