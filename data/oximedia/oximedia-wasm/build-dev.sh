#!/usr/bin/env bash
# Development build script for `OxiMedia` WASM
# This builds in debug mode with debug symbols.
#
# Output is local/dev use only and is never published to npm -- this script
# intentionally does not pass --scope or rename package.json, so the
# generated package.json keeps wasm-pack's default unscoped name
# ("oximedia-wasm"), which cannot be confused with the published
# @cooljapan/oximedia package or with the separate @cooljapan/oximedia-web
# package (see /web in this monorepo).

set -e

echo "Building OxiMedia WASM (debug mode, local use only, not published)..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed"
    echo "Install it with: cargo install wasm-pack"
    exit 1
fi

# Build for web target in debug mode
echo "Building for web target (debug)..."
wasm-pack build --dev --target web --out-dir pkg-web-dev

echo "Debug build complete!"
echo "Output: pkg-web-dev/ (local use only, not published)"
