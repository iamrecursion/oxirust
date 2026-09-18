#!/usr/bin/env bash
# Build script for `OxiMedia` WASM (the monolithic oximedia-wasm crate).
#
# Publishes ONLY the bundler-target build as @cooljapan/oximedia on npm.
# pkg-web/ and pkg-node/ are also built here for local testing/inspection,
# but they are NOT published packages -- do not `npm install` them, and do
# not rename them to @cooljapan/oximedia-* names. In particular,
# @cooljapan/oximedia-web is a separate, independently-versioned package
# (see /web in this monorepo) and must not be confused with pkg-web/ here.

set -e

SCOPE="cooljapan"
NPM_NAME="@${SCOPE}/oximedia"

echo "Building OxiMedia WASM (publishing target: ${NPM_NAME})..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed"
    echo "Install it with: cargo install wasm-pack"
    exit 1
fi

# Build for web target (local/dev use only -- not published)
echo "Building for web target (local use only, not published)..."
wasm-pack build --scope "${SCOPE}" --target web --out-dir pkg-web --release

# Build for Node.js target (local/dev use only -- not published)
echo "Building for Node.js target (local use only, not published)..."
wasm-pack build --scope "${SCOPE}" --target nodejs --out-dir pkg-node --release

# Build for bundler target (webpack, etc.) -- this is the published target
echo "Building for bundler target (the published npm package)..."
wasm-pack build --scope "${SCOPE}" --target bundler --out-dir pkg-bundler --release

# Only the bundler build is renamed/published as @cooljapan/oximedia.
# pkg-web/ and pkg-node/ keep wasm-pack's default scoped name
# (@cooljapan/oximedia-wasm) so their package.json never claims to be an
# installable package under a name nobody publishes.
if [ -f "pkg-bundler/package.json" ]; then
    sed -i '' "s|\"@${SCOPE}/oximedia-wasm\"|\"${NPM_NAME}\"|g" "pkg-bundler/package.json"
    echo "  Updated pkg-bundler/package.json → ${NPM_NAME}"
fi

echo ""
echo "Build complete!"
echo ""
echo "Output directories:"
echo "  - pkg-bundler/  : Published to npm as ${NPM_NAME} (webpack/rollup/vite/etc.)"
echo "  - pkg-web/      : Local/dev use only, NOT published (package.json name:"
echo "                    @${SCOPE}/oximedia-wasm). For a published browser-native"
echo "                    package, use @cooljapan/oximedia-web from /web instead."
echo "  - pkg-node/     : Local/dev use only, NOT published (package.json name:"
echo "                    @${SCOPE}/oximedia-wasm)."
echo ""
echo "To publish (bundler target is the only published npm package):"
echo "  cd pkg-bundler && npm publish --access public"
