#!/usr/bin/env bash
# Setup mobile development toolchains for SciRS2 v0.2.0
# Copyright: COOLJAPAN OU (Team KitaSan)

set -e

echo "==> Setting up SciRS2 mobile development toolchains..."

# iOS targets
echo "==> Installing iOS targets..."
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim
rustup target add x86_64-apple-ios

# Android targets
echo "==> Installing Android targets..."
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add i686-linux-android
rustup target add x86_64-linux-android

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$NDK_HOME" ]; then
    echo "WARNING: Android NDK not found. Please set ANDROID_NDK_HOME or NDK_HOME"
    echo "Download from: https://developer.android.com/ndk/downloads"
else
    NDK_PATH="${ANDROID_NDK_HOME:-$NDK_HOME}"
    echo "==> Found Android NDK at: $NDK_PATH"

    # Setup NDK toolchain paths
    export PATH="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH"

    # Verify NDK tools
    if command -v aarch64-linux-android-clang &> /dev/null; then
        echo "✓ Android NDK toolchain configured successfully"
    else
        echo "WARNING: NDK tools not in PATH. You may need to update your shell configuration"
        echo "Add this to your ~/.zshrc or ~/.bashrc:"
        echo "export PATH=\"\$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin:\$PATH\""
    fi
fi

# Check for Xcode (iOS development)
if command -v xcodebuild &> /dev/null; then
    XCODE_VERSION=$(xcodebuild -version | head -n 1)
    echo "✓ Found $XCODE_VERSION"
else
    echo "WARNING: Xcode not found. Required for iOS development"
    echo "Install from App Store or https://developer.apple.com/xcode/"
fi

# Install cargo-lipo for iOS universal libraries
echo "==> Installing cargo-lipo for iOS universal library creation..."
cargo install --force cargo-lipo 2>/dev/null || echo "cargo-lipo already installed or failed to install"

# Install cbindgen for C header generation
echo "==> Installing cbindgen for FFI header generation..."
cargo install --force cbindgen 2>/dev/null || echo "cbindgen already installed or failed to install"

echo ""
echo "==> Mobile toolchain setup complete!"
echo ""
echo "Available targets:"
echo "  iOS (ARM64):     aarch64-apple-ios"
echo "  iOS (Simulator): aarch64-apple-ios-sim, x86_64-apple-ios"
echo "  Android (ARM64): aarch64-linux-android"
echo "  Android (ARMv7): armv7-linux-androideabi"
echo ""
echo "Next steps:"
echo "  1. For iOS: Run './scripts/build_ios.sh'"
echo "  2. For Android: Run './scripts/build_android.sh'"
echo ""
