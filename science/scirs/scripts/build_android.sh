#!/usr/bin/env bash
# Build SciRS2 for Android targets
# Copyright: COOLJAPAN OU (Team KitaSan)

set -e

# Configuration
PROJECT_NAME="scirs2"
BUILD_DIR="target/android"
JNI_LIBS_DIR="$BUILD_DIR/jniLibs"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_status() {
    echo -e "${GREEN}==>${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

echo_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Check prerequisites
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$NDK_HOME" ]; then
    echo_error "Android NDK not found. Please set ANDROID_NDK_HOME or NDK_HOME"
    echo "Download from: https://developer.android.com/ndk/downloads"
    exit 1
fi

NDK_PATH="${ANDROID_NDK_HOME:-$NDK_HOME}"
echo_status "Using Android NDK: $NDK_PATH"

# Setup NDK toolchain
export PATH="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH"

# Verify NDK tools
if ! command -v aarch64-linux-android-clang &> /dev/null; then
    echo_error "Android NDK toolchain not found in PATH"
    exit 1
fi

# Parse arguments
RELEASE_MODE="--release"
FEATURES="mobile,android"
CRATE="scirs2-core"
BUILD_ALL_ABIS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            RELEASE_MODE=""
            shift
            ;;
        --crate)
            CRATE="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --abi)
            ABI="$2"
            BUILD_ALL_ABIS=false
            shift 2
            ;;
        *)
            echo_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo_status "Building SciRS2 for Android..."
echo "  Crate: $CRATE"
echo "  Features: $FEATURES"
echo "  Mode: $([ -z "$RELEASE_MODE" ] && echo "debug" || echo "release")"

# Clean previous builds
echo_status "Cleaning previous Android builds..."
rm -rf "$BUILD_DIR"
mkdir -p "$JNI_LIBS_DIR"

# Determine build profile directory
PROFILE_DIR=$([ -z "$RELEASE_MODE" ] && echo "debug" || echo "release")

# Build function
build_for_target() {
    local target=$1
    local abi=$2

    echo_status "Building for $abi ($target)..."

    cargo build --target "$target" \
        --package "$CRATE" \
        --features "$FEATURES" \
        $RELEASE_MODE

    # Copy library to jniLibs
    mkdir -p "$JNI_LIBS_DIR/$abi"
    cp "target/$target/$PROFILE_DIR/lib${CRATE//-/_}.so" \
        "$JNI_LIBS_DIR/$abi/lib${CRATE//-/_}.so"
}

# Build for Android architectures
if [ "$BUILD_ALL_ABIS" = true ]; then
    build_for_target "aarch64-linux-android" "arm64-v8a"
    build_for_target "armv7-linux-androideabi" "armeabi-v7a"
    build_for_target "i686-linux-android" "x86"
    build_for_target "x86_64-linux-android" "x86_64"
else
    case $ABI in
        arm64-v8a)
            build_for_target "aarch64-linux-android" "arm64-v8a"
            ;;
        armeabi-v7a)
            build_for_target "armv7-linux-androideabi" "armeabi-v7a"
            ;;
        x86)
            build_for_target "i686-linux-android" "x86"
            ;;
        x86_64)
            build_for_target "x86_64-linux-android" "x86_64"
            ;;
        *)
            echo_error "Unknown ABI: $ABI"
            exit 1
            ;;
    esac
fi

# Generate JNI headers using cbindgen
echo_status "Generating JNI headers..."
if command -v cbindgen &> /dev/null; then
    cbindgen --config cbindgen.toml \
        --crate "$CRATE" \
        --lang c \
        --output "$BUILD_DIR/scirs2_jni.h" 2>/dev/null || \
        echo_warning "cbindgen failed, header generation skipped"
else
    echo_warning "cbindgen not found, header generation skipped"
fi

# Create AAR structure
echo_status "Creating AAR structure..."
AAR_DIR="$BUILD_DIR/aar"
mkdir -p "$AAR_DIR"
cp -r "$JNI_LIBS_DIR" "$AAR_DIR/"

# Create AndroidManifest.xml
cat > "$AAR_DIR/AndroidManifest.xml" << EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.cooljapan.scirs2"
    android:versionCode="1"
    android:versionName="0.2.0">

    <uses-sdk android:minSdkVersion="21" android:targetSdkVersion="34" />
</manifest>
EOF

echo ""
echo_status "Android build complete!"
echo "  JNI libraries: $JNI_LIBS_DIR"
echo "  AAR structure: $AAR_DIR"
echo ""
echo "Integration:"
echo "  1. Copy jniLibs to your Android project's src/main/ directory"
echo "  2. Add JNI interface code to load the library"
echo "  3. In build.gradle, add: implementation files('libs/scirs2.aar')"
echo ""
echo "Available ABIs:"
ls -1 "$JNI_LIBS_DIR"
echo ""
