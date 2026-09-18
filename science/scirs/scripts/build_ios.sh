#!/usr/bin/env bash
# Build SciRS2 for iOS targets
# Copyright: COOLJAPAN OU (Team KitaSan)

set -e

# Configuration
PROJECT_NAME="scirs2"
BUILD_DIR="target/ios"
FRAMEWORK_DIR="$BUILD_DIR/SciRS2.xcframework"

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
if ! command -v xcodebuild &> /dev/null; then
    echo_error "Xcode not found. Please install Xcode from App Store"
    exit 1
fi

if ! command -v cargo-lipo &> /dev/null; then
    echo_warning "cargo-lipo not found. Installing..."
    cargo install cargo-lipo
fi

# Parse arguments
RELEASE_MODE="--release"
FEATURES="mobile,ios"
CRATE="scirs2-core"

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
        *)
            echo_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo_status "Building SciRS2 for iOS..."
echo "  Crate: $CRATE"
echo "  Features: $FEATURES"
echo "  Mode: $([ -z "$RELEASE_MODE" ] && echo "debug" || echo "release")"

# Clean previous builds
echo_status "Cleaning previous iOS builds..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Build for iOS device (ARM64)
echo_status "Building for iOS device (ARM64)..."
cargo build --target aarch64-apple-ios \
    --package "$CRATE" \
    --features "$FEATURES" \
    $RELEASE_MODE

# Build for iOS simulator (ARM64)
echo_status "Building for iOS simulator (ARM64)..."
cargo build --target aarch64-apple-ios-sim \
    --package "$CRATE" \
    --features "$FEATURES" \
    $RELEASE_MODE

# Build for iOS simulator (x86_64)
echo_status "Building for iOS simulator (x86_64)..."
cargo build --target x86_64-apple-ios \
    --package "$CRATE" \
    --features "$FEATURES" \
    $RELEASE_MODE

# Determine build profile directory
PROFILE_DIR=$([ -z "$RELEASE_MODE" ] && echo "debug" || echo "release")

# Create universal library for simulators
echo_status "Creating universal simulator library..."
mkdir -p "$BUILD_DIR/simulator"
lipo -create \
    "target/aarch64-apple-ios-sim/$PROFILE_DIR/lib${CRATE//-/_}.a" \
    "target/x86_64-apple-ios/$PROFILE_DIR/lib${CRATE//-/_}.a" \
    -output "$BUILD_DIR/simulator/lib${CRATE//-/_}.a"

# Copy device library
echo_status "Copying device library..."
mkdir -p "$BUILD_DIR/device"
cp "target/aarch64-apple-ios/$PROFILE_DIR/lib${CRATE//-/_}.a" \
    "$BUILD_DIR/device/"

# Generate C headers using cbindgen
echo_status "Generating C headers..."
if command -v cbindgen &> /dev/null; then
    cbindgen --config cbindgen.toml \
        --crate "$CRATE" \
        --output "$BUILD_DIR/SciRS2.h" 2>/dev/null || \
        echo_warning "cbindgen failed, header generation skipped"
else
    echo_warning "cbindgen not found, header generation skipped"
fi

# Create XCFramework
echo_status "Creating XCFramework..."
rm -rf "$FRAMEWORK_DIR"
xcodebuild -create-xcframework \
    -library "$BUILD_DIR/device/lib${CRATE//-/_}.a" \
    -headers "$BUILD_DIR" \
    -library "$BUILD_DIR/simulator/lib${CRATE//-/_}.a" \
    -headers "$BUILD_DIR" \
    -output "$FRAMEWORK_DIR"

echo ""
echo_status "iOS build complete!"
echo "  Framework: $FRAMEWORK_DIR"
echo ""
echo "Integration:"
echo "  1. Drag $FRAMEWORK_DIR into your Xcode project"
echo "  2. Add to 'Frameworks, Libraries, and Embedded Content'"
echo "  3. Import in Swift: import SciRS2"
echo ""
