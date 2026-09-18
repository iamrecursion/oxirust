#!/usr/bin/env bash
# Minimize crash inputs to the smallest reproducing case
# Makes debugging easier by removing unnecessary bytes

set -euo pipefail

# Configuration
TARGET="${1:-}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
MINIMIZED_DIR="${MINIMIZED_DIR:-minimized}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Change to fuzz directory
cd "$(dirname "$0")/.."

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target>"
    echo ""
    echo "Available targets:"
    cargo fuzz list
    exit 1
fi

echo "========================================="
echo "Minimizing Crash Inputs"
echo "========================================="
echo "Target: $TARGET"
echo ""

# Check if artifacts exist
if [ ! -d "$ARTIFACTS_DIR/$TARGET" ]; then
    echo -e "${YELLOW}No artifacts found for $TARGET${NC}"
    exit 0
fi

# Count crashes
CRASH_COUNT=$(find "$ARTIFACTS_DIR/$TARGET" -type f | wc -l)

if [ "$CRASH_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No crashes to minimize${NC}"
    exit 0
fi

echo "Found $CRASH_COUNT crash input(s)"
echo ""

# Create minimized directory
mkdir -p "$MINIMIZED_DIR/$TARGET"

# Minimize each crash
for crash_file in "$ARTIFACTS_DIR/$TARGET"/*; do
    if [ -f "$crash_file" ]; then
        crash_name=$(basename "$crash_file")
        output_file="$MINIMIZED_DIR/$TARGET/$crash_name"

        echo -e "${YELLOW}Minimizing: $crash_name${NC}"

        # Get original size
        original_size=$(wc -c < "$crash_file")

        # Run minimization
        if cargo fuzz tmin "$TARGET" "$crash_file" -- -minimize_timeout=60; then
            # The minimized output is written to the same path
            # We'll copy it to our minimized directory
            minimized_size=$(wc -c < "$crash_file")

            cp "$crash_file" "$output_file"

            reduction=$((100 * (original_size - minimized_size) / original_size))

            echo -e "${GREEN}  Original: $original_size bytes${NC}"
            echo -e "${GREEN}  Minimized: $minimized_size bytes${NC}"
            echo -e "${GREEN}  Reduction: $reduction%${NC}"
        else
            echo -e "${RED}  Minimization failed${NC}"
        fi
        echo ""
    fi
done

echo -e "${GREEN}Minimization complete${NC}"
echo "Minimized inputs saved to: $MINIMIZED_DIR/$TARGET/"
