#!/usr/bin/env bash
# Generate code coverage reports for fuzz targets
# Shows which code paths are being exercised by fuzzing

set -euo pipefail

# Configuration
TARGET="${1:-}"
OUTPUT_DIR="${OUTPUT_DIR:-coverage}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Change to fuzz directory
cd "$(dirname "$0")/.."

echo "========================================="
echo "OxiMedia Fuzzing Coverage Report"
echo "========================================="

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target>"
    echo ""
    echo "Available targets:"
    cargo fuzz list
    exit 1
fi

echo "Target: $TARGET"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Generate coverage
echo -e "${YELLOW}Generating coverage data...${NC}"
cargo fuzz coverage "$TARGET"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate HTML report if llvm-cov is available
if command -v llvm-cov &> /dev/null; then
    echo -e "${YELLOW}Generating HTML coverage report...${NC}"

    # Find the profraw file
    PROFRAW=$(find target -name "*.profraw" | head -n 1)

    if [ -n "$PROFRAW" ]; then
        # Find the binary
        BINARY=$(find target -type f -name "$TARGET" | grep "coverage" | head -n 1)

        if [ -n "$BINARY" ]; then
            # Convert profraw to profdata
            llvm-profdata merge -sparse "$PROFRAW" -o "$OUTPUT_DIR/coverage.profdata"

            # Generate HTML report
            llvm-cov show "$BINARY" \
                -instr-profile="$OUTPUT_DIR/coverage.profdata" \
                -format=html \
                -output-dir="$OUTPUT_DIR/html" \
                -show-line-counts-or-regions \
                -show-instantiations

            echo -e "${GREEN}HTML report generated: $OUTPUT_DIR/html/index.html${NC}"

            # Generate summary
            llvm-cov report "$BINARY" \
                -instr-profile="$OUTPUT_DIR/coverage.profdata" \
                > "$OUTPUT_DIR/summary.txt"

            echo ""
            echo "Coverage Summary:"
            cat "$OUTPUT_DIR/summary.txt"
        else
            echo "Error: Could not find coverage binary for $TARGET"
            exit 1
        fi
    else
        echo "Error: No profraw file found"
        exit 1
    fi
else
    echo "Warning: llvm-cov not found. Only basic coverage data available."
fi

echo ""
echo -e "${GREEN}Coverage report complete${NC}"
