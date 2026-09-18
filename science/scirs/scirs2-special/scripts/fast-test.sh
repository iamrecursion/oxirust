#!/bin/bash
# Fast development testing script
# Optimizes compilation speed for rapid iteration

echo "🚀 Fast Development Mode"
echo "========================"

# Enable fast test mode
export FAST_TESTS=1

# Use fast compilation mode with minimal features
echo "Running tests with fast configuration..."
RUSTFLAGS="-C opt-level=0 -C debuginfo=0" cargo nextest run --features fast-compile

echo "Done! Tests completed in fast mode."
echo "For full testing, use: cargo nextest run"