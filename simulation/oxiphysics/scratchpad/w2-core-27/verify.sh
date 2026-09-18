#!/bin/bash
# Verification pipeline for package w2-core-27 (runs in the lane b build directory).
cd /Users/kitasan/work/oxiphysics || exit 1
export CARGO_TARGET_DIR=/Users/kitasan/work/oxiphysics/target-lane-b
LOG=scratchpad/w2-core-27

cargo nextest run -p oxiphysics-core topology_optimization --no-fail-fast > "$LOG/nextest4.log" 2>&1
echo "exit: $?" >> "$LOG/nextest4.log"

cargo clippy -p oxiphysics-core --all-targets -- -D warnings > "$LOG/clippy1.log" 2>&1
echo "exit: $?" >> "$LOG/clippy1.log"

cargo test --doc -p oxiphysics-core topology_optimization > "$LOG/doctest1.log" 2>&1
echo "exit: $?" >> "$LOG/doctest1.log"

echo "all done" > "$LOG/verify.done"
