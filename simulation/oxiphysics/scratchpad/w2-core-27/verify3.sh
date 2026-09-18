#!/bin/bash
cd /Users/kitasan/work/oxiphysics || exit 1
export CARGO_TARGET_DIR=/Users/kitasan/work/oxiphysics/target-lane-b
LOG=scratchpad/w2-core-27
cargo clippy -p oxiphysics-core --all-targets -- -D warnings > "$LOG/clippy2.log" 2>&1
echo "exit: $?" >> "$LOG/clippy2.log"
