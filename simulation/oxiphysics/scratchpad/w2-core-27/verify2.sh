#!/bin/bash
cd /Users/kitasan/work/oxiphysics || exit 1
export CARGO_TARGET_DIR=/Users/kitasan/work/oxiphysics/target-lane-b
LOG=scratchpad/w2-core-27
cargo nextest run -p oxiphysics-core topology_optimization --no-fail-fast > "$LOG/nextest5.log" 2>&1
echo "exit: $?" >> "$LOG/nextest5.log"
