#!/bin/bash
# Poll until `cargo check -p oxiphysics-softbody --all-targets` succeeds (other agents are mid-edit in the shared tree).
cd /Users/kitasan/work/oxiphysics || exit 2
for attempt in $(seq 1 30); do
  CARGO_TARGET_DIR=/Users/kitasan/work/oxiphysics/target-lane-d cargo check -p oxiphysics-softbody --all-targets > scratchpad/w1b-softbody-06/poll_check.log 2>&1
  status=$?
  echo "attempt=$attempt exit=$status $(date '+%H:%M:%S')" >> scratchpad/w1b-softbody-06/poll_status.log
  if [ $status -eq 0 ]; then exit 0; fi
  sleep 60
done
exit 1
