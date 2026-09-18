#!/usr/bin/env bash
# live-broadcast.sh - Production switcher + scheduled playout demo chain.
#
# Demonstrates the broadcast surface:
#
#   1. `switcher create`       -> instantiate an M/E switcher session.
#   2. `switcher add-source`   -> wire up several inputs (cameras, NDI, files).
#   3. `switcher switch`       -> cut/mix to a chosen input.
#   4. `switcher record start` -> capture the program output to file.
#   5. `playout schedule`      -> author a JSON schedule.
#   6. `playout start`         -> run the playout server against the schedule.
#
# Each `switcher` invocation is stateless from the shell's perspective; they
# operate on the long-running daemon (see `oximedia switcher --help`).
#
# Usage:
#   ./live-broadcast.sh <output_dir>
#
# Environment overrides:
#   FPS              25 | 29.97 | 30 | 50 | 59.94 | 60 (default: 25).
#   ME_ROWS          1-4 (default: 1).
#   INPUTS           2-40 (default: 4).
#   PRESET           basic | professional | broadcast.
#   SCHEDULE_DATE    YYYY-MM-DD (default: today).
#   CHANNEL          Playout channel name (default: "Channel 1").
#
# Requires: oximedia on PATH.

set -euo pipefail

OUTPUT_DIR="${1:-./live-out}"
FPS="${FPS:-25}"
ME_ROWS="${ME_ROWS:-1}"
INPUTS="${INPUTS:-4}"
SCHEDULE_DATE="${SCHEDULE_DATE:-$(date -u +%Y-%m-%d)}"
CHANNEL="${CHANNEL:-Channel 1}"

mkdir -p "$OUTPUT_DIR"

PROGRAM_REC="${OUTPUT_DIR}/program.mkv"
SCHEDULE="${OUTPUT_DIR}/schedule.json"

echo "-> switcher create (me_rows=${ME_ROWS}, inputs=${INPUTS}, fps=${FPS})"
if [ -n "${PRESET:-}" ]; then
    oximedia switcher create \
        --me-rows "$ME_ROWS" \
        --inputs "$INPUTS" \
        --fps "$FPS" \
        --preset "$PRESET"
else
    oximedia switcher create \
        --me-rows "$ME_ROWS" \
        --inputs "$INPUTS" \
        --fps "$FPS"
fi

echo "-> switcher add-source: cam-a (sdi)"
oximedia switcher add-source \
    --name "cam-a" \
    --source-type sdi \
    --uri "sdi://0/1" \
    --slot 0

echo "-> switcher add-source: cam-b (ndi)"
oximedia switcher add-source \
    --name "cam-b" \
    --source-type ndi \
    --uri "ndi://STUDIO-A (Camera B)" \
    --slot 1

echo "-> switcher add-source: bgcard (test_pattern)"
oximedia switcher add-source \
    --name "bg-card" \
    --source-type test_pattern \
    --slot 2

echo "-> switcher preview cam-b"
oximedia switcher preview --input 1

echo "-> switcher switch (mix transition over 30 frames)"
oximedia switcher switch \
    --input 1 \
    --transition mix \
    --duration 30

echo "-> switcher record start -> ${PROGRAM_REC}"
oximedia switcher record start \
    --output "$PROGRAM_REC" \
    --codec av1

echo "-> playout schedule -> ${SCHEDULE}"
oximedia playout schedule \
    --output "$SCHEDULE" \
    --channel "$CHANNEL" \
    --format hd1080p25 \
    --date "$SCHEDULE_DATE"

echo "-> playout start (genlock=internal, monitor on :9090)"
oximedia playout start \
    --schedule "$SCHEDULE" \
    --clock-source internal \
    --buffer-size 10 \
    --monitor-port 9090

echo "OK: switcher recording to ${PROGRAM_REC}; playout running for ${CHANNEL}"
echo "    To stop: oximedia switcher record stop  &&  oximedia playout stop --channel '${CHANNEL}'"
