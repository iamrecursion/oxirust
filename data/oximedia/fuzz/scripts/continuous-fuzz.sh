#!/usr/bin/env bash
# Continuous fuzzing script for long-term fuzzing campaigns
# Runs fuzzers in parallel across all available CPU cores

set -euo pipefail

# Configuration
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"  # Number of fuzzers to run in parallel
MEMORY_LIMIT="${MEMORY_LIMIT:-2048}"  # 2GB memory limit per fuzzer
LOG_DIR="${LOG_DIR:-logs}"

# Colors for output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# Change to fuzz directory
cd "$(dirname "$0")/.."

echo "========================================="
echo "OxiMedia Continuous Fuzzing"
echo "========================================="
echo "Parallel jobs: ${PARALLEL_JOBS}"
echo "Memory limit: ${MEMORY_LIMIT}MB per job"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Get list of all fuzz targets
TARGETS=$(cargo fuzz list)

if [ -z "$TARGETS" ]; then
    echo "ERROR: No fuzz targets found"
    exit 1
fi

echo "Found $(echo "$TARGETS" | wc -w) fuzz targets"
echo ""

# Function to run a single fuzzer
run_fuzzer() {
    local target=$1
    local log_file="$LOG_DIR/${target}.log"

    echo -e "${YELLOW}Starting continuous fuzzing: $target${NC}"
    echo "  Log: $log_file"

    # Determine dictionary file if it exists
    local dict_arg=""
    case "$target" in
        matroska_parser)
            if [ -f "dictionaries/matroska.dict" ]; then
                dict_arg="-dict=dictionaries/matroska.dict"
            fi
            ;;
        ogg_parser)
            if [ -f "dictionaries/ogg.dict" ]; then
                dict_arg="-dict=dictionaries/ogg.dict"
            fi
            ;;
        av1_decoder)
            if [ -f "dictionaries/av1.dict" ]; then
                dict_arg="-dict=dictionaries/av1.dict"
            fi
            ;;
        hls_parser)
            if [ -f "dictionaries/hls.dict" ]; then
                dict_arg="-dict=dictionaries/hls.dict"
            fi
            ;;
        dash_parser)
            if [ -f "dictionaries/dash.dict" ]; then
                dict_arg="-dict=dictionaries/dash.dict"
            fi
            ;;
        rtmp_parser)
            if [ -f "dictionaries/rtmp.dict" ]; then
                dict_arg="-dict=dictionaries/rtmp.dict"
            fi
            ;;
    esac

    # Run the fuzzer with fork mode for parallel execution
    cargo fuzz run "$target" \
        -- -fork="$PARALLEL_JOBS" \
           -rss_limit_mb="$MEMORY_LIMIT" \
           -print_final_stats=1 \
           $dict_arg \
        > "$log_file" 2>&1 &

    echo "$!" > "$LOG_DIR/${target}.pid"
}

# Function to stop all fuzzers
stop_all() {
    echo ""
    echo "Stopping all fuzzers..."

    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" || true
            fi
            rm "$pid_file"
        fi
    done

    echo -e "${GREEN}All fuzzers stopped${NC}"
    exit 0
}

# Set up signal handling
trap stop_all SIGINT SIGTERM

# Start fuzzers for each target
for target in $TARGETS; do
    run_fuzzer "$target"
    sleep 2  # Stagger starts slightly
done

echo ""
echo -e "${GREEN}All fuzzers started${NC}"
echo "Logs are being written to: $LOG_DIR/"
echo "Press Ctrl+C to stop all fuzzers"
echo ""

# Wait for any fuzzer to exit
wait

# If we get here, a fuzzer exited (possibly due to a crash)
echo ""
echo "A fuzzer exited. Stopping all fuzzers..."
stop_all
