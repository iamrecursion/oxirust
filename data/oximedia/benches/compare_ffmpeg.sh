#!/bin/bash
# Compare OxiMedia performance with FFmpeg
# This script runs equivalent operations in both OxiMedia and FFmpeg

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BENCH_DIR")"
TEST_DATA_DIR="$PROJECT_ROOT/test_data"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$TEST_DATA_DIR"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}OxiMedia vs FFmpeg Comparison${NC}"
echo -e "${BLUE}==============================${NC}\n"

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg not found. Please install FFmpeg first.${NC}"
    exit 1
fi

FFMPEG_VERSION=$(ffmpeg -version | head -n1)
echo -e "${GREEN}Found: $FFMPEG_VERSION${NC}\n"

# Generate test files if they don't exist
generate_test_files() {
    echo -e "${BLUE}Generating test media files...${NC}"

    # Generate 720p test video
    if [ ! -f "$TEST_DATA_DIR/test_720p.webm" ]; then
        echo "Creating 720p test video..."
        ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 \
               -c:v libvpx-vp9 -b:v 1M -threads 1 \
               "$TEST_DATA_DIR/test_720p.webm" -y 2>/dev/null
    fi

    # Generate 1080p test video
    if [ ! -f "$TEST_DATA_DIR/test_1080p.webm" ]; then
        echo "Creating 1080p test video..."
        ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 \
               -c:v libvpx-vp9 -b:v 2M -threads 1 \
               "$TEST_DATA_DIR/test_1080p.webm" -y 2>/dev/null
    fi

    # Generate audio test file
    if [ ! -f "$TEST_DATA_DIR/test_audio.opus" ]; then
        echo "Creating Opus test audio..."
        ffmpeg -f lavfi -i sine=frequency=440:duration=10 \
               -c:a libopus -b:a 128k \
               "$TEST_DATA_DIR/test_audio.opus" -y 2>/dev/null
    fi

    # Generate FLAC test file
    if [ ! -f "$TEST_DATA_DIR/test_audio.flac" ]; then
        echo "Creating FLAC test audio..."
        ffmpeg -f lavfi -i sine=frequency=440:duration=10 \
               -c:a flac \
               "$TEST_DATA_DIR/test_audio.flac" -y 2>/dev/null
    fi

    echo -e "${GREEN}✓ Test files generated${NC}\n"
}

# Benchmark FFmpeg demuxing
benchmark_ffmpeg_demux() {
    local input_file=$1
    local description=$2

    echo -e "${YELLOW}FFmpeg Demux: $description${NC}"

    local start_time=$(date +%s%N)
    ffmpeg -benchmark -i "$input_file" -f null - 2>&1 | grep "utime" || true
    local end_time=$(date +%s%N)

    local duration_ms=$(( (end_time - start_time) / 1000000 ))
    local file_size=$(stat -f%z "$input_file" 2>/dev/null || stat -c%s "$input_file")
    local throughput=$(echo "scale=2; $file_size / 1024 / 1024 / ($duration_ms / 1000)" | bc)

    echo "  Duration: ${duration_ms}ms"
    echo "  Throughput: ${throughput} MB/s"
    echo ""
}

# Benchmark FFmpeg decoding
benchmark_ffmpeg_decode() {
    local input_file=$1
    local description=$2

    echo -e "${YELLOW}FFmpeg Decode: $description${NC}"

    # Decode to raw video, measure FPS
    local output=$(ffmpeg -benchmark -i "$input_file" -f rawvideo -pix_fmt yuv420p - 2>&1)

    # Extract FPS from benchmark output
    local fps=$(echo "$output" | grep "fps=" | tail -1 | sed -n 's/.*fps=\s*\([0-9.]*\).*/\1/p')

    if [ -n "$fps" ]; then
        echo "  FPS: $fps"
    fi

    echo ""
}

# Benchmark audio decoding
benchmark_ffmpeg_audio_decode() {
    local input_file=$1
    local description=$2

    echo -e "${YELLOW}FFmpeg Audio Decode: $description${NC}"

    local start_time=$(date +%s%N)
    ffmpeg -benchmark -i "$input_file" -f null - 2>&1 > /dev/null
    local end_time=$(date +%s%N)

    local duration_ms=$(( (end_time - start_time) / 1000000 ))

    # Get audio duration
    local audio_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file")
    local realtime_factor=$(echo "scale=2; $audio_duration * 1000 / $duration_ms" | bc)

    echo "  Duration: ${duration_ms}ms"
    echo "  Real-time factor: ${realtime_factor}x"
    echo ""
}

# Generate test files
generate_test_files

# Create comparison report
REPORT_FILE="$RESULTS_DIR/ffmpeg_comparison_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOF
# OxiMedia vs FFmpeg Performance Comparison

**Generated:** $(date)
**FFmpeg Version:** $FFMPEG_VERSION

## System Information

- **OS:** $(uname -s)
- **Architecture:** $(uname -m)
- **CPU:** $(lscpu 2>/dev/null | grep "Model name" | sed 's/Model name: *//' || echo "N/A")
- **CPU Cores:** $(nproc 2>/dev/null || echo "N/A")

## Container Demuxing

### WebM/Matroska

EOF

echo -e "\n${BLUE}=== Container Demuxing ===${NC}\n"

# Benchmark 720p demux
benchmark_ffmpeg_demux "$TEST_DATA_DIR/test_720p.webm" "720p WebM" | tee -a "$REPORT_FILE"

# Benchmark 1080p demux
benchmark_ffmpeg_demux "$TEST_DATA_DIR/test_1080p.webm" "1080p WebM" | tee -a "$REPORT_FILE"

cat >> "$REPORT_FILE" << EOF

## Video Decoding

### VP9

EOF

echo -e "\n${BLUE}=== Video Decoding ===${NC}\n"

# Benchmark 720p decode
benchmark_ffmpeg_decode "$TEST_DATA_DIR/test_720p.webm" "720p VP9" | tee -a "$REPORT_FILE"

# Benchmark 1080p decode
benchmark_ffmpeg_decode "$TEST_DATA_DIR/test_1080p.webm" "1080p VP9" | tee -a "$REPORT_FILE"

cat >> "$REPORT_FILE" << EOF

## Audio Decoding

### Opus

EOF

echo -e "\n${BLUE}=== Audio Decoding ===${NC}\n"

# Benchmark Opus decode
benchmark_ffmpeg_audio_decode "$TEST_DATA_DIR/test_audio.opus" "Opus 128kbps" | tee -a "$REPORT_FILE"

cat >> "$REPORT_FILE" << EOF

### FLAC

EOF

# Benchmark FLAC decode
benchmark_ffmpeg_audio_decode "$TEST_DATA_DIR/test_audio.flac" "FLAC" | tee -a "$REPORT_FILE"

cat >> "$REPORT_FILE" << EOF

## Notes

- FFmpeg benchmarks run with \`-benchmark\` flag
- Single-threaded decoding (when applicable)
- Results may vary based on system load

## Comparison Summary

| Operation | FFmpeg | OxiMedia Target | Status |
|-----------|--------|-----------------|--------|
| WebM Demux (720p) | See above | 80%+ of FFmpeg | Target |
| VP9 Decode (720p) | See above | 70%+ of FFmpeg | Target |
| Opus Decode | See above | 80%+ of FFmpeg | Target |
| FLAC Decode | See above | 85%+ of FFmpeg | Target |

## Running OxiMedia Benchmarks

To run corresponding OxiMedia benchmarks:

\`\`\`bash
cargo bench --manifest-path benches/Cargo.toml
\`\`\`

Or use the convenience script:

\`\`\`bash
./benches/run_benchmarks.sh
\`\`\`

EOF

echo -e "${GREEN}✓ Comparison complete!${NC}\n"
echo -e "Report saved to: ${BLUE}$REPORT_FILE${NC}\n"
echo -e "To run OxiMedia benchmarks for comparison:"
echo -e "${YELLOW}./benches/run_benchmarks.sh${NC}\n"

exit 0
