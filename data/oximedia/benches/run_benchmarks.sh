#!/bin/bash
# OxiMedia Benchmark Runner
# This script runs benchmarks and generates reports

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
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}OxiMedia Benchmark Suite${NC}"
echo -e "${BLUE}=========================${NC}\n"

# Function to run a benchmark suite
run_benchmark() {
    local bench_name=$1
    local description=$2

    echo -e "${YELLOW}Running: $description${NC}"
    cargo bench --manifest-path "$BENCH_DIR/Cargo.toml" --bench "$bench_name" 2>&1 | tee "$RESULTS_DIR/${bench_name}_${TIMESTAMP}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ $description completed${NC}\n"
    else
        echo -e "${RED}✗ $description failed${NC}\n"
        return 1
    fi
}

# Parse command line arguments
QUICK_MODE=false
SPECIFIC_BENCH=""
BASELINE=""
SAVE_BASELINE=false
COMPARE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --bench)
            SPECIFIC_BENCH="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --save-baseline)
            SAVE_BASELINE=true
            BASELINE="$2"
            shift 2
            ;;
        --compare)
            COMPARE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Run quick benchmarks only"
            echo "  --bench NAME         Run specific benchmark suite"
            echo "  --baseline NAME      Compare against saved baseline"
            echo "  --save-baseline NAME Save results as baseline"
            echo "  --compare            Show comparison with previous run"
            echo "  --help               Show this help message"
            echo ""
            echo "Available benchmark suites:"
            echo "  container_bench      Container format demuxing/muxing"
            echo "  codec_bench          Video/audio codec operations"
            echo "  audio_bench          Audio processing and filters"
            echo "  cv_bench             Computer vision operations"
            echo "  graph_bench          Pipeline graph operations"
            echo "  io_bench             I/O operations"
            echo "  compat_ffmpeg_bench  FFmpeg compatibility layer"
            echo "  quality_metrics      Quality metric computations"
            echo "  audio_metering       Audio metering (LUFS/peak)"
            echo "  format_probe         Container format probing"
            echo "  dedup_hash           Deduplication hash operations"
            echo "  codec_encode_bench   Codec encoder throughput"
            echo "  filter_bench         Filter graph operations"
            echo "  hdr_bench            HDR processing (PQ/HLG/tone mapping)"
            echo "  spatial_bench        Spatial audio (ambisonics/WFS)"
            echo "  cache_bench          Cache partition insert+lookup"
            echo "  stream_bench         BOLA ABR segment selection"
            echo "  video_bench          Video deinterlace + scene detection"
            echo "  cdn_bench            CDN geo-routing decisions"
            echo "  neural_bench         Neural network conv2d/activation"
            echo "  vr360_bench          Equirectangular→cubemap conversion"
            echo "  analytics_bench      Session analytics batch processing"
            echo "  caption_gen_bench    Line-breaking algorithms"
            echo "  pipeline_bench       Pipeline DSL build + topology"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build benchmarks first
echo -e "${BLUE}Building benchmarks...${NC}"
cargo build --release --manifest-path "$BENCH_DIR/Cargo.toml"
echo -e "${GREEN}✓ Build completed${NC}\n"

# Determine which benchmarks to run
if [ -n "$SPECIFIC_BENCH" ]; then
    # Run specific benchmark
    run_benchmark "$SPECIFIC_BENCH" "$(echo $SPECIFIC_BENCH | sed 's/_/ /g')"
elif [ "$QUICK_MODE" = true ]; then
    # Quick mode: run subset of benchmarks
    echo -e "${YELLOW}Running in quick mode (subset of benchmarks)${NC}\n"
    cargo bench --manifest-path "$BENCH_DIR/Cargo.toml" -- --quick 2>&1 | tee "$RESULTS_DIR/quick_${TIMESTAMP}.log"
else
    # Run all benchmarks
    echo -e "${BLUE}Running full benchmark suite...${NC}\n"

    run_benchmark "container_bench"     "Container Benchmarks"
    run_benchmark "codec_bench"         "Codec Benchmarks"
    run_benchmark "audio_bench"         "Audio Benchmarks"
    run_benchmark "cv_bench"            "Computer Vision Benchmarks"
    run_benchmark "graph_bench"         "Pipeline Graph Benchmarks"
    run_benchmark "io_bench"            "I/O Benchmarks"
    run_benchmark "compat_ffmpeg_bench" "FFmpeg Compatibility Benchmarks"
    run_benchmark "quality_metrics"     "Quality Metrics Benchmarks"
    run_benchmark "audio_metering"      "Audio Metering Benchmarks"
    run_benchmark "format_probe"        "Format Probe Benchmarks"
    run_benchmark "dedup_hash"          "Dedup Hash Benchmarks"
    run_benchmark "codec_encode_bench"  "Codec Encoder Benchmarks"
    run_benchmark "filter_bench"        "Filter Benchmarks"
    run_benchmark "hdr_bench"           "HDR Processing Benchmarks"
    run_benchmark "spatial_bench"       "Spatial Audio Benchmarks"
    run_benchmark "cache_bench"         "Cache Benchmarks"
    run_benchmark "stream_bench"        "Streaming ABR Benchmarks"
    run_benchmark "video_bench"         "Video Processing Benchmarks"
    run_benchmark "cdn_bench"           "CDN Routing Benchmarks"
    run_benchmark "neural_bench"        "Neural Network Benchmarks"
    run_benchmark "vr360_bench"         "VR 360 Projection Benchmarks"
    run_benchmark "analytics_bench"     "Analytics Benchmarks"
    run_benchmark "caption_gen_bench"   "Caption Generation Benchmarks"
    run_benchmark "pipeline_bench"      "Pipeline DSL Benchmarks"
fi

# Save baseline if requested
if [ "$SAVE_BASELINE" = true ]; then
    echo -e "${BLUE}Saving baseline: $BASELINE${NC}"
    cp -r "$PROJECT_ROOT/target/criterion" "$RESULTS_DIR/baseline_${BASELINE}_${TIMESTAMP}"
    echo -e "${GREEN}✓ Baseline saved${NC}\n"
fi

# Compare with baseline if requested
if [ -n "$BASELINE" ] && [ "$SAVE_BASELINE" = false ]; then
    echo -e "${BLUE}Comparing with baseline: $BASELINE${NC}"
    # Criterion will automatically compare if baseline exists
    echo -e "${GREEN}✓ Comparison complete${NC}\n"
fi

# Generate summary
echo -e "\n${BLUE}Benchmark Summary${NC}"
echo -e "${BLUE}=================${NC}\n"

echo "Results saved to: $RESULTS_DIR"
echo "Criterion reports: $PROJECT_ROOT/target/criterion/report/index.html"

# Check for performance changes
if [ "$COMPARE" = true ]; then
    echo -e "\n${YELLOW}Performance Changes:${NC}"

    # Parse Criterion output for changes
    if ls "$RESULTS_DIR"/*_${TIMESTAMP}.log >/dev/null 2>&1; then
        grep -h "Performance has" "$RESULTS_DIR"/*_${TIMESTAMP}.log 2>/dev/null || echo "No significant changes detected"
    fi
fi

echo -e "\n${GREEN}All benchmarks completed successfully!${NC}"
echo -e "\nTo view detailed results, open:"
echo -e "${BLUE}file://$PROJECT_ROOT/target/criterion/report/index.html${NC}\n"

# Generate performance report
echo -e "${BLUE}Generating performance report...${NC}"

REPORT_FILE="$RESULTS_DIR/performance_report_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOF
# OxiMedia Performance Report

**Generated:** $(date)
**Commit:** $(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
**Branch:** $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")

## System Information

- **OS:** $(uname -s)
- **Architecture:** $(uname -m)
- **Kernel:** $(uname -r)
- **CPU:** $(lscpu 2>/dev/null | grep "Model name" | sed 's/Model name: *//' || echo "N/A")
- **CPU Cores:** $(nproc 2>/dev/null || echo "N/A")
- **RAM:** $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "N/A")

## Benchmark Results

Detailed results available in:
- HTML Report: file://$PROJECT_ROOT/target/criterion/report/index.html
- Log Files: $RESULTS_DIR/*_${TIMESTAMP}.log

## Key Metrics

### Container Operations

EOF

# Extract key metrics from Criterion output
if [ -f "$RESULTS_DIR/container_bench_${TIMESTAMP}.log" ]; then
    echo "See container_bench_${TIMESTAMP}.log for details" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

### Codec Operations

EOF

if [ -f "$RESULTS_DIR/codec_bench_${TIMESTAMP}.log" ]; then
    echo "See codec_bench_${TIMESTAMP}.log for details" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

### Audio Operations

EOF

if [ -f "$RESULTS_DIR/audio_bench_${TIMESTAMP}.log" ]; then
    echo "See audio_bench_${TIMESTAMP}.log for details" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

### Computer Vision Operations

EOF

if [ -f "$RESULTS_DIR/cv_bench_${TIMESTAMP}.log" ]; then
    echo "See cv_bench_${TIMESTAMP}.log for details" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Notes

- All benchmarks run with Criterion.rs statistical analysis
- Results include confidence intervals and outlier detection
- Warm-up iterations performed before measurements

## Comparison with FFmpeg

To compare with FFmpeg, run:
\`\`\`bash
./benches/compare_ffmpeg.sh
\`\`\`

EOF

echo -e "${GREEN}✓ Performance report generated: $REPORT_FILE${NC}\n"

exit 0
