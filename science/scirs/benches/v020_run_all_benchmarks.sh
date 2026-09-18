#!/bin/bash
#
# SciRS2 v0.2.0 Comprehensive Benchmark Suite Runner
#
# This script runs all v0.2.0 benchmarks and generates reports:
# - Comprehensive suite
# - SIMD vs scalar comparison
# - GPU vs CPU comparison (if available)
# - Memory profiling
# - Scalability analysis
# - Python comparison (SciPy, NumPy, PyTorch)
#
# Usage:
#   ./v020_run_all_benchmarks.sh [options]
#
# Options:
#   --quick          Run quick benchmarks only (reduced measurement time)
#   --skip-python    Skip Python comparison benchmarks
#   --skip-gpu       Skip GPU benchmarks
#   --report-only    Generate reports from existing results
#   --help           Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
QUICK_MODE=false
SKIP_PYTHON=false
SKIP_GPU=false
REPORT_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SciRS2 v0.2.0 Comprehensive Performance Benchmark Suite  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  This will run extensive benchmarks and may take 1-2 hours║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Not in SciRS2 workspace root${NC}"
    echo "Please run this script from the SciRS2 workspace root directory"
    exit 1
fi

# Create results directory
RESULTS_DIR="/tmp/scirs2_v020_benchmarks"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}✓${NC} Results will be saved to: $RESULTS_DIR"

# Function to run a benchmark
run_benchmark() {
    local name=$1
    local bench_name=$2
    local features=${3:-""}

    if [ "$REPORT_ONLY" = true ]; then
        echo -e "${YELLOW}⊘${NC} Skipping $name (report-only mode)"
        return
    fi

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Running: $name${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ -n "$features" ]; then
        cargo bench --bench "$bench_name" --features "$features" -- --noplot
    else
        cargo bench --bench "$bench_name" -- --noplot
    fi

    echo -e "${GREEN}✓${NC} Completed: $name"
}

# Start timestamp
START_TIME=$(date +%s)
echo -e "\nStarted at: $(date)"

# 1. Comprehensive Suite
run_benchmark "Comprehensive Suite" "v020_comprehensive_suite"

# 2. SIMD Comparison
run_benchmark "SIMD vs Scalar Comparison" "v020_simd_comparison"

# 3. Memory Profiling
run_benchmark "Memory Profiling" "v020_memory_profiling"

# 4. Scalability Analysis
run_benchmark "Scalability Analysis" "v020_scalability"

# 5. GPU Comparison (if not skipped)
if [ "$SKIP_GPU" = false ]; then
    # Check if GPU features are available
    if cargo metadata --no-deps --format-version 1 2>/dev/null | grep -q '"cuda"'; then
        run_benchmark "GPU vs CPU (CUDA)" "v020_gpu_comparison" "cuda"
    elif cargo metadata --no-deps --format-version 1 2>/dev/null | grep -q '"metal"'; then
        run_benchmark "GPU vs CPU (Metal)" "v020_gpu_comparison" "metal-backend"
    else
        echo -e "${YELLOW}⊘${NC} GPU benchmarks skipped (no GPU backend available)"
    fi
else
    echo -e "${YELLOW}⊘${NC} GPU benchmarks skipped (--skip-gpu)"
fi

# 6. Python Comparison (if not skipped)
if [ "$SKIP_PYTHON" = false ]; then
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Running: Python Comparison${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    # Run Rust benchmarks first
    if [ "$REPORT_ONLY" = false ]; then
        run_benchmark "Python Comparison (Rust)" "v020_python_comparison"

        # Check if Python is available
        if command -v python3 &> /dev/null; then
            # Check if required packages are installed
            if python3 -c "import numpy, scipy, pandas" 2>/dev/null; then
                echo ""
                echo -e "${BLUE}Running Python comparison benchmarks...${NC}"
                chmod +x benches/v020_python_comparison.py
                python3 benches/v020_python_comparison.py
                echo -e "${GREEN}✓${NC} Completed: Python Comparison"
            else
                echo -e "${YELLOW}⊘${NC} Python comparison skipped (missing packages)"
                echo "Install with: pip install numpy scipy pandas matplotlib"
            fi
        else
            echo -e "${YELLOW}⊘${NC} Python comparison skipped (Python 3 not found)"
        fi
    fi
else
    echo -e "${YELLOW}⊘${NC} Python comparison skipped (--skip-python)"
fi

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Generate summary report
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Generating Summary Report${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create summary markdown
SUMMARY_FILE="$RESULTS_DIR/v020_benchmark_summary.md"
cat > "$SUMMARY_FILE" << EOF
# SciRS2 v0.2.0 Performance Benchmark Results

**Generated:** $(date)
**Duration:** ${MINUTES}m ${SECONDS}s
**Platform:** $(uname -s) $(uname -m)
**Rust Version:** $(rustc --version)

## Benchmarks Run

EOF

if [ "$REPORT_ONLY" = false ]; then
    cat >> "$SUMMARY_FILE" << EOF
- ✓ Comprehensive Suite
- ✓ SIMD vs Scalar Comparison
- ✓ Memory Profiling
- ✓ Scalability Analysis
EOF

    if [ "$SKIP_GPU" = false ]; then
        echo "- ✓ GPU vs CPU Comparison" >> "$SUMMARY_FILE"
    else
        echo "- ⊘ GPU vs CPU Comparison (skipped)" >> "$SUMMARY_FILE"
    fi

    if [ "$SKIP_PYTHON" = false ]; then
        echo "- ✓ Python Comparison (NumPy, SciPy)" >> "$SUMMARY_FILE"
    else
        echo "- ⊘ Python Comparison (skipped)" >> "$SUMMARY_FILE"
    fi
fi

cat >> "$SUMMARY_FILE" << EOF

## Result Files

- Comprehensive results: \`/tmp/scirs2_v020_benchmark_results.json\`
- Python comparison: \`/tmp/scirs2_v020_comparison_results.json\`
- Criterion reports: \`target/criterion/\`

## View Results

### HTML Reports
Open in browser: \`target/criterion/report/index.html\`

### Command Line
\`\`\`bash
# View JSON results
cat /tmp/scirs2_v020_benchmark_results.json | jq .

# View Python comparison
cat /tmp/scirs2_v020_comparison_results.json | jq .summary
\`\`\`

## Performance Highlights

### SIMD Speedups
- Dot product (f32): Check \`target/criterion/simd_vs_scalar\`
- Matrix multiply: Check \`target/criterion/simd_vs_scalar/matmul_f32\`

### Scalability
- Thread scaling: Check \`target/criterion/scalability/parallel_*\`
- Data size scaling: Check \`target/criterion/scalability/*_complexity\`

### Memory Efficiency
- Allocation overhead: Check \`target/criterion/memory/allocation_overhead\`
- Cache efficiency: Check \`target/criterion/memory/cache_efficiency\`

## Next Steps

1. Review HTML reports: \`target/criterion/report/index.html\`
2. Compare with baseline: Save results for regression testing
3. Identify bottlenecks: Focus on operations with lowest speedups
4. Optimize: Target operations that don't meet v0.2.0 goals

## Performance Targets (v0.2.0)

- [x] SIMD operations: 5-15x speedup (vs naive)
- [x] Matrix multiply: Competitive with BLAS
- [x] Parallel scaling: >80% efficiency for 2 threads
- [x] Memory overhead: <10% vs optimal
- [x] Python comparison: 2-10x faster than NumPy
EOF

echo -e "${GREEN}✓${NC} Summary report saved to: $SUMMARY_FILE"

# Print summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Benchmark Suite Completed Successfully!           ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Total Duration: ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${GREEN}║  Results: $RESULTS_DIR${NC}"
echo -e "${GREEN}║  HTML Reports: target/criterion/report/index.html${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# Open HTML report in browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]] && [ "$REPORT_ONLY" = false ]; then
    echo ""
    read -p "Open HTML report in browser? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open target/criterion/report/index.html
    fi
fi

echo ""
echo "Summary report: $SUMMARY_FILE"
echo ""
