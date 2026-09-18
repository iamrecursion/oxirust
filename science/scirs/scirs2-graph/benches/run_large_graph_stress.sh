#!/bin/bash

# Large Graph Stress Testing Script for scirs2-graph
# This script runs comprehensive stress tests on graphs with >1M nodes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Large Graph Stress Testing Suite ===${NC}"
echo "Testing scirs2-graph with graphs up to 5M nodes"
echo ""

# Check system resources
echo -e "${YELLOW}System Information:${NC}"
echo "CPU Cores: $(nproc)"
echo "Total Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Available Memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo ""

# Function to monitor memory during tests
monitor_memory() {
    local test_name=$1
    local pid=$2
    local max_mem=0
    
    while kill -0 $pid 2>/dev/null; do
        if [ -f /proc/$pid/status ]; then
            local curr_mem=$(grep VmRSS /proc/$pid/status | awk '{print $2}')
            if [ -n "$curr_mem" ] && [ "$curr_mem" -gt "$max_mem" ]; then
                max_mem=$curr_mem
            fi
        fi
        sleep 0.5
    done
    
    echo "Peak memory for $test_name: $((max_mem / 1024)) MB"
}

# Create results directory
RESULTS_DIR="stress_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${YELLOW}Results will be saved to: $RESULTS_DIR${NC}"
echo ""

# Test 1: Basic stress tests with criterion benchmarks
echo -e "${GREEN}Test 1: Running criterion benchmarks...${NC}"
cargo bench --bench large_graph_stress -- --output-format bencher | tee "$RESULTS_DIR/criterion_benchmarks.txt"

# Test 2: Million node stress test
echo -e "${GREEN}Test 2: Running million node stress test...${NC}"
RUST_BACKTRACE=1 cargo test --release --test stress_tests -- --ignored test_extreme_scale_graph --nocapture 2>&1 | tee "$RESULTS_DIR/million_node_test.txt" &
TEST_PID=$!
monitor_memory "million_node_test" $TEST_PID &
MONITOR_PID=$!
wait $TEST_PID
kill $MONITOR_PID 2>/dev/null || true

# Test 3: Parallel algorithm comparison
echo -e "${GREEN}Test 3: Testing parallel algorithm performance...${NC}"
cargo test --release --features parallel --test stress_tests -- --ignored test_parallel_algorithms_large_graph --nocapture 2>&1 | tee "$RESULTS_DIR/parallel_test.txt"

# Test 4: Algorithm scaling analysis
echo -e "${GREEN}Test 4: Running algorithm scaling analysis...${NC}"
cargo test --release --test stress_tests -- --ignored test_algorithm_scaling --nocapture 2>&1 | tee "$RESULTS_DIR/scaling_analysis.txt"

# Test 5: Memory efficiency test
echo -e "${GREEN}Test 5: Testing memory efficiency...${NC}"
cargo test --release --test stress_tests -- --ignored test_memory_efficient_operations --nocapture 2>&1 | tee "$RESULTS_DIR/memory_efficiency.txt"

# Test 6: Custom large graph stress test
echo -e "${GREEN}Test 6: Running custom stress test suite...${NC}"
cargo test --release --bench large_graph_stress -- --ignored stress_test_million_nodes --nocapture 2>&1 | tee "$RESULTS_DIR/custom_stress_test.txt"

# Test 7: Five million node test (optional, requires lots of memory)
echo -e "${YELLOW}Test 7: Five million node test (requires 32GB+ RAM)${NC}"
read -p "Run 5M node test? This requires significant memory. (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cargo test --release --bench large_graph_stress -- --ignored stress_test_five_million_nodes --nocapture 2>&1 | tee "$RESULTS_DIR/five_million_test.txt"
fi

# Generate summary report
echo -e "${GREEN}Generating summary report...${NC}"

cat > "$RESULTS_DIR/summary_report.md" << EOF
# Large Graph Stress Test Summary

**Date**: $(date)
**System**: $(uname -a)
**Rust Version**: $(rustc --version)

## Test Results

### 1. Criterion Benchmarks
See \`criterion_benchmarks.txt\` for detailed results.

### 2. Million Node Test
$(grep -E "(Generation time|Total time|Memory)" "$RESULTS_DIR/million_node_test.txt" | head -10 || echo "Test results not found")

### 3. Parallel Performance
$(grep -E "(Speedup|Parallel)" "$RESULTS_DIR/parallel_test.txt" | head -10 || echo "Test results not found")

### 4. Algorithm Scaling
$(grep -E "(BFS time|PageRank.*time|Connected components time)" "$RESULTS_DIR/scaling_analysis.txt" | head -15 || echo "Test results not found")

### 5. Memory Efficiency
$(grep -E "(Memory|Degree stats|clustering)" "$RESULTS_DIR/memory_efficiency.txt" | head -10 || echo "Test results not found")

## Performance Highlights

- Largest graph successfully tested: $(grep -oE "[0-9]+ nodes" "$RESULTS_DIR"/*.txt | grep -oE "[0-9]+" | sort -n | tail -1 || echo "Unknown") nodes
- Peak memory usage: $(grep -oE "Peak memory.*[0-9]+ MB" "$RESULTS_DIR"/*.txt | grep -oE "[0-9]+" | sort -n | tail -1 || echo "Unknown") MB

## Notes

- All tests were run with release optimizations
- Parallel features were enabled where applicable
- Memory measurements are approximate and system-dependent
EOF

echo -e "${GREEN}Summary report saved to: $RESULTS_DIR/summary_report.md${NC}"

# Analyze performance vs NetworkX (if comparison data exists)
if [ -f "networkx_benchmark_results.json" ]; then
    echo -e "${GREEN}Comparing with NetworkX results...${NC}"
    python3 - << 'PYTHON_SCRIPT' > "$RESULTS_DIR/performance_comparison.txt"
import json
import sys

try:
    with open('networkx_benchmark_results.json', 'r') as f:
        nx_results = json.load(f)
    
    print("Performance Comparison: scirs2-graph vs NetworkX")
    print("=" * 50)
    
    # Extract and compare common benchmarks
    for algo, times in nx_results.items():
        print(f"\n{algo}:")
        for size, nx_time in times.items():
            print(f"  Size {size}: NetworkX = {nx_time:.3f}s")
            # Note: Would need to parse Rust results for actual comparison
            
except Exception as e:
    print(f"Could not perform comparison: {e}")
PYTHON_SCRIPT
fi

echo ""
echo -e "${GREEN}=== Stress Testing Complete ===${NC}"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view the summary: cat $RESULTS_DIR/summary_report.md"

# Check for any failures
if grep -q "FAILED\|Error\|panic" "$RESULTS_DIR"/*.txt 2>/dev/null; then
    echo -e "${RED}Warning: Some tests may have failed. Check the detailed logs.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests completed successfully!${NC}"
fi