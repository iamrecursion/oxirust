#!/bin/bash
# Script to run comprehensive stress tests for scirs2-graph

set -e

echo "==================================="
echo "SciRS2-Graph Stress Test Suite"
echo "==================================="
echo ""

# Check if running in release mode is requested
RELEASE_FLAG=""
if [ "$1" == "--release" ]; then
    RELEASE_FLAG="--release"
    echo "Running in RELEASE mode for accurate performance measurements"
else
    echo "Running in DEBUG mode (use --release for performance testing)"
fi

echo ""
echo "System Information:"
echo "-------------------"
uname -a
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "Cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Create results directory
RESULTS_DIR="stress_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run a test and save output
run_test() {
    local test_name=$1
    local output_file="$RESULTS_DIR/${test_name}.log"
    
    echo "Running: $test_name"
    echo "-----------------------------------"
    
    # Run the test and capture output
    if cargo test $RELEASE_FLAG "$test_name" -- --ignored --test-threads=1 --nocapture > "$output_file" 2>&1; then
        echo "✓ Completed successfully"
        # Extract key metrics
        grep -E "(Generation time|Total time|Operations time|nodes|edges)" "$output_file" | tail -10
    else
        echo "✗ Test failed (see $output_file for details)"
    fi
    echo ""
}

# Run individual stress tests
echo "1. Running Large Graph Generation Tests"
echo "======================================="
run_test "test_large_erdos_renyi_graph"
run_test "test_large_barabasi_albert_graph"
run_test "test_large_grid_graph"

echo ""
echo "2. Running Algorithm Performance Tests"
echo "======================================"
run_test "test_large_directed_graph_algorithms"
run_test "test_memory_efficient_operations"
run_test "test_algorithm_scaling"

echo ""
echo "3. Running Parallel Processing Tests"
echo "===================================="
if cargo test --features parallel --version > /dev/null 2>&1; then
    PARALLEL_FLAG="--features parallel"
    cargo test $RELEASE_FLAG $PARALLEL_FLAG test_parallel_algorithms_on_large_graphs -- --ignored --test-threads=1 --nocapture > "$RESULTS_DIR/parallel_algorithms.log" 2>&1
    echo "✓ Parallel tests completed"
else
    echo "⚠ Parallel feature not available"
fi

echo ""
echo "4. Running Extreme Scale Test (5M nodes)"
echo "========================================"
echo "⚠ This test requires significant memory (>2GB)"
read -p "Run extreme scale test? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_test "test_extreme_scale_graph"
else
    echo "Skipped extreme scale test"
fi

echo ""
echo "5. Generating Summary Report"
echo "============================"

# Create summary report
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
{
    echo "Stress Test Summary Report"
    echo "========================="
    echo "Date: $(date)"
    echo "Mode: ${RELEASE_FLAG:-DEBUG}"
    echo ""
    echo "Test Results:"
    echo "-------------"
    
    for log_file in "$RESULTS_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            test_name=$(basename "$log_file" .log)
            echo ""
            echo "### $test_name ###"
            # Extract generation times
            grep -E "Generation time: [0-9.]+" "$log_file" | tail -3 || true
            # Extract total times
            grep -E "Total time: [0-9.]+" "$log_file" | tail -3 || true
            # Extract any errors
            grep -i "error\|panic\|failed" "$log_file" | head -3 || true
        fi
    done
    
    echo ""
    echo "Memory Usage Estimates:"
    echo "----------------------"
    grep -h "Memory estimate" "$RESULTS_DIR"/*.log 2>/dev/null || echo "No memory estimates found"
    
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

echo ""
echo "6. Comparing with NetworkX (optional)"
echo "===================================="
if command -v python3 &> /dev/null && python3 -c "import networkx" 2>/dev/null; then
    read -p "Run NetworkX comparison? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running NetworkX benchmarks..."
        cd benches
        python3 large_graph_comparison.py > "../$RESULTS_DIR/networkx_comparison.log" 2>&1
        cd ..
        echo "NetworkX comparison saved to: $RESULTS_DIR/networkx_comparison.log"
    fi
else
    echo "⚠ NetworkX not available for comparison"
fi

echo ""
echo "==================================="
echo "Stress testing completed!"
echo "Results saved in: $RESULTS_DIR"
echo "==================================="

# Create a compressed archive of results
if command -v tar &> /dev/null; then
    tar -czf "${RESULTS_DIR}.tar.gz" "$RESULTS_DIR"
    echo ""
    echo "Compressed results: ${RESULTS_DIR}.tar.gz"
fi