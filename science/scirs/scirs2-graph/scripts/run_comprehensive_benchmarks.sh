#!/bin/bash

# Comprehensive benchmark runner for scirs2-graph
# This script runs all benchmark suites and generates performance reports

set -e

# Configuration
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
BASELINE_DIR="benchmark_baselines"
RUST_FLAGS="-C target-cpu=native -C opt-level=3"
MEMORY_LIMIT_GB=16

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  scirs2-graph Comprehensive Benchmarks${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_requirements() {
    print_step "Checking requirements..."
    
    # Check Rust version
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Check criterion is available
    if ! cargo bench --help | grep -q "criterion" 2>/dev/null; then
        print_warning "Criterion may not be available. Installing dependencies..."
        cargo build --benches
    fi
    
    # Check system resources
    AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$AVAILABLE_MEM" -lt "$MEMORY_LIMIT_GB" ]; then
        print_warning "Available memory ($AVAILABLE_MEM GB) is less than recommended ($MEMORY_LIMIT_GB GB)"
    fi
    
    # Check disk space (need ~1GB for results)
    AVAILABLE_SPACE=$(df . | awk 'NR==2{print $4}')
    if [ "$AVAILABLE_SPACE" -lt 1048576 ]; then # 1GB in KB
        print_warning "Available disk space is low. May not be sufficient for benchmark results."
    fi
    
    echo "✅ Requirements check passed"
}

setup_environment() {
    print_step "Setting up benchmark environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$BASELINE_DIR"
    
    # Set Rust flags for optimization
    export RUSTFLAGS="$RUST_FLAGS"
    
    # Collect system information
    cat > "$OUTPUT_DIR/system_info.txt" << EOF
System Information for Benchmark Run
====================================
Date: $(date)
OS: $(uname -a)
CPU: $(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
CPU Cores: $(nproc)
Memory: $(free -h | grep Mem | awk '{print $2}')
Rust Version: $(rustc --version)
Cargo Version: $(cargo --version)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "Not a git repository")
Git Status: $(git status --porcelain 2>/dev/null | wc -l) modified files
Rust Flags: $RUSTFLAGS
EOF

    echo "✅ Environment setup complete"
}

run_build_tests() {
    print_step "Running build and basic tests..."
    
    # Clean build first
    cargo clean
    
    # Build with optimizations
    if ! cargo build --release; then
        print_error "Release build failed"
        exit 1
    fi
    
    # Run tests to ensure correctness
    if ! cargo nextest run --release; then
        print_error "Tests failed"
        exit 1
    fi
    
    echo "✅ Build and tests passed"
}

run_benchmark_suite() {
    local suite_name=$1
    local timeout=${2:-3600} # Default 1 hour timeout
    
    print_step "Running benchmark suite: $suite_name"
    
    local start_time=$(date +%s)
    local output_file="$OUTPUT_DIR/${suite_name}_results.json"
    local log_file="$OUTPUT_DIR/${suite_name}_log.txt"
    
    # Run benchmark with timeout and capture both stdout and stderr
    if timeout "${timeout}s" cargo bench \
        --bench "$suite_name" \
        -- \
        --output-format json \
        > "$output_file" \
        2> "$log_file"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo "  ✅ Completed in ${duration}s"
        
        # Check for warnings in the log
        if grep -q "warning:" "$log_file"; then
            print_warning "Warnings detected in $suite_name"
        fi
        
        return 0
    else
        local exit_code=$?
        print_error "Failed (exit code: $exit_code)"
        
        if [ $exit_code -eq 124 ]; then
            echo "  ⏰ Timeout after ${timeout}s"
        fi
        
        return $exit_code
    fi
}

run_memory_profiling() {
    print_step "Running memory profiling benchmarks..."
    
    # Run memory-intensive benchmarks with monitoring
    local memory_log="$OUTPUT_DIR/memory_usage.log"
    
    # Start memory monitoring in background
    (
        while true; do
            echo "$(date +%s) $(cat /proc/meminfo | grep MemAvailable | awk '{print $2}')" >> "$memory_log"
            sleep 1
        done
    ) &
    local monitor_pid=$!
    
    # Run memory benchmarks
    run_benchmark_suite "memory_benchmarks" 1800
    
    # Stop memory monitoring
    kill $monitor_pid 2>/dev/null || true
    
    echo "✅ Memory profiling complete"
}

run_stress_tests() {
    print_step "Running large graph stress tests..."
    
    # Check available memory before stress tests
    local available_mem_kb=$(cat /proc/meminfo | grep MemAvailable | awk '{print $2}')
    local available_mem_gb=$((available_mem_kb / 1024 / 1024))
    
    if [ "$available_mem_gb" -lt 8 ]; then
        print_warning "Skipping stress tests due to insufficient memory ($available_mem_gb GB < 8 GB)"
        return 0
    fi
    
    # Run with extended timeout for stress tests
    run_benchmark_suite "large_graph_stress" 7200 # 2 hours
    
    echo "✅ Stress tests complete"
}

run_all_benchmarks() {
    print_step "Running all benchmark suites..."
    
    local suites=(
        "graph_benchmarks:1800"
        "memory_benchmarks:1800"
        "advanced_algorithms:3600"
        "performance_optimizations:2400"
        "large_graph_stress:7200"
    )
    
    local successful=0
    local failed=0
    
    for suite_spec in "${suites[@]}"; do
        local suite_name=$(echo "$suite_spec" | cut -d: -f1)
        local timeout=$(echo "$suite_spec" | cut -d: -f2)
        
        if run_benchmark_suite "$suite_name" "$timeout"; then
            ((successful++))
        else
            ((failed++))
        fi
    done
    
    echo
    echo "📊 Benchmark Summary:"
    echo "  ✅ Successful: $successful"
    echo "  ❌ Failed: $failed"
    echo "  📁 Results in: $OUTPUT_DIR"
}

generate_reports() {
    print_step "Generating comprehensive reports..."
    
    local report_file="$OUTPUT_DIR/comprehensive_report.md"
    
    cat > "$report_file" << EOF
# scirs2-graph Benchmark Report

**Generated:** $(date)
**Git Commit:** $(git rev-parse HEAD 2>/dev/null || echo "Unknown")

## System Information

$(cat "$OUTPUT_DIR/system_info.txt")

## Benchmark Results

EOF

    # Process each benchmark result
    for result_file in "$OUTPUT_DIR"/*_results.json; do
        if [ -f "$result_file" ]; then
            local suite_name=$(basename "$result_file" _results.json)
            echo "### $suite_name" >> "$report_file"
            echo >> "$report_file"
            
            # Basic result parsing (would be more sophisticated in practice)
            if grep -q "error" "$result_file"; then
                echo "❌ **Status:** Failed" >> "$report_file"
            else
                echo "✅ **Status:** Success" >> "$report_file"
                
                # Extract timing information if available
                local log_file="$OUTPUT_DIR/${suite_name}_log.txt"
                if [ -f "$log_file" ]; then
                    local duration=$(grep -o "test result: ok.*in [0-9.]*s" "$log_file" | tail -1 | grep -o "[0-9.]*s" || echo "Unknown")
                    echo "⏱️ **Duration:** $duration" >> "$report_file"
                fi
            fi
            echo >> "$report_file"
        fi
    done
    
    # Add performance comparison if baseline exists
    if [ -f "$BASELINE_DIR/baseline_summary.json" ]; then
        echo "## Performance Comparison" >> "$report_file"
        echo "Comparison with baseline from $BASELINE_DIR" >> "$report_file"
        echo >> "$report_file"
    fi
    
    # Generate HTML report
    if command -v pandoc &> /dev/null; then
        pandoc "$report_file" -o "$OUTPUT_DIR/benchmark_report.html"
        echo "📄 HTML report generated: $OUTPUT_DIR/benchmark_report.html"
    fi
    
    echo "📄 Markdown report generated: $report_file"
}

create_baseline() {
    print_step "Creating performance baseline..."
    
    # Copy current results as baseline
    cp -r "$OUTPUT_DIR" "$BASELINE_DIR/baseline_$(date +%Y%m%d_%H%M%S)"
    
    # Create baseline summary
    cat > "$BASELINE_DIR/baseline_summary.json" << EOF
{
    "date": "$(date -Iseconds)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo "unknown")",
    "results_dir": "$OUTPUT_DIR"
}
EOF

    echo "✅ Baseline created in $BASELINE_DIR"
}

cleanup() {
    print_step "Cleaning up..."
    
    # Compress results for storage
    if command -v tar &> /dev/null; then
        tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"
        echo "📦 Results archived to ${OUTPUT_DIR}.tar.gz"
    fi
    
    # Clean up temporary files
    cargo clean
    
    echo "✅ Cleanup complete"
}

main() {
    print_header
    
    # Parse command line arguments
    local create_baseline_flag=false
    local skip_stress_tests=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --create-baseline)
                create_baseline_flag=true
                shift
                ;;
            --skip-stress)
                skip_stress_tests=true
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --memory-limit)
                MEMORY_LIMIT_GB="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --create-baseline    Create performance baseline"
                echo "  --skip-stress        Skip stress tests"
                echo "  --output-dir DIR     Set output directory"
                echo "  --memory-limit GB    Set memory limit"
                echo "  -h, --help          Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run benchmark pipeline
    check_requirements
    setup_environment
    run_build_tests
    
    echo
    print_step "Starting benchmark execution..."
    run_all_benchmarks
    
    if [ "$skip_stress_tests" = false ]; then
        run_memory_profiling
        run_stress_tests
    fi
    
    generate_reports
    
    if [ "$create_baseline_flag" = true ]; then
        create_baseline
    fi
    
    cleanup
    
    echo
    print_step "Benchmark run completed successfully!"
    echo -e "${GREEN}📁 Results available in: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}📊 Open $OUTPUT_DIR/benchmark_report.html in your browser${NC}"
}

# Run main function with all arguments
main "$@"