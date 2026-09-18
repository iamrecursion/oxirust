#!/bin/bash
#
# Performance Regression Testing Script for CI/CD
#
# This script runs performance benchmarks and checks for regressions
# compared to baseline measurements stored in the repository.
#
# Usage:
#   ./scripts/performance_regression_ci.sh [OPTIONS]
#
# Options:
#   --baseline-file FILE    Path to baseline benchmark results (default: performance_baseline.json)
#   --output-file FILE      Path to output new benchmark results (default: current_performance.json)  
#   --threshold PERCENT     Regression threshold percentage (default: 10)
#   --features FEATURES     Comma-separated list of features to enable (default: simd,parallel)
#   --verbose               Enable verbose output
#   --help                  Show this help message

set -euo pipefail

# Default configuration
BASELINE_FILE="performance_baseline.json"
OUTPUT_FILE="current_performance.json"
REGRESSION_THRESHOLD=10
FEATURES="simd,parallel"
VERBOSE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Performance Regression Testing Script for CI/CD

This script runs performance benchmarks and checks for regressions
compared to baseline measurements stored in the repository.

Usage:
    $0 [OPTIONS]

Options:
    --baseline-file FILE    Path to baseline benchmark results (default: $BASELINE_FILE)
    --output-file FILE      Path to output new benchmark results (default: $OUTPUT_FILE)
    --threshold PERCENT     Regression threshold percentage (default: $REGRESSION_THRESHOLD)
    --features FEATURES     Comma-separated list of features to enable (default: $FEATURES)
    --verbose               Enable verbose output
    --help                  Show this help message

Examples:
    # Run with default settings
    $0

    # Run with custom threshold and features
    $0 --threshold 5 --features "simd,parallel,gpu"

    # Run in verbose mode
    $0 --verbose

    # Compare against specific baseline
    $0 --baseline-file benchmarks/baseline_v0.1.0.json

Environment Variables:
    CARGO_FEATURES          Additional cargo features to enable
    BENCHMARK_TIMEOUT       Timeout for benchmark execution (default: 1800s)
    CI                      Set to 'true' to enable CI-specific optimizations
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --baseline-file)
                BASELINE_FILE="$2"
                shift 2
                ;;
            --output-file)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --threshold)
                REGRESSION_THRESHOLD="$2"
                shift 2
                ;;
            --features)
                FEATURES="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check for required tools
    local required_tools=("cargo" "jq" "bc")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
        verbose_log "Found $tool: $(command -v "$tool")"
    done
    
    # Check if we're in the right directory
    if [ ! -f "$ROOT_DIR/Cargo.toml" ]; then
        log_error "Not in a Rust project directory. Please run from the project root."
        exit 1
    fi
    
    # Check if criterion is available
    if ! grep -q "criterion" "$ROOT_DIR/Cargo.toml"; then
        log_error "Criterion benchmarking framework not found in Cargo.toml"
        exit 1
    fi
    
    log_success "All dependencies verified"
}

# Set up environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Set environment variables for consistent benchmarking
    export RUST_BACKTRACE=1
    export CARGO_TARGET_DIR="${ROOT_DIR}/target/benchmarks"
    export BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-1800}"
    
    # CI-specific optimizations
    if [ "${CI:-false}" = "true" ]; then
        verbose_log "CI environment detected, applying optimizations"
        export RUSTFLAGS="-C target-cpu=native"
        export CARGO_BUILD_JOBS="$(nproc)"
    fi
    
    # Create output directory
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    
    verbose_log "Environment variables set:"
    verbose_log "  CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
    verbose_log "  BENCHMARK_TIMEOUT=$BENCHMARK_TIMEOUT"
    verbose_log "  Features: $FEATURES"
}

# Build the project with specified features
build_project() {
    log_info "Building project with features: $FEATURES"
    
    local feature_flags=""
    if [ -n "$FEATURES" ]; then
        feature_flags="--features $FEATURES"
    fi
    
    # Add any additional features from environment
    if [ -n "${CARGO_FEATURES:-}" ]; then
        feature_flags="$feature_flags,$CARGO_FEATURES"
    fi
    
    verbose_log "Running: cargo build --release --benches $feature_flags"
    
    if ! timeout "$BENCHMARK_TIMEOUT" cargo build --release --benches $feature_flags; then
        log_error "Failed to build project"
        exit 1
    fi
    
    log_success "Project built successfully"
}

# Run performance benchmarks
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    local feature_flags=""
    if [ -n "$FEATURES" ]; then
        feature_flags="--features $FEATURES"
    fi
    
    # Add any additional features from environment
    if [ -n "${CARGO_FEATURES:-}" ]; then
        feature_flags="$feature_flags,$CARGO_FEATURES"
    fi
    
    local bench_cmd="cargo bench --bench performance_regression $feature_flags -- --output-format json"
    verbose_log "Running: $bench_cmd"
    
    # Run benchmarks and capture output
    if ! timeout "$BENCHMARK_TIMEOUT" $bench_cmd > "$OUTPUT_FILE.raw" 2>&1; then
        log_error "Benchmark execution failed or timed out"
        if [ -f "$OUTPUT_FILE.raw" ]; then
            log_error "Partial output:"
            tail -20 "$OUTPUT_FILE.raw"
        fi
        exit 1
    fi
    
    # Extract JSON results from criterion output
    if ! grep "^{" "$OUTPUT_FILE.raw" > "$OUTPUT_FILE"; then
        log_error "Failed to extract JSON results from benchmark output"
        exit 1
    fi
    
    log_success "Benchmarks completed successfully"
    verbose_log "Results saved to: $OUTPUT_FILE"
}

# Compare benchmark results
compare_results() {
    log_info "Comparing benchmark results against baseline..."
    
    if [ ! -f "$BASELINE_FILE" ]; then
        log_warn "Baseline file '$BASELINE_FILE' not found. Skipping regression check."
        log_info "Current benchmark results saved to '$OUTPUT_FILE'"
        return 0
    fi
    
    # Create comparison report
    local comparison_report="comparison_report.txt"
    local regressions_found=false
    
    echo "Performance Regression Report" > "$comparison_report"
    echo "=============================" >> "$comparison_report"
    echo "Baseline: $BASELINE_FILE" >> "$comparison_report"
    echo "Current:  $OUTPUT_FILE" >> "$comparison_report"
    echo "Threshold: ${REGRESSION_THRESHOLD}%" >> "$comparison_report"
    echo "" >> "$comparison_report"
    
    # Compare each benchmark
    local benchmark_names
    benchmark_names=$(jq -r 'keys[]' "$OUTPUT_FILE" 2>/dev/null || echo "")
    
    if [ -z "$benchmark_names" ]; then
        log_error "Failed to parse benchmark results"
        exit 1
    fi
    
    for benchmark in $benchmark_names; do
        verbose_log "Comparing benchmark: $benchmark"
        
        # Get baseline and current times (in nanoseconds)
        local baseline_time
        local current_time
        
        baseline_time=$(jq -r ".[\"$benchmark\"].mean.estimate" "$BASELINE_FILE" 2>/dev/null || echo "null")
        current_time=$(jq -r ".[\"$benchmark\"].mean.estimate" "$OUTPUT_FILE" 2>/dev/null || echo "null")
        
        if [ "$baseline_time" = "null" ] || [ "$current_time" = "null" ]; then
            echo "SKIP $benchmark: Missing data" >> "$comparison_report"
            continue
        fi
        
        # Calculate percentage change
        local percent_change
        percent_change=$(echo "scale=2; ($current_time - $baseline_time) / $baseline_time * 100" | bc)
        
        # Check if this is a regression
        local is_regression
        is_regression=$(echo "$percent_change > $REGRESSION_THRESHOLD" | bc)
        
        if [ "$is_regression" -eq 1 ]; then
            echo "REGRESSION $benchmark: ${percent_change}% slower (${baseline_time}ns -> ${current_time}ns)" >> "$comparison_report"
            log_warn "Regression detected in $benchmark: ${percent_change}% slower"
            regressions_found=true
        elif [ "$(echo "$percent_change < -5" | bc)" -eq 1 ]; then
            echo "IMPROVEMENT $benchmark: ${percent_change}% faster (${baseline_time}ns -> ${current_time}ns)" >> "$comparison_report"
        else
            echo "OK $benchmark: ${percent_change}% change (${baseline_time}ns -> ${current_time}ns)" >> "$comparison_report"
        fi
    done
    
    # Display report
    cat "$comparison_report"
    
    if [ "$regressions_found" = true ]; then
        log_error "Performance regressions detected! See report above."
        exit 1
    else
        log_success "No performance regressions detected"
    fi
}

# Cleanup function
cleanup() {
    verbose_log "Cleaning up temporary files..."
    rm -f "$OUTPUT_FILE.raw" "comparison_report.txt"
}

# Main execution
main() {
    trap cleanup EXIT
    
    log_info "Starting performance regression testing..."
    
    parse_args "$@"
    check_dependencies
    setup_environment
    build_project
    run_benchmarks
    compare_results
    
    log_success "Performance regression testing completed successfully"
}

# Run main function with all arguments
main "$@"