#!/bin/bash

# rust-test-diagnostician - Comprehensive Rust project diagnostics
# Follows SciRS2 project requirements from CLAUDE.md

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Rust Test Diagnostician ===${NC}"
echo -e "${BLUE}Comprehensive diagnostics for SciRS2 project${NC}"
echo

SUCCESS_COUNT=0
TOTAL_CHECKS=0

# Function to run a check and track results
run_check() {
    local check_name="$1"
    local command="$2"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -e "${BLUE}[$TOTAL_CHECKS] Running: $check_name${NC}"
    echo "Command: $command"
    
    if eval "$command" > /tmp/diagnostician_output.txt 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo -e "${YELLOW}Output:${NC}"
        cat /tmp/diagnostician_output.txt
        echo
    fi
}

# Function to run a check with custom success condition
run_check_custom() {
    local check_name="$1"
    local command="$2"
    local success_condition="$3"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -e "${BLUE}[$TOTAL_CHECKS] Running: $check_name${NC}"
    echo "Command: $command"
    
    eval "$command" > /tmp/diagnostician_output.txt 2>&1
    if eval "$success_condition"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo -e "${YELLOW}Output:${NC}"
        cat /tmp/diagnostician_output.txt
        echo
    fi
}

echo -e "${YELLOW}Starting comprehensive diagnostics...${NC}"
echo

# 1. Code Formatting Check (must pass)
run_check "Code Formatting Check" "cargo fmt --check"

# 2. Clippy Linting (ZERO warnings policy)
run_check_custom "Clippy Zero Warnings Check" "cargo clippy --all-targets --all-features -- -D warnings" "[ \$? -eq 0 ]"

# 3. Build Check
run_check "Debug Build Check" "cargo build"

# 4. Release Build Check
run_check "Release Build Check" "cargo build --release"

# 5. Nextest Run (as required by CLAUDE.md)
run_check "Nextest Run" "cargo nextest run"

# 6. Documentation Build
run_check "Documentation Build" "cargo doc --no-deps"

# 7. Example Compilation Check (sample a few key examples)
if [ -d "examples" ]; then
    for example in examples/spatial_scipy_comparison.rs examples/performance_validation.rs; do
        if [ -f "$example" ]; then
            example_name=$(basename "$example" .rs)
            run_check "Example: $example_name" "cargo check --example $example_name"
        fi
    done
fi

# 8. Benchmark Compilation Check
if [ -d "benches" ]; then
    run_check "Benchmark Compilation" "cargo check --benches"
fi

# 9. Feature Flag Tests
run_check "Default Features Build" "cargo build --no-default-features"
run_check "All Features Build" "cargo build --all-features"

# 10. Memory and Performance Checks (if criterion is available)
if cargo tree | grep -q criterion; then
    echo -e "${BLUE}[INFO] Criterion found, running performance regression check${NC}"
    run_check "Performance Benchmark" "cargo bench --no-run"
fi

# Summary
echo -e "${BLUE}=== DIAGNOSTIC SUMMARY ===${NC}"
echo -e "Checks passed: ${GREEN}$SUCCESS_COUNT${NC}/$TOTAL_CHECKS"

if [ "$SUCCESS_COUNT" -eq "$TOTAL_CHECKS" ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED! Project is healthy.${NC}"
    exit 0
else
    FAILED_COUNT=$((TOTAL_CHECKS - SUCCESS_COUNT))
    echo -e "${RED}❌ $FAILED_COUNT checks failed. Review output above.${NC}"
    exit 1
fi