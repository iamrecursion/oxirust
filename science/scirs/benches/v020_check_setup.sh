#!/bin/bash
#
# SciRS2 v0.2.0 Benchmark Setup Checker
#
# This script verifies that all dependencies and requirements
# are met for running the benchmark suite.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     SciRS2 v0.2.0 Benchmark Setup Checker                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

WARNINGS=0
ERRORS=0

# Check Rust
echo -n "Checking Rust... "
if command -v rustc &> /dev/null; then
    VERSION=$(rustc --version)
    echo -e "${GREEN}✓${NC} $VERSION"
else
    echo -e "${RED}✗${NC} Rust not found"
    echo "  Install from: https://rustup.rs/"
    ERRORS=$((ERRORS + 1))
fi

# Check Cargo
echo -n "Checking Cargo... "
if command -v cargo &> /dev/null; then
    VERSION=$(cargo --version)
    echo -e "${GREEN}✓${NC} $VERSION"
else
    echo -e "${RED}✗${NC} Cargo not found"
    ERRORS=$((ERRORS + 1))
fi

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $VERSION"

    # Check Python packages
    echo "Checking Python packages..."

    PACKAGES=("numpy" "scipy" "pandas" "matplotlib")
    for pkg in "${PACKAGES[@]}"; do
        echo -n "  $pkg... "
        if python3 -c "import $pkg" 2>/dev/null; then
            VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            echo -e "${GREEN}✓${NC} $VERSION"
        else
            echo -e "${YELLOW}⊘${NC} Not installed"
            WARNINGS=$((WARNINGS + 1))
        fi
    done

    if [ $WARNINGS -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Warning:${NC} Some Python packages are missing"
        echo "Install with: pip install -r benches/requirements.txt"
    fi
else
    echo -e "${YELLOW}⊘${NC} Python not found (optional for Python comparison)"
    echo "  Python benchmarks will be skipped"
fi

# Check criterion
echo -n "Checking criterion... "
if grep -q "criterion" Cargo.toml; then
    echo -e "${GREEN}✓${NC} Found in workspace"
else
    echo -e "${RED}✗${NC} Not found in Cargo.toml"
    ERRORS=$((ERRORS + 1))
fi

# Check GPU backends
echo "Checking GPU backends..."

echo -n "  CUDA... "
if command -v nvcc &> /dev/null; then
    VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo -e "${GREEN}✓${NC} $VERSION"
else
    echo -e "${YELLOW}⊘${NC} Not available (optional)"
fi

echo -n "  Metal... "
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}✓${NC} Available on macOS"
else
    echo -e "${YELLOW}⊘${NC} macOS only"
fi

# Check disk space
echo -n "Checking disk space... "
if command -v df &> /dev/null; then
    AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
    echo -e "${GREEN}✓${NC} $AVAILABLE available"

    # Convert to GB for comparison (simplified)
    AVAILABLE_NUM=$(df -h . | tail -1 | awk '{print $4}' | sed 's/[^0-9.]//g')
    if (( $(echo "$AVAILABLE_NUM < 5" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${YELLOW}Warning:${NC} Low disk space (recommend >5GB for benchmarks)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⊘${NC} Cannot check"
fi

# Check memory
echo -n "Checking memory... "
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}')
    echo -e "${GREEN}✓${NC} $TOTAL_MEM total"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    TOTAL_MEM=$(free -h | grep Mem | awk '{print $2}')
    echo -e "${GREEN}✓${NC} $TOTAL_MEM total"
else
    echo -e "${YELLOW}⊘${NC} Cannot check"
fi

# Check if in correct directory
echo -n "Checking directory... "
if [ -f "Cargo.toml" ] && [ -d "benches" ]; then
    echo -e "${GREEN}✓${NC} In SciRS2 workspace root"
else
    echo -e "${RED}✗${NC} Not in SciRS2 workspace root"
    echo "  Please run from the workspace root directory"
    ERRORS=$((ERRORS + 1))
fi

# Check benchmark files
echo "Checking benchmark files..."
BENCH_FILES=(
    "v020_comprehensive_suite.rs"
    "v020_simd_comparison.rs"
    "v020_gpu_comparison.rs"
    "v020_memory_profiling.rs"
    "v020_scalability.rs"
    "v020_python_comparison.rs"
    "v020_python_comparison.py"
    "v020_run_all_benchmarks.sh"
)

for file in "${BENCH_FILES[@]}"; do
    echo -n "  $file... "
    if [ -f "benches/$file" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC} Missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Ready to run benchmarks:"
    echo "  cd benches"
    echo "  ./v020_run_all_benchmarks.sh"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can run benchmarks, but some features may be unavailable:"
    echo "  cd benches"
    echo "  ./v020_run_all_benchmarks.sh --skip-python"
else
    echo -e "${RED}✗ Setup incomplete: $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before running benchmarks."
    exit 1
fi

echo ""
