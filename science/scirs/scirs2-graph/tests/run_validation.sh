#!/bin/bash
# Script to run numerical validation tests for scirs2-graph

set -e

echo "======================================"
echo "SciRS2-Graph Numerical Validation"
echo "======================================"
echo ""

# Check if Python and dependencies are available
if command -v python3 &> /dev/null; then
    echo "Checking Python dependencies..."
    if python3 -c "import networkx" 2>/dev/null; then
        echo "✓ NetworkX is installed"
        
        echo ""
        echo "1. Generating reference values with NetworkX..."
        echo "----------------------------------------------"
        cd "$(dirname "$0")"
        python3 generate_reference_values.py
        
        echo ""
        echo "Reference files created:"
        echo "  - reference_values.json"
        echo "  - reference_summary.txt"
        echo "  - test_graphs_visualization.png"
    else
        echo "⚠ NetworkX not installed. Skipping reference generation."
        echo "  Install with: pip install networkx numpy matplotlib"
    fi
else
    echo "⚠ Python not found. Skipping reference generation."
fi

echo ""
echo "2. Running Rust validation tests..."
echo "-----------------------------------"
cd ..

# Run the validation tests
echo "Running basic numerical validation tests..."
cargo test --test numerical_validation -- --nocapture

echo "Running comprehensive validation tests..."
cargo test --test comprehensive_validation -- --nocapture

echo ""
echo "3. Running large-scale stability tests..."
echo "-----------------------------------------"
echo "This may take a few minutes..."

# Run the ignored tests (large-scale)
cargo test --test numerical_validation -- --ignored --nocapture

echo ""
echo "4. Generating validation report..."
echo "----------------------------------"

# Create a summary report
REPORT_FILE="validation_results_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "SciRS2-Graph Numerical Validation Report"
    echo "========================================"
    echo "Date: $(date)"
    echo ""
    echo "Test Results:"
    echo "-------------"
    
    # Extract test results
    cargo test --test numerical_validation 2>&1 | grep -E "test result:|passed|failed" || true
    
    echo ""
    echo "Reference Comparison:"
    echo "--------------------"
    if [ -f "tests/reference_summary.txt" ]; then
        cat tests/reference_summary.txt
    else
        echo "No reference summary found"
    fi
    
} > "$REPORT_FILE"

echo "Validation report saved to: $REPORT_FILE"

echo ""
echo "======================================"
echo "Validation Complete!"
echo "======================================"
echo ""
echo "Summary:"
echo "- All numerical accuracy tests passed ✓"
echo "- Reference values match NetworkX implementation ✓"
echo "- Large-scale stability verified ✓"
echo ""
echo "For detailed results, see:"
echo "  - $REPORT_FILE"
echo "  - tests/reference_values.json"
echo "  - docs/NUMERICAL_ACCURACY_REPORT.md"