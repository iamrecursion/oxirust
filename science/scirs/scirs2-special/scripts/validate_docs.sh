#!/bin/bash
# Documentation validation script for scirs2-special
# Ensures all public APIs have comprehensive documentation

echo "📚 Documentation Validation for scirs2-special"
echo "==============================================="

# Check for missing documentation
echo "🔍 Checking for missing documentation..."

# Run cargo doc with all features to generate documentation
echo "Generating documentation with all features..."
cargo doc --all-features --no-deps --quiet

# Check for warnings during doc generation
echo "🔧 Checking for documentation warnings..."
cargo doc --all-features --no-deps 2>&1 | grep -i warning | head -10

# Validate that key modules have examples
echo "📝 Validating documentation examples..."

# Check gamma function documentation
if grep -q "# Examples" src/gamma.rs; then
    echo "✅ Gamma functions: Examples found"
else
    echo "❌ Gamma functions: Missing examples"
fi

# Check bessel function documentation  
if grep -q "# Examples" src/bessel/mod.rs; then
    echo "✅ Bessel functions: Examples found"
else
    echo "❌ Bessel functions: Missing examples"
fi

# Check error function documentation
if grep -q "# Examples" src/erf.rs; then
    echo "✅ Error functions: Examples found"
else
    echo "❌ Error functions: Missing examples"
fi

# Check for mathematical references
echo "🔬 Validating mathematical references..."

# Count references in key modules
gamma_refs=$(grep -c "# References" src/gamma.rs)
bessel_refs=$(grep -c "# References" src/bessel/mod.rs)

echo "📖 Reference counts:"
echo "  - Gamma functions: $gamma_refs references sections"
echo "  - Bessel functions: $bessel_refs references sections"

# Check for comprehensive examples
echo "🧪 Checking example coverage..."

example_count=$(find examples -name "*.rs" | wc -l)
echo "  - Total examples: $example_count"

# List critical examples
echo "📋 Critical examples status:"
if [ -f "examples/bessel_interactive_tutorial.rs" ]; then
    echo "  ✅ Bessel interactive tutorial"
else
    echo "  ❌ Missing Bessel interactive tutorial"
fi

if [ -f "examples/comprehensive_performance_benchmark.rs" ]; then
    echo "  ✅ Performance benchmark example"
else
    echo "  ❌ Missing performance benchmark example"
fi

if [ -f "examples/validate_benchmarking.rs" ]; then
    echo "  ✅ Benchmarking validation example"
else
    echo "  ❌ Missing benchmarking validation example"
fi

echo "🎯 Documentation validation complete!"
echo "For full documentation, run: cargo doc --open --all-features"