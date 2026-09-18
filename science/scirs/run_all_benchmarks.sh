#!/bin/bash

# Unified benchmark runner for all SciRS2 modules
# This script runs benchmarks across the entire SciRS2 ecosystem

set -e

echo "=== SciRS2 Comprehensive Ecosystem Benchmarking ==="
echo "Starting at: $(date)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create directories
mkdir -p target/ecosystem-benchmarks
mkdir -p target/ecosystem-plots

echo -e "${BLUE}Setting up ecosystem benchmark environment...${NC}"

# List of all modules with benchmarks
MODULES_WITH_BENCHMARKS=(
    "scirs2:linalg_benchmarks:Linear Algebra Performance"
    "scirs2:memory_efficiency:Memory Efficiency" 
    "scirs2:numerical_stability:Numerical Stability"
    "scirs2:scipy_comparison:SciPy Comparison"
    "scirs2-core:core_benchmarks:Core Operations"
    "scirs2-linalg:linalg_benchmarks:Linear Algebra Module"
    "scirs2-integrate:integration_benchmarks:Integration Methods"
    "scirs2-interpolate:interpolation_benchmarks:Interpolation Methods"
    "scirs2-fft:fft_benchmarks:FFT Operations"
    "scirs2-special:special_benchmarks:Special Functions"
    "scirs2-signal:signal_benchmarks:Signal Processing"
    "scirs2-spatial:spatial_benchmarks:Spatial Operations"
    "scirs2-cluster:cluster_benchmarks:Clustering Algorithms"
    "scirs2-ndimage:ndimage_benchmarks:N-dimensional Image Processing"
    "scirs2-metrics:metrics_benchmarks:Performance Metrics"
    "scirs2-sparse:sparse_benchmarks:Sparse Matrix Operations"
    "scirs2-neural:neural_benchmarks:Neural Network Operations"
)

# Function to run a module benchmark
run_module_benchmark() {
    local module_info=$1
    IFS=':' read -r module bench_name description <<< "$module_info"
    
    echo -e "${CYAN}Running $description ($module)...${NC}"
    
    local output_dir="target/ecosystem-benchmarks/${module}"
    mkdir -p "$output_dir"
    
    # Change to module directory and run benchmark
    if [ -d "$module" ]; then
        cd "$module"
        
        if [ -f "Cargo.toml" ] && [ -d "benches" ]; then
            if cargo bench --bench "$bench_name" -- --output-format html > "../$output_dir/benchmark_output.log" 2>&1; then
                echo -e "${GREEN}✓ $description completed successfully${NC}"
                
                # Move HTML reports to organized location
                if [ -d "target/criterion" ]; then
                    mv target/criterion "../$output_dir/criterion_reports"
                fi
                
                # Extract key metrics to JSON
                if [ -f "../$output_dir/criterion_reports" ]; then
                    echo "{\"module\": \"$module\", \"benchmark\": \"$bench_name\", \"status\": \"success\", \"timestamp\": \"$(date -Iseconds)\"}" > "../$output_dir/metrics.json"
                fi
            else
                echo -e "${RED}✗ $description failed${NC}"
                echo "Check target/ecosystem-benchmarks/${module}/benchmark_output.log for details"
                echo "{\"module\": \"$module\", \"benchmark\": \"$bench_name\", \"status\": \"failed\", \"timestamp\": \"$(date -Iseconds)\"}" > "../$output_dir/metrics.json"
            fi
        else
            echo -e "${YELLOW}⚠ No benchmarks found for $module${NC}"
            echo "{\"module\": \"$module\", \"benchmark\": \"$bench_name\", \"status\": \"not_found\", \"timestamp\": \"$(date -Iseconds)\"}" > "../$output_dir/metrics.json"
        fi
        
        cd ..
    else
        echo -e "${YELLOW}⚠ Module directory $module not found${NC}"
    fi
    
    sleep 1  # Brief pause between benchmarks
}

# Function to check system requirements
check_system_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"
    
    # Check if we're in the right directory
    if [ ! -f "Cargo.toml" ] || [ ! -d "scirs2" ]; then
        echo -e "${RED}Error: Must be run from SciRS2 workspace root${NC}"
        exit 1
    fi
    
    # Check Rust toolchain
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}Error: Cargo not found. Please install Rust.${NC}"
        exit 1
    fi
    
    # Check if we can compile
    if ! cargo check &> /dev/null; then
        echo -e "${RED}Error: Compilation failed. Please fix build errors first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ System requirements check passed${NC}"
}

# Function to generate ecosystem report
generate_ecosystem_report() {
    echo -e "${BLUE}Generating ecosystem benchmark report...${NC}"
    
    local report_file="target/ecosystem-benchmarks/ECOSYSTEM_REPORT.md"
    
    cat > "$report_file" << 'EOF'
# SciRS2 Ecosystem Performance Report

This report contains comprehensive performance benchmarks for the entire SciRS2 ecosystem.

## Benchmarked Modules

### Core Infrastructure
- **scirs2-core**: Core operations and utilities
- **scirs2-linalg**: Linear algebra operations

### Mathematical Functions
- **scirs2-integrate**: Integration methods
- **scirs2-interpolate**: Interpolation algorithms  
- **scirs2-fft**: Fast Fourier Transform operations
- **scirs2-special**: Special mathematical functions

### Data Processing
- **scirs2-signal**: Signal processing algorithms
- **scirs2-spatial**: Spatial data structures and operations
- **scirs2-sparse**: Sparse matrix operations
- **scirs2-ndimage**: N-dimensional image processing

### Machine Learning
- **scirs2-neural**: Neural network building blocks
- **scirs2-cluster**: Clustering algorithms
- **scirs2-metrics**: Performance metrics and evaluation

## Performance Categories

### 1. Core Operations
- Basic array operations
- Memory management
- Parallel processing efficiency

### 2. Mathematical Algorithms
- Numerical accuracy
- Algorithmic complexity
- Stability under various conditions

### 3. Memory Efficiency
- Memory usage patterns
- Zero-copy operations
- Large dataset handling

### 4. Comparative Performance
- Performance vs SciPy/NumPy
- Cross-platform consistency
- Scalability analysis

## Report Structure

Each module has its own subdirectory containing:
- `criterion_reports/`: Interactive HTML reports from Criterion
- `benchmark_output.log`: Raw benchmark execution logs
- `metrics.json`: Summary metrics in JSON format

## Reading the Results

### HTML Reports
Open `criterion_reports/report/index.html` in each module directory for detailed interactive analysis.

### Performance Metrics
- **Throughput**: Operations per second
- **Latency**: Time per operation  
- **Memory Usage**: Peak memory consumption
- **Scalability**: Performance across different input sizes

### Comparative Analysis
Cross-module performance comparisons and ecosystem-wide optimization opportunities.

EOF

    # Aggregate all metrics
    echo "## Module Status Summary" >> "$report_file"
    echo "" >> "$report_file"
    
    for module_dir in target/ecosystem-benchmarks/*/; do
        if [ -f "$module_dir/metrics.json" ]; then
            local module=$(basename "$module_dir")
            local status=$(grep -o '"status": "[^"]*"' "$module_dir/metrics.json" | cut -d'"' -f4)
            local timestamp=$(grep -o '"timestamp": "[^"]*"' "$module_dir/metrics.json" | cut -d'"' -f4)
            
            case $status in
                "success")
                    echo "- ✅ **$module**: Benchmarks completed successfully ($timestamp)" >> "$report_file"
                    ;;
                "failed") 
                    echo "- ❌ **$module**: Benchmarks failed ($timestamp)" >> "$report_file"
                    ;;
                "not_found")
                    echo "- ⚠️ **$module**: No benchmarks found ($timestamp)" >> "$report_file"
                    ;;
            esac
        fi
    done
    
    echo -e "${GREEN}✓ Ecosystem report generated: $report_file${NC}"
}

# Function to create ecosystem visualization
create_ecosystem_visualization() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Python not available - skipping visualizations${NC}"
        return
    fi
    
    cat > target/create_ecosystem_plots.py << 'EOF'
#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

def collect_all_metrics():
    """Collect metrics from all module benchmark results."""
    metrics = []
    
    # Find all metrics.json files
    for metrics_file in glob.glob('target/ecosystem-benchmarks/*/metrics.json'):
        try:
            with open(metrics_file, 'r') as f:
                metric = json.load(f)
                metrics.append(metric)
        except Exception as e:
            print(f"Warning: Could not read {metrics_file}: {e}")
    
    return metrics

def create_module_status_plot():
    """Create module benchmark status overview."""
    metrics = collect_all_metrics()
    
    if not metrics:
        print("No metrics found for visualization")
        return
    
    status_counts = {}
    modules = []
    
    for metric in metrics:
        status = metric.get('status', 'unknown')
        module = metric.get('module', 'unknown')
        
        modules.append(module)
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Create status overview plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Status pie chart
    statuses = list(status_counts.keys())
    counts = list(status_counts.values())
    colors = {'success': 'green', 'failed': 'red', 'not_found': 'orange', 'unknown': 'gray'}
    plot_colors = [colors.get(s, 'gray') for s in statuses]
    
    ax1.pie(counts, labels=statuses, colors=plot_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Benchmark Status Distribution')
    
    # Module status bar chart
    module_statuses = [m.get('status', 'unknown') for m in metrics]
    module_names = [m.get('module', 'unknown')[:15] + '...' if len(m.get('module', '')) > 15 else m.get('module', 'unknown') for m in metrics]
    
    status_colors = [colors.get(s, 'gray') for s in module_statuses]
    
    ax2.barh(range(len(module_names)), [1]*len(module_names), color=status_colors)
    ax2.set_yticks(range(len(module_names)))
    ax2.set_yticklabels(module_names, fontsize=8)
    ax2.set_xlabel('Status')
    ax2.set_title('Per-Module Benchmark Status')
    ax2.set_xlim(0, 1)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[s], label=s.title()) 
                      for s in colors.keys() if s in statuses]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('target/ecosystem-plots/module_status_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created module status overview plot")

def create_ecosystem_coverage_plot():
    """Create ecosystem benchmark coverage visualization."""
    metrics = collect_all_metrics()
    
    if not metrics:
        return
    
    # Categorize modules
    categories = {
        'Core': ['scirs2', 'scirs2-core'],
        'Math': ['scirs2-linalg', 'scirs2-integrate', 'scirs2-interpolate', 'scirs2-fft', 'scirs2-special'],
        'Data': ['scirs2-signal', 'scirs2-spatial', 'scirs2-sparse', 'scirs2-ndimage'],
        'ML': ['scirs2-neural', 'scirs2-cluster', 'scirs2-metrics'],
        'Other': []
    }
    
    # Classify modules
    category_status = {cat: {'success': 0, 'failed': 0, 'not_found': 0} for cat in categories}
    
    for metric in metrics:
        module = metric.get('module', '')
        status = metric.get('status', 'unknown')
        
        categorized = False
        for cat, modules in categories.items():
            if module in modules:
                if status in category_status[cat]:
                    category_status[cat][status] += 1
                categorized = True
                break
        
        if not categorized:
            if status in category_status['Other']:
                category_status['Other'][status] += 1
    
    # Create stacked bar chart
    categories_list = list(category_status.keys())
    success_counts = [category_status[cat]['success'] for cat in categories_list]
    failed_counts = [category_status[cat]['failed'] for cat in categories_list]
    not_found_counts = [category_status[cat]['not_found'] for cat in categories_list]
    
    x = np.arange(len(categories_list))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x, success_counts, width, label='Success', color='green', alpha=0.8)
    ax.bar(x, failed_counts, width, bottom=success_counts, label='Failed', color='red', alpha=0.8)
    ax.bar(x, not_found_counts, width, 
           bottom=np.array(success_counts) + np.array(failed_counts), 
           label='Not Found', color='orange', alpha=0.8)
    
    ax.set_xlabel('Module Categories')
    ax.set_ylabel('Number of Modules')
    ax.set_title('Ecosystem Benchmark Coverage by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories_list)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target/ecosystem-plots/ecosystem_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created ecosystem coverage plot")

def main():
    os.makedirs('target/ecosystem-plots', exist_ok=True)
    
    print("Creating ecosystem visualization plots...")
    create_module_status_plot()
    create_ecosystem_coverage_plot()
    print("Ecosystem visualization plots created in target/ecosystem-plots/")

if __name__ == '__main__':
    main()
EOF
    
    echo -e "${BLUE}Creating ecosystem visualizations...${NC}"
    if python3 target/create_ecosystem_plots.py > target/ecosystem-benchmarks/visualization_output.log 2>&1; then
        echo -e "${GREEN}✓ Ecosystem visualizations created${NC}"
    else
        echo -e "${RED}✗ Visualization creation failed${NC}"
        echo "Check target/ecosystem-benchmarks/visualization_output.log for details"
    fi
}

# Main execution function
main() {
    check_system_requirements
    
    # Clean previous results
    echo -e "${BLUE}Cleaning previous ecosystem benchmark results...${NC}"
    rm -rf target/ecosystem-benchmarks/* target/ecosystem-plots/*
    
    echo -e "${YELLOW}=== Running Ecosystem Benchmarks ===${NC}"
    echo "Found ${#MODULES_WITH_BENCHMARKS[@]} modules to benchmark"
    
    local successful=0
    local failed=0
    local not_found=0
    
    # Run benchmarks for each module
    for module_info in "${MODULES_WITH_BENCHMARKS[@]}"; do
        run_module_benchmark "$module_info"
        
        # Count results
        IFS=':' read -r module bench_name description <<< "$module_info"
        if [ -f "target/ecosystem-benchmarks/${module}/metrics.json" ]; then
            local status=$(grep -o '"status": "[^"]*"' "target/ecosystem-benchmarks/${module}/metrics.json" | cut -d'"' -f4)
            case $status in
                "success") ((successful++)) ;;
                "failed") ((failed++)) ;;
                "not_found") ((not_found++)) ;;
            esac
        fi
    done
    
    echo -e "${YELLOW}=== Benchmark Summary ===${NC}"
    echo -e "${GREEN}✓ Successful: $successful${NC}"
    echo -e "${RED}✗ Failed: $failed${NC}"
    echo -e "${YELLOW}⚠ Not Found: $not_found${NC}"
    
    # Generate comprehensive report
    generate_ecosystem_report
    
    # Create visualizations
    create_ecosystem_visualization
    
    echo -e "${GREEN}=== Ecosystem Benchmarking Complete ===${NC}"
    echo "Results available in target/ecosystem-benchmarks/"
    echo "Report: target/ecosystem-benchmarks/ECOSYSTEM_REPORT.md"
    echo "Visualizations: target/ecosystem-plots/"
    
    echo ""
    echo "Quick access to HTML reports:"
    for module_dir in target/ecosystem-benchmarks/*/; do
        if [ -d "$module_dir/criterion_reports/report" ]; then
            local module=$(basename "$module_dir")
            echo "  $module: file://$(realpath "$module_dir/criterion_reports/report/index.html")"
        fi
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [--help]"
            echo ""
            echo "Runs comprehensive benchmarks across the entire SciRS2 ecosystem."
            echo ""
            echo "Options:"
            echo "  --help         Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Check system requirements"
            echo "  2. Run benchmarks for all modules with benchmark suites"
            echo "  3. Generate comprehensive reports and visualizations"
            echo "  4. Provide direct links to HTML reports"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main benchmarking
main