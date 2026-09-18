# scirs2-graph Performance Benchmarks

This directory contains a comprehensive benchmarking infrastructure for scirs2-graph, including performance benchmarks comparing with NetworkX and other graph libraries.

## Overview

The benchmarking infrastructure consists of multiple specialized benchmark suites:

### Core Benchmark Suites

- **`graph_benchmarks.rs`** - Core graph algorithms (BFS, DFS, shortest paths, centrality, etc.)
- **`memory_benchmarks.rs`** - Memory usage comparisons between graph representations
- **`large_graph_stress.rs`** - Large-scale stress testing (up to 5M nodes)
- **`advanced_algorithms.rs`** - Community detection, motifs, embeddings, flow algorithms
- **`performance_optimizations.rs`** - SIMD, parallel processing, memory-mapped graphs
- **`benchmark_runner.rs`** - Automated benchmark execution and reporting

### Key Performance Areas

- **Graph Creation**: Node and edge insertion
- **Traversal**: BFS and DFS algorithms  
- **Shortest Paths**: Dijkstra's algorithm
- **Connectivity**: Connected components, strongly connected components
- **Centrality**: PageRank, betweenness centrality
- **Spanning Trees**: Minimum spanning tree algorithms
- **I/O Operations**: Graph serialization and queries
- **Generators**: Random and scale-free graph generation
- **Advanced Algorithms**: Community detection, motif finding, embeddings
- **Performance Optimizations**: SIMD operations, parallel processing, memory-mapped storage

## Running the Benchmarks

### Comprehensive Benchmark Suite

Use the automated script for complete performance analysis:

```bash
# Full benchmark run with all suites
../scripts/run_comprehensive_benchmarks.sh

# Skip stress tests (for lower-memory systems) 
../scripts/run_comprehensive_benchmarks.sh --skip-stress

# Create performance baseline for future comparisons
../scripts/run_comprehensive_benchmarks.sh --create-baseline

# Custom output directory
../scripts/run_comprehensive_benchmarks.sh --output-dir my_results
```

### Individual Benchmark Suites

```bash
# Core algorithm benchmarks
cargo bench --bench graph_benchmarks

# Memory efficiency benchmarks  
cargo bench --bench memory_benchmarks

# Large-scale stress tests (requires 8GB+ RAM)
cargo bench --bench large_graph_stress

# Advanced algorithm benchmarks
cargo bench --bench advanced_algorithms

# Performance optimization benchmarks
cargo bench --bench performance_optimizations

# Automated benchmark runner
cargo bench --bench benchmark_runner
```

### Traditional Benchmark Groups

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench graph_creation
cargo bench traversal  
cargo bench centrality

# Generate HTML report
cargo bench -- --save-baseline main
```

Results are saved in `target/criterion/` with detailed HTML reports.

### Python Benchmarks (NetworkX)

```bash
# Install dependencies
pip install networkx numpy pandas matplotlib

# Run NetworkX benchmarks
cd benches
python networkx_comparison.py
```

This generates:
- `networkx_benchmark_results.json` - Raw timing data
- `networkx_benchmarks.csv` - Detailed results table

### Comparing Results

After running both benchmark suites:

```bash
python compare_results.py
```

This creates:
- `benchmark_comparison.md` - Detailed comparison report
- `performance_comparison.png` - Performance charts
- `speedup_distribution.png` - Speedup analysis

## Advanced Benchmark Suites

### Community Detection (`advanced_algorithms.rs`)

Tests state-of-the-art community detection algorithms:

- **Louvain Algorithm**: Modularity optimization for community detection
- **Label Propagation**: Fast community detection based on label spreading
- **Greedy Modularity**: Greedy approach to modularity optimization
- **K-Clique Percolation**: Finding overlapping communities

### Motif Finding (`advanced_algorithms.rs`)

Benchmarks graph pattern recognition:

- **Triangle Counting**: Efficient triangle enumeration and counting
- **4-Motif Analysis**: Detection of 4-node subgraph patterns
- **K-Clique Detection**: Finding cliques of arbitrary size
- **Subgraph Isomorphism**: Pattern matching in graphs

### Graph Embeddings (`advanced_algorithms.rs`)

Tests embedding generation algorithms:

- **Node2Vec**: Biased random walk embeddings with p/q parameters
- **DeepWalk**: Random walk based embeddings
- **Skip-gram Training**: Word2Vec-style training for graph embeddings
- **Random Walk Generation**: Various random walk strategies

### Flow Algorithms (`advanced_algorithms.rs`)

Benchmarks network flow computations:

- **Ford-Fulkerson**: Classic maximum flow algorithm
- **Edmonds-Karp**: BFS-based max flow implementation  
- **Push-Relabel**: Preflow-based maximum flow
- **Min-Cost Max Flow**: Cost-constrained flow optimization

### SIMD Operations (`performance_optimizations.rs`)

Tests vectorized operations vs scalar fallbacks:

- **Vector Operations**: Addition, dot product, normalization
- **Distance Calculations**: Euclidean, cosine similarity
- **Platform Detection**: Automatic SIMD capability detection
- **Fallback Performance**: Scalar implementations for compatibility

### Memory-Mapped Graphs (`performance_optimizations.rs`)

Benchmarks out-of-core graph processing:

- **Disk-Based Storage**: CSR format on disk with memory mapping
- **Streaming Operations**: Processing graphs larger than RAM
- **Batch Access**: Efficient bulk neighbor queries
- **I/O Performance**: Disk vs memory access patterns

### Parallel Processing (`performance_optimizations.rs`)

Compares parallel vs sequential implementations:

- **Multi-Source BFS**: Parallel breadth-first search from multiple roots
- **Batch Centrality**: Parallel centrality computation
- **Degree Computation**: Parallel degree calculation
- **Thread Scaling**: Performance scaling with thread count

## Benchmark Design

### Graph Sizes

Different algorithms use appropriate graph sizes:
- **Creation/Traversal**: 100, 1,000, 10,000 nodes
- **Shortest Paths**: 100, 500, 1,000 nodes
- **Centrality**: 50, 100, 200 nodes (computationally expensive)
- **Connectivity**: 100, 1,000, 5,000 nodes

### Graph Types

- **Sparse Random Graphs**: Erdős–Rényi with p=0.01
- **Dense Random Graphs**: Erdős–Rényi with p=0.1
- **Scale-Free Graphs**: Barabási–Albert preferential attachment

### Methodology

- **Rust**: Uses Criterion.rs with statistical analysis
  - Multiple iterations with outlier detection
  - Confidence intervals and regression analysis
  - Automatic warm-up detection

- **Python**: Uses time.perf_counter() 
  - High-resolution timing
  - Consistent with standard Python benchmarking practices

## System Requirements

### Minimum Requirements
- **RAM**: 4GB for basic benchmarks
- **CPU**: 2+ cores recommended  
- **Disk**: 1GB free space for results
- **OS**: Linux, macOS, or Windows

### Recommended for Full Suite
- **RAM**: 16GB+ for stress tests
- **CPU**: 8+ cores for parallel benchmarks
- **Disk**: 5GB for comprehensive results
- **CPU Features**: AVX2 support for SIMD benchmarks

## Expected Results

### Core Algorithm Performance

Based on typical Rust vs Python performance:

| Operation | Expected Speedup |
|-----------|------------------|
| Graph Creation | 5-10x |
| Traversal (BFS/DFS) | 10-20x |
| Shortest Paths | 15-25x |
| Centrality Measures | 20-50x |
| Large Graph Operations | 30-100x |

### Advanced Algorithm Performance

Expected performance characteristics:

| Algorithm Category | 1K nodes | 10K nodes | 100K nodes |
|-------------------|----------|-----------|-------------|
| Community Detection | < 50ms | < 5s | < 2min |
| Motif Finding | < 10ms | < 1s | < 30s |
| Graph Embeddings | < 500ms | < 30s | < 10min |
| Flow Algorithms | < 20ms | < 2s | < 1min |

### Optimization Performance

| Optimization | Improvement | Conditions |
|-------------|-------------|------------|
| SIMD Operations | 2-4x | AVX2 compatible CPU |
| Parallel Processing | 2-8x | Multi-core systems |
| Memory-Mapped I/O | 10-100x | Large graphs (>1M nodes) |
| Lazy Evaluation | 1-1000x | Repeated computations |

Actual speedups depend on:
- Graph size and structure
- Algorithm complexity
- Memory access patterns
- CPU cache efficiency
- Hardware capabilities

## Performance Tips

### For scirs2-graph

1. **Enable optimizations**: Build with `--release`
2. **Use appropriate types**: Prefer `u32` node indices for most graphs
3. **Batch operations**: Add multiple edges/nodes at once
4. **Parallel algorithms**: Enable when graph size > 10,000 nodes

### For NetworkX

1. **Use NumPy arrays**: For numerical computations
2. **Avoid Python loops**: Use built-in NetworkX functions
3. **Consider graph views**: For read-only operations
4. **Profile bottlenecks**: Use cProfile for detailed analysis

## Extending Benchmarks

To add new benchmarks:

1. **Rust**: Add function to `graph_benchmarks.rs`:
```rust
fn bench_new_algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_algorithm");
    // ... benchmark implementation
}
```

2. **Python**: Add function to `networkx_comparison.py`:
```python
def benchmark_new_algorithm(sizes: List[int]) -> Dict[str, List[float]]:
    # ... benchmark implementation
```

3. Update `compare_results.py` to include the new algorithm in visualizations.

## Troubleshooting

### Common Issues

1. **Memory usage**: Large graphs may require significant RAM
   - Solution: Reduce maximum graph size or use a machine with more memory

2. **Long runtime**: Some algorithms are computationally expensive
   - Solution: Reduce sample size with `group.sample_size(10)`

3. **Inconsistent results**: System load affects timing
   - Solution: Run benchmarks on an idle system, close other applications

### Platform-Specific Notes

- **Linux**: Best performance, most consistent results
- **macOS**: Good performance, may need to allow terminal performance access
- **Windows**: Higher variance, ensure power settings favor performance

## Contributing

When submitting performance improvements:

1. Run benchmarks before and after changes
2. Include benchmark results in PR description
3. Explain the optimization technique used
4. Ensure no regression in other algorithms

## Large Graph Stress Testing

In addition to standard benchmarks, we provide comprehensive stress tests for graphs with >1M nodes.

### Running Stress Tests

```bash
# Quick start - run all stress tests
./run_stress_tests.sh --release

# Individual stress tests
cargo test --release stress_tests -- --ignored --test-threads=1 --nocapture

# Specific test
cargo test --release test_large_barabasi_albert_graph -- --ignored --nocapture
```

### Available Stress Tests

Located in `../tests/stress_tests.rs`:

1. **test_large_erdos_renyi_graph** - Random graphs up to 1M nodes
2. **test_large_barabasi_albert_graph** - Scale-free networks up to 1M nodes  
3. **test_large_grid_graph** - 2D grid graphs up to 1M nodes
4. **test_large_directed_graph_algorithms** - Directed graph algorithms at scale
5. **test_memory_efficient_operations** - Memory-conscious processing
6. **test_parallel_algorithms_on_large_graphs** - Parallel processing performance
7. **test_extreme_scale_graph** - 5M node graphs (requires >2GB RAM)
8. **test_algorithm_scaling** - How algorithms scale with graph size

### Stress Test Results

Expected performance for large graphs:

| Graph Size | Type | Generation Time | Memory Usage |
|------------|------|----------------|--------------|
| 100K nodes | Sparse BA | <1s | ~10MB |
| 500K nodes | Sparse BA | <5s | ~50MB |
| 1M nodes | Sparse BA | <10s | ~100MB |
| 1M nodes | Grid | <5s | ~80MB |
| 5M nodes | Sparse BA | <60s | ~500MB |

### Large Graph Comparison with NetworkX

For direct comparison on large graphs:

```bash
# Run NetworkX large graph benchmarks
python large_graph_comparison.py

# This generates:
# - large_graph_benchmark_results.csv
# - large_graph_benchmark_plots.png
# - Memory usage analysis
```

See [STRESS_TESTING_GUIDE.md](../docs/STRESS_TESTING_GUIDE.md) for detailed instructions.

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [NetworkX Performance](https://networkx.org/documentation/stable/reference/performance.html)
- [Graph Algorithm Complexity](https://en.wikipedia.org/wiki/Graph_algorithms)
- [Stress Testing Guide](../docs/STRESS_TESTING_GUIDE.md)