# Wavelet Benchmarks

This document contains benchmarks for various wavelet transform implementations in scirs2-signal. These benchmarks can help users select the most appropriate wavelet for their specific application based on performance characteristics.

## Benchmark Methodology

The benchmark tests the performance of different wavelet transforms on signals of varying sizes. The main metrics measured are:

1. **Decomposition time**: Time taken to decompose a signal into wavelet coefficients
2. **Reconstruction time**: Time taken to reconstruct a signal from wavelet coefficients

For each wavelet family, the benchmark tests:
- Single-level DWT decomposition and reconstruction
- Multi-level DWT decomposition and reconstruction
- Stationary Wavelet Transform (SWT) decomposition and reconstruction
- Wavelet Packet Transform (WPT) decomposition and reconstruction

## Wavelet Families

The following wavelet families are included in the benchmarks:

- **Haar**: The simplest wavelet, with rectangular shape
- **Daubechies (DB)**: Wavelets with different numbers of vanishing moments
- **Symlets (Sym)**: Nearly symmetric wavelets
- **Coiflets (Coif)**: Wavelets with specific vanishing moments properties
- **Meyer**: Wavelets defined in the frequency domain
- **Discrete Meyer (DMeyer)**: More computationally efficient approximation of Meyer wavelets
- **Biorthogonal**: Wavelets with biorthogonal properties

## General Performance Characteristics

Based on complexity and implementation details, the expected performance characteristics are:

1. **Fastest wavelets** (in approximate order):
   - Haar (fastest due to simple coefficients)
   - Daubechies with small order (DB2, DB4)
   - Biorthogonal with small order
   
2. **Medium performance wavelets**:
   - Symlets
   - Coiflets
   - Higher-order Daubechies
   - Discrete Meyer (DMeyer)
   
3. **Slower wavelets**:
   - Meyer (slower due to longer filter lengths)

## Recommendations

Based on the wavelet family properties and performance characteristics:

1. **For speed-critical applications**:
   - Use Haar or low-order Daubechies (DB2, DB4) wavelets
   - Consider using single-level DWT instead of multi-level or SWT

2. **For applications requiring frequency localization**:
   - Use Meyer or DMeyer wavelets (DMeyer offers better performance)
   - Consider Coiflets for a balance between performance and localization

3. **For applications requiring symmetry**:
   - Use Symlets or Biorthogonal wavelets
   - Biorthogonal wavelets with lower orders provide better performance

4. **For signal denoising**:
   - SWT generally provides better results than DWT but is slower
   - Meyer wavelets often work well for smooth signals
   - Daubechies wavelets work well for signals with sharp transitions

## Running the Benchmarks

The full benchmarks are available in the `benches/wavelet_bench.rs` file. You can run them with:

```bash
cargo bench -p scirs2-signal --bench wavelet_bench
```

For quick tests with fewer wavelet families and smaller signal sizes:

```rust
// Example benchmark code for single-level DWT
fn bench_wavelets_single_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_level_dwt");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    
    // Signal sizes to test
    let sizes = [1024];
    
    // Wavelet families to test
    let wavelets = [
        (Wavelet::Haar, "Haar"),
        (Wavelet::DB(4), "DB4"),
        (Wavelet::Meyer, "Meyer"),
        (Wavelet::DMeyer, "DMeyer"),
    ];
    
    for &size in &sizes {
        let signal = generate_signal(size);
        
        for (wavelet, name) in &wavelets {
            // Benchmark decomposition
            group.bench_with_input(
                BenchmarkId::new(format!("{}_decompose", name), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(dwt_decompose(&signal, *wavelet, None).unwrap())
                    })
                },
            );
            
            // Also benchmark reconstruction
            let (approx, detail) = dwt_decompose(&signal, *wavelet, None).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}_reconstruct", name), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(dwt_reconstruct(&approx, &detail, *wavelet).unwrap())
                    })
                },
            );
        }
    }
    
    group.finish();
}
```

## Benchmarking Different Transform Types

In addition to comparing different wavelet families, it's valuable to compare different types of wavelet transforms for specific applications:

### Discrete Wavelet Transform (DWT) vs. Stationary Wavelet Transform (SWT)

- **DWT**: Faster but not translation-invariant
- **SWT**: Translation-invariant but requires more computation and memory
- SWT generally provides better denoising results but at higher computational cost

### Standard DWT vs. Wavelet Packet Transform (WPT)

- **DWT**: Only decomposes approximation coefficients at each level
- **WPT**: Decomposes both approximation and detail coefficients
- WPT provides better frequency resolution but requires more computation
- WPT is beneficial for applications that need detailed frequency analysis

## Example Result Analysis

When analyzing benchmark results, consider these factors:

1. **Filter Length**: Wavelets with longer filter lengths (e.g., Meyer, higher-order Daubechies) are generally slower
2. **Implementation Efficiency**: Some wavelets have optimized implementations
3. **Signal Size**: Performance characteristics may change with signal size
4. **Memory Usage**: Some wavelets require more memory, which can affect cache performance

For specific applications, the best wavelet choice depends on the balance between:
- Performance requirements
- Signal characteristics
- Analysis precision
- Memory constraints