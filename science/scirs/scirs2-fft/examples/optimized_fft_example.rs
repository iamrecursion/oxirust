// The optimized_fft module is currently gated behind a feature flag
// This example is temporarily disabled until the module is available

#[allow(dead_code)]
fn main() {
    println!("Optimized FFT Example");
    println!("--------------------");
    println!("This example is currently disabled as the optimized_fft module is not available.");
    println!("Please use other examples in this directory for FFT functionality.");

    // Explain what optimized FFT is about
    println!("\nOptimized FFT features include:");
    println!("1. Automatic algorithm selection based on input size");
    println!("2. SIMD optimizations for compatible processors");
    println!("3. Cache-aware implementations");
    println!("4. Multi-threaded parallel computation");
    println!("5. Mixed-radix FFT algorithms");

    println!("\nOptimization levels:");
    println!("- Basic: Standard FFT implementation");
    println!("- Auto: Automatic algorithm selection");
    println!("- SIMD: Use SIMD instructions where available");
    println!("- Parallel: Multi-threaded execution");
    println!("- Aggressive: All optimizations enabled");
}
