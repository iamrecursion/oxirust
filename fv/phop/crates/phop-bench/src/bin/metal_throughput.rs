//! Apple Metal vs CPU forward-evaluation throughput benchmark (M4 perf gate).
//!
//! Times the end-to-end forward evaluation of a fixed EML tree over increasing batch sizes on the
//! Metal backend (`phop_core::MetalEmlEngine`, including host↔device transfer) against the CPU path
//! (`phop_core::eval_tree`), and reports the per-call latency, speedup, and max abs error.
//!
//! Run (requires a macOS machine with a Metal device):
//! ```bash
//! cargo run -p phop-bench --features gpu-metal --release --bin metal_throughput
//! ```

#[cfg(feature = "gpu-metal")]
fn main() {
    use phop_core::{eval_tree, metal_available, EmlTree, MetalEmlEngine};
    use scirs2_core::ndarray::Array2;
    use std::time::Instant;

    if !metal_available() {
        eprintln!("metal_throughput: no Metal device available; nothing to benchmark");
        return;
    }
    let engine = match MetalEmlEngine::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("metal_throughput: failed to create Metal engine: {e}");
            return;
        }
    };
    println!("metal_throughput: device = {}", engine.device_name());

    // A representative depth-2 tree: eml(eml(x0, 1), x1).
    let inner = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
    let tree = EmlTree::eml(&inner, &EmlTree::var(1));

    println!(
        "{:>10}  {:>12}  {:>12}  {:>9}  {:>12}",
        "batch", "GPU ms/call", "CPU ms/call", "speedup", "max abs err"
    );
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = (i % 1000) as f64 * 0.001;
            data[[i, 0]] = t;
            data[[i, 1]] = 1.0 + t;
        }

        // Warm up (first launch compiles + caches the MSL pipeline).
        if let Err(e) = engine.eval_tree(&tree, &data) {
            eprintln!("metal_throughput: GPU warmup failed at n={n}: {e}");
            return;
        }
        let iters = 20;

        let t0 = Instant::now();
        for _ in 0..iters {
            if let Err(e) = engine.eval_tree(&tree, &data) {
                eprintln!("metal_throughput: GPU eval failed at n={n}: {e}");
                return;
            }
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);

        let t1 = Instant::now();
        for _ in 0..iters {
            if let Err(e) = eval_tree(&tree, &data) {
                eprintln!("metal_throughput: CPU eval failed at n={n}: {e}");
                return;
            }
        }
        let cpu_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(iters);

        // Max abs error between one GPU and one CPU forward.
        let gpu_pred = match engine.eval_tree(&tree, &data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("metal_throughput: GPU eval failed at n={n}: {e}");
                return;
            }
        };
        let cpu_pred = match eval_tree(&tree, &data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("metal_throughput: CPU eval failed at n={n}: {e}");
                return;
            }
        };
        let max_abs = gpu_pred
            .iter()
            .zip(cpu_pred.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0_f64, f64::max);

        println!(
            "{n:>10}  {gpu_ms:>12.3}  {cpu_ms:>12.3}  {:>8.1}x  {max_abs:>12.3e}",
            cpu_ms / gpu_ms
        );
    }
}

#[cfg(not(feature = "gpu-metal"))]
fn main() {
    eprintln!("metal_throughput: build with `--features gpu-metal` to run the Metal benchmark");
}
