//! GPU vs CPU forward-evaluation throughput benchmark (M4 perf gate).
//!
//! Times the end-to-end forward evaluation of a fixed EML tree over increasing batch sizes on
//! the CUDA backend (`phop_core::CudaEmlEngine`, including host↔device transfer) against the CPU
//! path (`phop_core::eval_tree`), and reports the per-call latency and speedup.
//!
//! Run (requires an NVIDIA GPU and the CUDA driver on the loader path):
//! ```bash
//! LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
//!   cargo run -p phop-bench --features gpu-cuda --release --bin gpu_throughput
//! ```

#[cfg(feature = "gpu-cuda")]
fn main() {
    use phop_core::{cuda_available, eval_tree, CudaEmlEngine, EmlTree};
    use scirs2_core::ndarray::Array2;
    use std::time::Instant;

    if !cuda_available() {
        eprintln!("gpu_throughput: no CUDA device available; nothing to benchmark");
        return;
    }
    let engine = CudaEmlEngine::new().expect("CUDA engine");

    // A representative depth-2 tree: eml(eml(x0, 1), x1).
    let inner = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
    let tree = EmlTree::eml(&inner, &EmlTree::var(1));

    println!(
        "{:>10}  {:>12}  {:>12}  {:>9}",
        "batch", "GPU ms/call", "CPU ms/call", "speedup"
    );
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = (i % 1000) as f64 * 0.001;
            data[[i, 0]] = t;
            data[[i, 1]] = 1.0 + t;
        }

        // Warm up (first launch pays JIT/alloc costs).
        let _ = engine.eval_tree(&tree, &data).expect("gpu warmup");
        let iters = 20;

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = engine.eval_tree(&tree, &data).expect("gpu eval");
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);

        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = eval_tree(&tree, &data).expect("cpu eval");
        }
        let cpu_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(iters);

        println!(
            "{n:>10}  {gpu_ms:>12.3}  {cpu_ms:>12.3}  {:>8.1}x",
            cpu_ms / gpu_ms
        );
    }

    // GPU-resident training loop: time per-epoch cost of fit_constants (the M4 "1 epoch" gate).
    use phop_core::DataSet;
    use scirs2_core::ndarray::Array1;
    println!("\nGPU-resident fit (eml(x0, c), 1 constant) — per-epoch cost:");
    println!("{:>10}  {:>14}", "batch", "ms/epoch");
    let fit_tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
    for &n in &[100_000usize, 1_000_000] {
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i % 1000) as f64 * 0.002;
            x[[i, 0]] = t;
            y[i] = t.exp() - 3.0_f64.ln();
        }
        let ds = DataSet::from_arrays(x, Array1::from(y)).expect("ds");
        let epochs = 30usize;
        let t = Instant::now();
        let _ = engine
            .fit_constants(&fit_tree, &ds, 0.1, epochs)
            .expect("fit");
        let ms_per_epoch = t.elapsed().as_secs_f64() * 1e3 / epochs as f64;
        println!("{n:>10}  {ms_per_epoch:>14.3}");
    }
}

#[cfg(not(feature = "gpu-cuda"))]
fn main() {
    eprintln!("gpu_throughput: build with `--features gpu-cuda` to run the GPU benchmark");
}
