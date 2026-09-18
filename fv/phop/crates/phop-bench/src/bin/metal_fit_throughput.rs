//! Apple Metal vs CPU constant-fitting throughput benchmark (M4 perf gate).
//!
//! Times the full constant-fitting step (the expensive autograd loop the discoverer runs per
//! candidate) over a fixed batch of 24 representative EML templates on the Metal backend
//! (`phop_core::MetalEmlEngine::fit_constants`, including host↔device transfer) against the CPU
//! path (`phop_core::fit_constants`), reporting total/per-fit times, the CPU/GPU speedup, and the
//! worst per-template MSE gap so the backends are shown to agree.
//!
//! Honest expectation: at tiny N the per-fit time is latency-bound — host↔device launch overhead
//! dominates the few-hundred-byte transfers — so Metal will likely be SLOWER than the CPU there.
//! The large-N variant shows where the GPU forward+reduce work begins to amortize that launch
//! overhead (the crossover), which is the regime the GPU backend is meant for.
//!
//! Run (requires a macOS machine with a Metal device):
//! ```bash
//! cargo run -p phop-bench --features gpu-metal --release --bin metal_fit_throughput
//! ```

#[cfg(feature = "gpu-metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use phop_core::{
        eval_tree, fit_constants, metal_available, Config, DataSet, EmlTree, MetalEmlEngine,
    };
    use scirs2_core::ndarray::{Array1, Array2};
    use std::time::Instant;

    if !metal_available() {
        eprintln!("metal_fit_throughput: no Metal device available; nothing to benchmark");
        return Ok(());
    }
    let engine = match MetalEmlEngine::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("metal_fit_throughput: failed to create Metal engine: {e}");
            return Ok(());
        }
    };
    println!("metal_fit_throughput: device = {}", engine.device_name());

    /// Number of representative templates fitted on each backend per row.
    const FIT_BUDGET: usize = 24;
    let lr = 0.05;

    // Build exactly 24 representative templates: four EML skeletons (1..=3 const leaves, vars in
    // 0..2) instantiated at six finite const seeds, giving 6 * 4 = 24. Derived seeds (c+0.5, c+1.0)
    // stay modest so every fit is well-conditioned.
    let var0 = EmlTree::var(0);
    let var1 = EmlTree::var(1);
    let cv = EmlTree::const_val;
    let mut templates: Vec<EmlTree> = Vec::with_capacity(FIT_BUDGET);
    for &c in &[0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0] {
        let c2 = c + 0.5;
        let c3 = c + 1.0;
        // Skeleton A (depth-2, 1 const): eml(eml(x0, c), x1).
        templates.push(EmlTree::eml(&EmlTree::eml(&var0, &cv(c)), &var1));
        // Skeleton B (depth-2, 1 const): eml(eml(x0, x1), c).
        templates.push(EmlTree::eml(&EmlTree::eml(&var0, &var1), &cv(c)));
        // Skeleton C (depth-3, 2 consts): eml(eml(x0, c), eml(x1, c2)).
        templates.push(EmlTree::eml(
            &EmlTree::eml(&var0, &cv(c)),
            &EmlTree::eml(&var1, &cv(c2)),
        ));
        // Skeleton D (depth-3, 3 consts): eml(eml(c, x0), eml(c2, eml(x1, c3))).
        templates.push(EmlTree::eml(
            &EmlTree::eml(&cv(c), &var0),
            &EmlTree::eml(&cv(c2), &EmlTree::eml(&var1, &cv(c3))),
        ));
    }
    assert_eq!(templates.len(), FIT_BUDGET);

    // Build a 2-variable dataset of `n` rows (both columns strictly positive) whose target is the
    // smooth truth tree eml(eml(x0, 1), x1), then wrap it as a `DataSet` for fitting.
    let build_ds =
        |n: usize, stride: usize, scale: f64| -> Result<DataSet, Box<dyn std::error::Error>> {
            let mut data = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                let t = (i % stride) as f64 * scale + 0.1;
                data[[i, 0]] = t;
                data[[i, 1]] = 1.0 + t;
            }
            let truth = EmlTree::eml(
                &EmlTree::eml(&EmlTree::var(0), &EmlTree::one()),
                &EmlTree::var(1),
            );
            let y: Array1<f64> = eval_tree(&truth, &data)?;
            let ds =
                DataSet::from_arrays(data, y).map_err(|e| format!("dataset build failed: {e}"))?;
            Ok(ds)
        };

    // ---- Tiny-N regime (latency-bound; Metal expected to be slower) ----
    let ds = build_ds(48, 12, 0.05)?;

    // Warm up once before timing (first launch compiles + caches the MSL pipeline).
    let _ = engine.fit_constants(&templates[0], &ds, lr, 50)?;

    println!(
        "tiny-N constant fit (n = 48, {FIT_BUDGET} templates per backend) — latency-bound regime:"
    );
    println!(
        "{:>7}  {:>13}  {:>13}  {:>13}  {:>13}  {:>9}  {:>12}  {:>12}",
        "epochs",
        "Metal tot ms",
        "CPU tot ms",
        "Metal/fit ms",
        "CPU/fit ms",
        "speedup",
        "med rel",
        "max rel",
    );
    for &e in &[300usize, 600, 2000] {
        // Timed Metal-only pass over all templates, collecting per-template MSEs.
        let mut metal_mses = Vec::with_capacity(FIT_BUDGET);
        let t0 = Instant::now();
        for t in &templates {
            let (_, mse) = engine.fit_constants(t, &ds, lr, e)?;
            metal_mses.push(mse);
        }
        let metal_total_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Timed CPU-only pass over all templates, collecting per-template MSEs.
        let cfg = Config::default().learning_rate(lr).max_epochs(e);
        let mut cpu_mses = Vec::with_capacity(FIT_BUDGET);
        let t1 = Instant::now();
        for t in &templates {
            let (_, mse) = fit_constants(t, &ds, &cfg)?;
            cpu_mses.push(mse);
        }
        let cpu_total_ms = t1.elapsed().as_secs_f64() * 1e3;

        // Per-template relative agreement |metal - cpu| / (|cpu| + 1.0). We report the median
        // (typical agreement, robust to the couple of ill-conditioned nested-`exp` templates whose
        // f32 forward overflows to a huge but finite MSE) alongside the max (that worst outlier).
        let mut rel_gaps: Vec<f64> = metal_mses
            .iter()
            .zip(&cpu_mses)
            .map(|(m, c)| (m - c).abs() / (c.abs() + 1.0))
            .collect();
        rel_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_rel_gap = rel_gaps.get(rel_gaps.len() / 2).copied().unwrap_or(0.0);
        let max_rel_gap = rel_gaps.last().copied().unwrap_or(0.0);
        let metal_per_fit = metal_total_ms / FIT_BUDGET as f64;
        let cpu_per_fit = cpu_total_ms / FIT_BUDGET as f64;
        let speedup = cpu_total_ms / metal_total_ms;
        println!(
            "{e:>7}  {metal_total_ms:>13.3}  {cpu_total_ms:>13.3}  {metal_per_fit:>13.3}  {cpu_per_fit:>13.3}  {speedup:>8.2}x  {med_rel_gap:>12.3e}  {max_rel_gap:>12.3e}"
        );
    }

    // ---- Large-N regime (work-bound; where the GPU amortizes launch overhead) ----
    println!();
    println!(
        "large-N constant fit (epochs = 300, {FIT_BUDGET} templates per backend) — work-bound regime:"
    );
    println!(
        "{:>9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>9}  {:>12}  {:>12}",
        "N",
        "Metal tot ms",
        "CPU tot ms",
        "Metal/fit ms",
        "CPU/fit ms",
        "speedup",
        "med rel",
        "max rel",
    );
    let e = 300usize;
    for &n in &[10_000usize, 100_000] {
        let ds = build_ds(n, 1000, 0.001)?;

        let mut metal_mses = Vec::with_capacity(FIT_BUDGET);
        let t0 = Instant::now();
        for t in &templates {
            let (_, mse) = engine.fit_constants(t, &ds, lr, e)?;
            metal_mses.push(mse);
        }
        let metal_total_ms = t0.elapsed().as_secs_f64() * 1e3;

        let cfg = Config::default().learning_rate(lr).max_epochs(e);
        let mut cpu_mses = Vec::with_capacity(FIT_BUDGET);
        let t1 = Instant::now();
        for t in &templates {
            let (_, mse) = fit_constants(t, &ds, &cfg)?;
            cpu_mses.push(mse);
        }
        let cpu_total_ms = t1.elapsed().as_secs_f64() * 1e3;

        // Per-template relative agreement |metal - cpu| / (|cpu| + 1.0). We report the median
        // (typical agreement, robust to the couple of ill-conditioned nested-`exp` templates whose
        // f32 forward overflows to a huge but finite MSE) alongside the max (that worst outlier).
        let mut rel_gaps: Vec<f64> = metal_mses
            .iter()
            .zip(&cpu_mses)
            .map(|(m, c)| (m - c).abs() / (c.abs() + 1.0))
            .collect();
        rel_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_rel_gap = rel_gaps.get(rel_gaps.len() / 2).copied().unwrap_or(0.0);
        let max_rel_gap = rel_gaps.last().copied().unwrap_or(0.0);
        let metal_per_fit = metal_total_ms / FIT_BUDGET as f64;
        let cpu_per_fit = cpu_total_ms / FIT_BUDGET as f64;
        let speedup = cpu_total_ms / metal_total_ms;
        println!(
            "{n:>9}  {metal_total_ms:>13.3}  {cpu_total_ms:>13.3}  {metal_per_fit:>13.3}  {cpu_per_fit:>13.3}  {speedup:>8.2}x  {med_rel_gap:>12.3e}  {max_rel_gap:>12.3e}"
        );
    }

    Ok(())
}

#[cfg(not(feature = "gpu-metal"))]
fn main() {
    eprintln!(
        "metal_fit_throughput: build with `--features gpu-metal` to run the Metal fit benchmark"
    );
}
