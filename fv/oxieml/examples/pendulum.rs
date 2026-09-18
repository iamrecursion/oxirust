//! Pendulum-period pipeline: rediscover the small-angle formula from data.
//!
//! A simple (massless-rod, point-mass) pendulum swings with period
//! `T(L) = 2·π · sqrt(L / g)` in the small-angle regime, where
//! `g = 9.81 m/s^2`. This is a clean single-input / single-output
//! symbolic-regression target: one variable, a square root, and two
//! constants folded together.
//!
//! The workflow mirrors `examples/physics_pipeline.rs`:
//!   1. Sample (length, period) pairs under ground truth.
//!   2. Let `SymRegEngine` discover a closed-form fit.
//!   3. Lower + simplify + pretty-print + compile to Rust.
//!   4. Evaluate on held-out data, report MSE / max-error / a tiny table.
//!
//! Run with: `cargo run --example pendulum --release`

use oxieml::compile::compile_to_rust_batch;
use oxieml::{EvalCtx, SymRegConfig, SymRegEngine};

const G: f64 = 9.81; // m/s^2

/// Ground-truth pendulum period: `T(L) = 2·π · sqrt(L / g)`.
///
/// Inputs: `vars[0] = L` (rod length in metres).
fn truth(length: f64) -> f64 {
    2.0 * std::f64::consts::PI * (length / G).sqrt()
}

/// Generate a synthetic dataset using a tiny inline LCG so the example has
/// no RNG dependency beyond `oxieml` itself.
fn generate_dataset(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Take the high 32 bits for uniform [0, 1).
        let bits = (state >> 32) as u32;
        f64::from(bits) / f64::from(u32::MAX)
    };

    let mut inputs = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for _ in 0..n {
        // Rod length in metres, uniformly in [0.1, 2.0].
        let length = 0.1 + 1.9 * next();
        inputs.push(vec![length]);
        targets.push(truth(length));
    }
    (inputs, targets)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiEML pendulum-period demo ===\n");

    // Step 1: generate training data
    let (inputs, targets) = generate_dataset(150, 0xBADC_0FFE);
    let min_t = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let max_t = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Generated {} training samples: target range [{:.3}, {:.3}] s",
        inputs.len(),
        min_t,
        max_t
    );

    // Step 2: symbolic regression.
    //
    // A single-variable formula with a square root lives comfortably in a
    // depth-4 EML tree, so we start from the defaults and only override
    // `max_depth` to keep the search honest.
    let config = SymRegConfig {
        max_depth: 4,
        ..Default::default()
    };
    let engine = SymRegEngine::new(config);
    println!("Running symbolic regression (this may take a few seconds)...");
    let formulas = engine.discover(&inputs, &targets, 1)?;

    let best = match formulas.first() {
        Some(f) => f,
        None => {
            eprintln!("no formulas discovered");
            return Ok(());
        }
    };
    println!("\nBest discovered formula (rank 1 of {}):", formulas.len());
    println!("  score:      {:.6}", best.score);
    println!("  MSE:        {:.6}", best.mse);
    println!("  complexity: {}", best.complexity);
    println!("  pretty:     {}", best.pretty);
    println!("  params:     {:?}", best.params);

    // Step 3: lower + simplify + pretty-print the tree itself.
    let lowered = best.eml_tree.lower();
    let simplified = lowered.simplify();
    println!("\nLowered IR:            {lowered}");
    println!("Lowered + simplified:  {simplified}");

    // Step 4: compile to Rust source code
    let rust_src = compile_to_rust_batch(&best.eml_tree, "pendulum_period");
    println!("\nCompiled Rust (single-point + batch):\n----------------------------");
    print!("{rust_src}");
    println!("----------------------------");

    // Step 5: evaluate on a held-out test set.
    //
    // Primary path: `EmlTree::eval_batch` (complex-arithmetic). If it fails
    // on some input (e.g. ln of a non-positive value) fall back to the
    // lowered stack machine which runs on plain f64.
    let (test_inputs, test_targets) = generate_dataset(50, 0xC0DE_F00D);
    let preds = match best.eml_tree.eval_batch(&test_inputs) {
        Ok(preds) => {
            println!("\nUsing EmlTree::eval_batch for held-out evaluation.");
            preds
        }
        Err(e) => {
            println!(
                "\nEmlTree::eval_batch produced {e:?}; falling back to LoweredOp::eval_batch."
            );
            simplified.eval_batch(&test_inputs)
        }
    };

    // Filter out non-finite results (NaN/inf from log of non-positives, etc.)
    // for honest MSE accounting, and report how many points survived.
    let pairs: Vec<(f64, f64)> = preds
        .iter()
        .zip(test_targets.iter())
        .filter_map(|(&p, &y)| if p.is_finite() { Some((p, y)) } else { None })
        .collect();
    let valid = pairs.len();
    let total = preds.len();

    let mut mse = 0.0;
    let mut max_abs_err = 0.0_f64;
    for (p, y) in &pairs {
        let d = p - y;
        mse += d * d;
        max_abs_err = max_abs_err.max(d.abs());
    }
    if valid > 0 {
        mse /= valid as f64;
    } else {
        mse = f64::NAN;
    }

    println!("\nHeld-out test set ({total} points, {valid} finite):");
    println!("  MSE:              {mse:.6}");
    println!("  max |error|:      {max_abs_err:.6}");

    // A small sanity-check table: show the first 5 predictions vs. truth.
    println!("\n  idx     L         truth        predicted     |error|");
    println!("  ----  ------     ---------    -----------    --------");
    for (i, (input, (pred, truth_val))) in test_inputs
        .iter()
        .zip(preds.iter().zip(test_targets.iter()))
        .take(5)
        .enumerate()
    {
        let err = (pred - truth_val).abs();
        println!(
            "  {:>3}   {:>5.3}      {:>8.4}     {:>10.4}     {:>8.5}",
            i, input[0], truth_val, pred, err
        );
    }

    // Extra sanity check: evaluate a single representative point via
    // `eval_real_lowered` — a convenience that returns native f64 precision
    // by routing the simplified lowered IR through the stack machine.
    let probe_length = 1.0_f64;
    let probe_ctx = EvalCtx::new(&[probe_length]);
    match best.eml_tree.eval_real_lowered(&probe_ctx) {
        Ok(value) => {
            let reference = truth(probe_length);
            println!(
                "\nSingle-point probe (eval_real_lowered): L = {:.3} m -> T = {:.6} s \
                 (truth {:.6} s, |delta| = {:.2e})",
                probe_length,
                value,
                reference,
                (value - reference).abs()
            );
        }
        Err(e) => {
            println!("\nSingle-point probe via eval_real_lowered failed: {e:?}");
        }
    }

    // Sanity check: compare against the LoweredOp::eval_batch evaluator.
    // The two evaluators should agree at every finite point.
    let lowered_preds = simplified.eval_batch(&test_inputs);
    let mut max_delta = 0.0_f64;
    let mut compared = 0usize;
    for (a, b) in preds.iter().zip(lowered_preds.iter()) {
        if a.is_finite() && b.is_finite() {
            max_delta = max_delta.max((a - b).abs());
            compared += 1;
        }
    }
    println!(
        "\nEmlTree vs LoweredOp batch evaluator agree to within {max_delta:.2e} \
         (max abs delta over {compared} finite points)."
    );

    println!("\nDone.");
    Ok(())
}
