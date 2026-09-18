//! Physics pipeline: discover Newtonian formulas from simulated data.
//!
//! This example demonstrates the end-to-end OxiEML workflow:
//!   1. Generate projectile-motion data under gravity.
//!   2. Let `SymRegEngine` discover a closed-form height-vs-time formula.
//!   3. Lower + simplify the discovered formula.
//!   4. Pretty-print and compile to runnable Rust source.
//!   5. Evaluate the tree on held-out test points and compare to ground truth.
//!
//! Run with: `cargo run --example physics_pipeline --release`

use oxieml::compile::compile_to_rust_batch;
use oxieml::{SymRegConfig, SymRegEngine};

const G: f64 = 9.81; // m/s^2

/// Ground-truth projectile height: `h(t) = v0 * t - 0.5 * g * t^2`.
///
/// Inputs: `vars[0] = t` (seconds), `vars[1] = v0` (initial vertical velocity, m/s).
fn truth(t: f64, v0: f64) -> f64 {
    v0 * t - 0.5 * G * t * t
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
        let t = 0.5 + 4.0 * next(); // seconds, roughly [0.5, 4.5]
        let v0 = 5.0 + 25.0 * next(); // m/s, roughly [5, 30]
        inputs.push(vec![t, v0]);
        targets.push(truth(t, v0));
    }
    (inputs, targets)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiEML physics pipeline demo ===\n");

    // Step 1: generate training data
    let (inputs, targets) = generate_dataset(120, 0xC001_F00D);
    let min_t = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let max_t = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Generated {} training samples: target range [{:.2}, {:.2}]",
        inputs.len(),
        min_t,
        max_t
    );

    // Step 2: symbolic regression
    //
    // We keep the search shallow so the demo finishes quickly. SymRegEngine
    // enumerates all EML topologies up to `max_depth`, then fits continuous
    // parameters via Adam. The projectile formula h(t) = v0*t - 0.5*g*t^2
    // is not directly representable in a tiny EML tree, but the search will
    // still find a useful approximation.
    let config = SymRegConfig {
        max_depth: 2,
        learning_rate: 5e-3,
        tolerance: 1e-8,
        max_iter: 800,
        complexity_penalty: 1e-4,
        num_restarts: 2,
        integer_rounding: false,
        cv_folds: None,
        ..SymRegConfig::default()
    };
    let engine = SymRegEngine::new(config);
    println!("Running symbolic regression (this may take a few seconds)...");
    let formulas = engine.discover(&inputs, &targets, 2)?;

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
    // `DiscoveredFormula::pretty` already reflects the simplified lowering,
    // but we do it explicitly here to show the moving parts.
    let lowered = best.eml_tree.lower();
    let simplified = lowered.simplify();
    println!("\nLowered IR:            {lowered}");
    println!("Lowered + simplified:  {simplified}");

    // Step 4: compile to Rust source code
    let rust_src = compile_to_rust_batch(&best.eml_tree, "predict_height");
    println!("\nCompiled Rust (single-point + batch):\n----------------------------");
    print!("{rust_src}");
    println!("----------------------------");

    // Step 5: evaluate on a held-out test set.
    //
    // We use `EmlTree::eval_batch` as the primary runtime path. It evaluates
    // through the EML operator using complex intermediates, which can fail
    // for inputs that drive some subtree into a genuinely complex value
    // (e.g. ln of a large negative number). When that happens we gracefully
    // fall back to the `LoweredOp::eval_batch` evaluator, which uses plain
    // f64 arithmetic and propagates NaN/inf silently.
    let (test_inputs, test_targets) = generate_dataset(40, 0xDEAD_BEEF);
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
    println!("  MSE:              {mse:.4}");
    println!("  max |error|:      {max_abs_err:.4}");

    // A small sanity-check table: show the first 5 predictions vs. truth.
    println!("\n  idx    t      v0       truth      predicted    |error|");
    println!("  ----  -----  ------   ---------   ----------   --------");
    for (i, (input, (pred, truth_val))) in test_inputs
        .iter()
        .zip(preds.iter().zip(test_targets.iter()))
        .take(5)
        .enumerate()
    {
        let err = (pred - truth_val).abs();
        println!(
            "  {:>3}   {:>4.2}   {:>5.2}   {:>9.3}   {:>10.3}   {:>8.4}",
            i, input[0], input[1], truth_val, pred, err
        );
    }

    // Sanity-check: compare against the LoweredOp::eval_batch evaluator.
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
