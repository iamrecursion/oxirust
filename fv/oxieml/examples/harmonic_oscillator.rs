//! Harmonic-oscillator pipeline: recover `x(t) = A · cos(ω·t)` from data.
//!
//! Simple harmonic motion — a mass on an ideal spring, the linearised
//! pendulum, an LC tank circuit, and a thousand other textbook systems —
//! has displacement `x(t) = A · cos(ω · t)` where `A` is the amplitude and
//! `ω = 2π/T` the angular frequency. This is a three-variable symbolic-
//! regression target (`t`, `A`, `ω`) with a cosine non-linearity, which
//! the EML lowering pipeline recognises natively and evaluates at full
//! `f64::cos` precision.
//!
//! The workflow mirrors `examples/physics_pipeline.rs`:
//!   1. Sample `(t, A, ω, x)` tuples under ground truth.
//!   2. Let `SymRegEngine` discover a closed-form fit (max_depth = 5).
//!   3. Lower + simplify + pretty-print + compile to Rust.
//!   4. Evaluate on held-out data, report MSE / max-error / a tiny table.
//!   5. Probe a single point via `eval_real_lowered` for full precision.
//!
//! Run with: `cargo run --example harmonic_oscillator --release`

use oxieml::compile::compile_to_rust_batch;
use oxieml::{EvalCtx, SymRegConfig, SymRegEngine};

/// Ground-truth displacement of a simple harmonic oscillator:
/// `x(t) = A · cos(ω · t)`.
///
/// Inputs: `vars[0] = t` (seconds), `vars[1] = A` (metres),
/// `vars[2] = ω` (rad/s).
fn truth(t: f64, amplitude: f64, omega: f64) -> f64 {
    amplitude * (omega * t).cos()
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
        let t = 2.0 * next(); // seconds, [0, 2]
        let amplitude = 0.5 + 1.5 * next(); // metres, [0.5, 2.0]
        let omega = 1.0 + 4.0 * next(); // rad/s, [1.0, 5.0]
        inputs.push(vec![t, amplitude, omega]);
        targets.push(truth(t, amplitude, omega));
    }
    (inputs, targets)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiEML harmonic-oscillator demo ===\n");

    // Step 1: generate training data
    let (inputs, targets) = generate_dataset(200, 0xFADE_D00D);
    let min_t = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let max_t = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Generated {} training samples: target range [{:.3}, {:.3}] m",
        inputs.len(),
        min_t,
        max_t
    );

    // Step 2: symbolic regression.
    //
    // Three variables and a cosine non-linearity call for a slightly deeper
    // tree than the single-variable pendulum demo. `max_depth: 5` keeps the
    // search tractable while still admitting the Euler-style cosine
    // construction that the lowering pipeline recognises and collapses.
    let config = SymRegConfig {
        max_depth: 5,
        ..Default::default()
    };
    let engine = SymRegEngine::new(config);
    println!("Running symbolic regression (this may take a while)...");
    let formulas = engine.discover(&inputs, &targets, 3)?;

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
    let rust_src = compile_to_rust_batch(&best.eml_tree, "oscillator_displacement");
    println!("\nCompiled Rust (single-point + batch):\n----------------------------");
    print!("{rust_src}");
    println!("----------------------------");

    // Step 5: evaluate on a held-out test set.
    //
    // Primary path: `EmlTree::eval_batch`. If some input drives a subtree
    // into a genuinely complex value (e.g. ln of a negative), fall back to
    // the lowered stack machine which keeps everything in f64 and silently
    // propagates NaN/inf for downstream filtering.
    let (test_inputs, test_targets) = generate_dataset(60, 0xDEAD_F00D);
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

    // Filter out non-finite results for an honest MSE.
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
    println!("\n  idx     t       A       omega     truth       predicted     |error|");
    println!("  ----  -----   -----   -------    ---------   -----------    --------");
    for (i, (input, (pred, truth_val))) in test_inputs
        .iter()
        .zip(preds.iter().zip(test_targets.iter()))
        .take(5)
        .enumerate()
    {
        let err = (pred - truth_val).abs();
        println!(
            "  {:>3}   {:>4.2}    {:>4.2}    {:>5.2}     {:>9.4}   {:>10.4}    {:>8.5}",
            i, input[0], input[1], input[2], truth_val, pred, err
        );
    }

    // Single-point probe: route one sample through `eval_real_lowered`,
    // which lowers + simplifies and dispatches through the stack machine
    // so recognised trig patterns use native `f64::cos` (~1e-15 precision).
    let probe = [0.5_f64, 1.0_f64, 2.0_f64];
    let probe_ctx = EvalCtx::new(&probe);
    match best.eml_tree.eval_real_lowered(&probe_ctx) {
        Ok(value) => {
            let reference = truth(probe[0], probe[1], probe[2]);
            println!(
                "\nSingle-point probe (eval_real_lowered): t = {:.2}, A = {:.2}, omega = {:.2} \
                 -> x = {:.6} (truth {:.6}, |delta| = {:.2e})",
                probe[0],
                probe[1],
                probe[2],
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
