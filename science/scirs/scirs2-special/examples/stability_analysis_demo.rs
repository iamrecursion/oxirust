//! Demonstration of numerical stability analysis for special functions
//!
//! This example shows how to analyze the numerical stability of special functions
//! across extreme parameter ranges and identify potential issues.

use scirs2_special::stability_analysis::run_stability_tests;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Special Functions Numerical Stability Analysis Demo");
    println!("=================================================\n");

    // Run comprehensive stability tests
    println!("Running stability analysis for all special functions...");
    run_stability_tests()?;

    // Demonstrate specific edge cases
    demonstrate_edge_cases();

    // Show recommended practices
    demonstrate_safe_usage();

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_edge_cases() {
    use scirs2_special::{bessel::j0, erf, erfc, gamma, gamma_safe};

    println!("\n\nEdge Cases and Numerical Issues");
    println!("================================\n");

    // Gamma function near negative integers
    println!("1. Gamma function near negative integers:");
    for eps in [1e-10, 1e-8, 1e-6, 1e-4] {
        let x = -1.0 + eps;
        let g = gamma(x);
        let g_safe = gamma_safe(x);
        println!("   gamma({:e}) = {:e}", x, g);
        println!("   gamma_safe({:e}) = {:?}", x, g_safe);
    }

    // Gamma function overflow
    println!("\n2. Gamma function overflow:");
    for x in [170.0, 171.0, 172.0] {
        let g = gamma(x);
        println!("   gamma({}) = {}", x, g);
    }

    // Bessel function for large arguments
    println!("\n3. Bessel J0 for large arguments:");
    for x in [100.0, 1000.0, 10000.0] {
        let j = j0(x);
        let expected_amplitude = (2.0 / (std::f64::consts::PI * x)).sqrt();
        println!(
            "   j0({}) = {:e} (expected amplitude ~ {:e})",
            x, j, expected_amplitude
        );
    }

    // Error function underflow
    println!("\n4. Complementary error function underflow:");
    for x in [5.0, 10.0, 20.0, 30.0] {
        let e = erfc(x);
        println!("   erfc({}) = {:e}", x, e);
    }

    // Catastrophic cancellation
    println!("\n5. Catastrophic cancellation in erf(x) - 1:");
    for x in [2.0, 3.0, 4.0, 5.0] {
        let erf_x = erf(x);
        let diff_bad: f64 = erf_x - 1.0;
        let diff_good: f64 = -erfc(x);
        let rel_error = (diff_bad - diff_good).abs() / diff_good.abs();
        println!(
            "   x={}: bad={:e}, good={:e}, rel_error={:e}",
            x, diff_bad, diff_good, rel_error
        );
    }
}

#[allow(dead_code)]
fn demonstrate_safe_usage() {
    use scirs2_special::{beta_safe, digamma_safe, gamma_safe};

    println!("\n\nSafe Usage Patterns");
    println!("===================\n");

    println!("1. Using safe versions with error handling:");

    // Safe gamma function
    match gamma_safe(-1.0) {
        Ok(val) => println!("   gamma_safe(-1.0) = {}", val),
        Err(e) => println!("   gamma_safe(-1.0) failed: {}", e),
    }

    match gamma_safe(5.0) {
        Ok(val) => println!("   gamma_safe(5.0) = {}", val),
        Err(e) => println!("   gamma_safe(5.0) failed: {}", e),
    }

    // Safe beta function
    match beta_safe(-1.0, 2.0) {
        Ok(val) => println!("   beta_safe(-1.0, 2.0) = {}", val),
        Err(e) => println!("   beta_safe(-1.0, 2.0) failed: {}", e),
    }

    match beta_safe(2.0, 3.0) {
        Ok(val) => println!("   beta_safe(2.0, 3.0) = {}", val),
        Err(e) => println!("   beta_safe(2.0, 3.0) failed: {}", e),
    }

    // Safe digamma function
    match digamma_safe(0.0) {
        Ok(val) => println!("   digamma_safe(0.0) = {}", val),
        Err(e) => println!("   digamma_safe(0.0) failed: {}", e),
    }

    println!("\n2. Recommended parameter ranges:");
    println!("   - Gamma: x ∈ [0.1, 170.0] for safe computation");
    println!("   - Bessel J: x ∈ [1e-6, 100.0], n ∈ [0, 50] for stability");
    println!("   - Error functions: x ∈ [-6, 6] for full precision");
    println!("   - Inverse error functions: p ∈ [-0.999, 0.999] for stability");

    println!("\n3. Handling extreme values:");
    println!("   - Use log-space computations for very large/small values");
    println!("   - Check for overflow/underflow before computation");
    println!("   - Use asymptotic expansions for extreme arguments");
    println!("   - Consider arbitrary precision for critical calculations");
}
