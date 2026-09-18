//! Demonstration of arbitrary precision special functions
//!
//! This example shows how to use the high-precision feature to compute
//! special functions with arbitrary precision beyond f64 limits.
//!
//! Run with:
//! ```bash
//! cargo run --example arbitrary_precision_demo --features high-precision
//! ```

#[cfg(not(feature = "high-precision"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'high-precision' feature to be enabled.");
    println!(
        "Please run with: cargo run --example arbitrary_precision_demo --features high-precision"
    );
}

#[cfg(feature = "high-precision")]
#[allow(dead_code)]
fn main() {
    use scirs2_special::{
        bessel_j_ap, bessel_y_ap, cleanup_cache, erf_ap, erfc_ap, gamma_ap, log_gamma_ap, to_f64,
        PrecisionContext,
    };

    println!("=== Arbitrary Precision Special Functions Demo ===\n");

    // Create precision contexts with different bit precisions
    let ctx_256 = PrecisionContext::new(256).expect("Operation failed");
    let ctx_512 = PrecisionContext::new(512).expect("Operation failed");
    let ctx_1024 = PrecisionContext::new(1024).expect("Operation failed");

    // Demonstrate Gamma function with increasing precision
    println!("1. Gamma Function with Different Precisions");
    println!("-------------------------------------------");

    let x = 0.5; // Γ(0.5) = √π

    // Standard f64 computation
    let gamma_f64 = scirs2_special::gamma(x);
    println!("Γ(0.5) with f64:        {:.20}", gamma_f64);

    // 256-bit precision
    let gamma_256 = gamma_ap(x, &ctx_256).expect("Operation failed");
    println!("Γ(0.5) with 256 bits:   {:.30}", gamma_256);

    // 512-bit precision
    let gamma_512 = gamma_ap(x, &ctx_512).expect("Operation failed");
    println!("Γ(0.5) with 512 bits:   {:.50}", gamma_512);

    // 1024-bit precision
    let gamma_1024 = gamma_ap(x, &ctx_1024).expect("Operation failed");
    println!("Γ(0.5) with 1024 bits:  {:.80}", gamma_1024);

    // Compare with theoretical value √π
    let sqrt_pi_1024 = ctx_1024.pi().sqrt();
    println!("\n√π with 1024 bits:      {:.80}", sqrt_pi_1024);
    println!("Difference: {:.3e}", (gamma_1024 - sqrt_pi_1024).abs());

    // Demonstrate log-gamma for large arguments
    println!("\n2. Log-Gamma Function for Large Arguments");
    println!("-----------------------------------------");

    let large_x = 100.0;

    // Standard f64 (may have precision loss)
    let log_gamma_f64 = scirs2_special::loggamma(large_x);
    println!("log Γ(100) with f64:     {:.20}", log_gamma_f64);

    // High precision computation
    let log_gamma_hp = log_gamma_ap(large_x, &ctx_512).expect("Operation failed");
    println!("log Γ(100) with 512 bits: {:.50}", log_gamma_hp);

    // Demonstrate Bessel functions
    println!("\n3. Bessel Functions with High Precision");
    println!("----------------------------------------");

    let x = 10.0;
    let n = 5;

    // Compare J_5(10) at different precisions
    let j5_f64 = scirs2_special::jn(n, x);
    let j5_hp = bessel_j_ap(n, x, &ctx_512).expect("Operation failed");

    println!("J_5(10) with f64:      {:.20}", j5_f64);
    println!("J_5(10) with 512 bits: {:.50}", j5_hp);

    // Bessel Y function (second kind)
    let y5_hp = bessel_y_ap(n, x, &ctx_512).expect("Operation failed");
    println!("Y_5(10) with 512 bits: {:.50}", y5_hp);

    // Demonstrate error functions
    println!("\n4. Error Functions with High Precision");
    println!("--------------------------------------");

    let x = 2.5;

    let erf_f64 = scirs2_special::erf(x);
    let erf_hp = erf_ap(x, &ctx_512).expect("Operation failed");
    let erfc_hp = erfc_ap(x, &ctx_512).expect("Operation failed");

    println!("erf(2.5) with f64:      {:.20}", erf_f64);
    println!("erf(2.5) with 512 bits: {:.50}", erf_hp);
    println!("erfc(2.5) with 512 bits: {:.50}", erfc_hp);

    // Verify erf(x) + erfc(x) = 1
    let sum = to_f64(&erf_hp) + to_f64(&erfc_hp);
    println!("\nerf(2.5) + erfc(2.5) = {:.20} (should be exactly 1)", sum);

    // Demonstrate computation of mathematical constants
    println!("\n5. Mathematical Constants with High Precision");
    println!("---------------------------------------------");

    let pi = ctx_1024.pi();
    let e = ctx_1024.e();
    let ln2 = ctx_1024.ln2();
    let euler = ctx_1024.euler_gamma();
    let catalan = ctx_1024.catalan();

    println!("π with 1024 bits:        {:.100}", pi);
    println!("e with 1024 bits:        {:.100}", e);
    println!("ln(2) with 1024 bits:    {:.100}", ln2);
    println!("Euler γ with 1024 bits:  {:.100}", euler);
    println!("Catalan with 1024 bits:  {:.100}", catalan);

    // Demonstrate precision loss in standard f64
    println!("\n6. Precision Loss Example");
    println!("-------------------------");

    // Compute (1 + 1e-16) - 1
    let tiny = 1e-16;
    let result_f64 = (1.0 + tiny) - 1.0;

    let one_hp = ctx_256.float(1.0);
    let tiny_hp = ctx_256.float(tiny);
    let result_hp = (one_hp.clone() + tiny_hp) - one_hp;

    println!("(1 + 1e-16) - 1 with f64:      {:.20e}", result_f64);
    println!("(1 + 1e-16) - 1 with 256 bits: {:.20e}", result_hp);
    println!("Expected value:                1.0000000000000000e-16");

    // Clean up MPFR cache
    cleanup_cache();

    println!("\n=== Demo Complete ===");
}
