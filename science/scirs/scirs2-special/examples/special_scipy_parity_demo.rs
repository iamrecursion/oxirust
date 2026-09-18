//! Demonstration of newly implemented SciPy parity functions
//!
//! This example showcases the newly implemented functions that improve
//! SciPy compatibility, including exponentially scaled Bessel functions,
//! Dawson's integral, and the polygamma function.

use scirs2_special::{
    // Dawson's integral
    dawsn,
    // Exponentially scaled Bessel functions
    i0e,
    i1e,
    ive,
    j0e,
    j1e,
    jne,
    jve,
    k0e,
    k1e,
    kve,
    // Polygamma function
    polygamma,
    y0e,
    y1e,
    yne,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciPy Parity Functions Demo ===\n");

    // Test exponentially scaled Bessel functions
    println!("1. Exponentially Scaled Bessel Functions:");
    let x = 10.0f64;

    println!("  Regular vs Exponentially Scaled (x = {}):", x);
    println!("    j0e({}) = {:.6e}", x, j0e(x));
    println!("    j1e({}) = {:.6e}", x, j1e(x));
    println!("    jne(5, {}) = {:.6e}", x, jne(5, x));
    println!("    jve(2.5, {}) = {:.6e}", x, jve(2.5, x));

    println!("    y0e({}) = {:.6e}", x, y0e(x));
    println!("    y1e({}) = {:.6e}", x, y1e(x));
    println!("    yne(3, {}) = {:.6e}", x, yne(3, x));

    println!("    i0e({}) = {:.6e}", x, i0e(x));
    println!("    i1e({}) = {:.6e}", x, i1e(x));
    println!("    ive(2.5, {}) = {:.6e}", x, ive(2.5, x));

    println!("    k0e({}) = {:.6e}", x, k0e(x));
    println!("    k1e({}) = {:.6e}", x, k1e(x));
    println!("    kve(2.5, {}) = {:.6e}", x, kve(2.5, x));

    // Test Dawson's integral
    println!("\n2. Dawson's Integral:");
    for &x_val in &[0.0, 0.5, 1.0, 2.0, 5.0] {
        println!("    dawsn({}) = {:.10}", x_val, dawsn(x_val));
    }

    // Verify D(-x) = -D(x)
    let x_test = 2.5f64;
    let d_pos = dawsn(x_test);
    let d_neg = dawsn(-x_test);
    println!(
        "    Odd function test: dawsn({}) = {:.6}, dawsn({}) = {:.6}",
        x_test, d_pos, -x_test, d_neg
    );
    let diff = (d_pos + d_neg).abs();
    println!("    Difference: {:.2e} (should be ~0)", diff);

    // Test polygamma function
    println!("\n3. Polygamma Function:");
    for n in 0..=4 {
        for &x_val in &[1.0, 2.0, 5.0] {
            println!(
                "    polygamma({}, {}) = {:.8}",
                n,
                x_val,
                polygamma(n, x_val)
            );
        }
    }

    // Demonstrate known values
    println!("\n4. Known Mathematical Values:");

    // dawsn(0) = 0
    println!("    dawsn(0) = {} (should be 0)", dawsn(0.0));

    // polygamma(0, 1) = -γ (Euler-Mascheroni constant)
    let euler_gamma = 0.5772156649015329;
    let psi0_1 = polygamma(0, 1.0);
    println!(
        "    polygamma(0, 1) = {:.10} (should be ≈ {:.10})",
        psi0_1, -euler_gamma
    );

    // polygamma(1, 1) = π²/6 (Basel problem)
    let pi_squared_over_6 = std::f64::consts::PI.powi(2) / 6.0;
    let psi1_1 = polygamma(1, 1.0);
    println!(
        "    polygamma(1, 1) = {:.10} (should be ≈ {:.10})",
        psi1_1, pi_squared_over_6
    );

    println!("\n=== Demo Complete ===");
    println!("These functions improve SciPy compatibility and provide better numerical stability.");

    Ok(())
}
