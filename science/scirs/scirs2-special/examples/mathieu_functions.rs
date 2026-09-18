use scirs2_special::{
    mathieu_a, mathieu_b, mathieu_cem, mathieu_even_coef, mathieu_odd_coef, mathieu_sem,
};
use std::f64::consts::PI;

#[allow(unused_variables)]
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mathieu Functions Example");
    println!("========================\n");

    // Example 1: Characteristic values for even and odd Mathieu functions
    println!("Characteristic Values:");
    println!("---------------------");

    println!("Even Mathieu Functions (a_m):");
    for m in 0..5 {
        let a0 = mathieu_a(m, 0.0)?;
        let a1 = mathieu_a(m, 1.0)?;
        let a5 = mathieu_a(m, 5.0)?;
        println!(
            "  m={}: q=0.0 => {:.6}, q=1.0 => {:.6}, q=5.0 => {:.6}",
            m, a0, a1, a5
        );
    }

    println!("\nOdd Mathieu Functions (b_m):");
    for m in 1..5 {
        let b0 = mathieu_b(m, 0.0)?;
        let b1 = mathieu_b(m, 1.0)?;
        let b5 = mathieu_b(m, 5.0)?;
        println!(
            "  m={}: q=0.0 => {:.6}, q=1.0 => {:.6}, q=5.0 => {:.6}",
            m, b0, b1, b5
        );
    }

    // Example 2: Fourier coefficients
    println!("\nFourier Coefficients:");
    println!("--------------------");

    println!("Even Mathieu Function Coefficients (m=2, q=1.0):");
    let even_coeffs = mathieu_even_coef(2, 1.0)?;
    for (i, coef) in even_coeffs.iter().take(5).enumerate() {
        println!("  A_2^({}) = {:.6}", 2 * i, coef);
    }

    println!("\nOdd Mathieu Function Coefficients (m=1, q=1.0):");
    let odd_coeffs = mathieu_odd_coef(1, 1.0)?;
    for (i, coef) in odd_coeffs.iter().take(5).enumerate() {
        println!("  B_1^({}) = {:.6}", 2 * i + 1, coef);
    }

    // Example 3: Evaluating Mathieu functions at different points
    println!("\nMathieu Function Values:");
    println!("----------------------");

    println!("Even Mathieu Function ce_2(x, q=1.0):");
    for i in 0..=4 {
        let x = i as f64 * PI / 4.0;
        let (ce, ce_prime) = mathieu_cem(2, 1.0, x)?;
        println!(
            "  x={:.3}π: ce_2={:.6}, ce_2'={:.6}",
            i as f64 / 4.0,
            ce,
            ce_prime
        );
    }

    println!("\nOdd Mathieu Function se_1(x, q=1.0):");
    for i in 0..=4 {
        let x = i as f64 * PI / 4.0;
        let (se, se_prime) = mathieu_sem(1, 1.0, x)?;
        println!(
            "  x={:.3}π: se_1={:.6}, se_1'={:.6}",
            i as f64 / 4.0,
            se,
            se_prime
        );
    }

    // Example 4: Verification of Mathieu differential equation
    // d²y/dx² + [a - 2q cos(2x)]y = 0
    println!("\nVerification of Mathieu Differential Equation:");
    println!("-------------------------------------------");

    let x = PI / 6.0;
    let q = 1.0;

    // For even function ce_2(x, q)
    let m = 2;
    let a = mathieu_a(m, q)?;
    let (ce, ce_prime) = mathieu_cem(m, q, x)?;

    // Verify the differential equation
    // We need the second derivative. We can approximate it:
    let h = 1e-6;
    let (ce_plus_, _) = mathieu_cem(m, q, x + h)?;
    let (ceminus_, _) = mathieu_cem(m, q, x - h)?;
    let ce_second_deriv = (ce_plus_ - 2.0 * ce + ceminus_) / (h * h);

    // Term of differential equation: d²y/dx² + [a - 2q cos(2x)]y = 0
    let diff_eq_term = ce_second_deriv + (a - 2.0 * q * (2.0 * x).cos()) * ce;

    println!("For ce_{}(x={:.3}π, q={})", m, x / PI, q);
    println!("  Characteristic value a = {:.6}", a);
    println!("  Function value ce = {:.6}", ce);
    println!(
        "  Left side of diff eq = {:.6} (should be near 0)",
        diff_eq_term
    );

    // For odd function se_1(x, q)
    let m = 1;
    let b = mathieu_b(m, q)?;
    let (se, se_prime) = mathieu_sem(m, q, x)?;

    // Verify the differential equation for se
    let (se_plus_, _) = mathieu_sem(m, q, x + h)?;
    let (seminus_, _) = mathieu_sem(m, q, x - h)?;
    let se_second_deriv = (se_plus_ - 2.0 * se + seminus_) / (h * h);

    // Term of differential equation: d²y/dx² + [b - 2q cos(2x)]y = 0
    let diff_eq_term = se_second_deriv + (b - 2.0 * q * (2.0 * x).cos()) * se;

    println!("\nFor se_{}(x={:.3}π, q={})", m, x / PI, q);
    println!("  Characteristic value b = {:.6}", b);
    println!("  Function value se = {:.6}", se);
    println!(
        "  Left side of diff eq = {:.6} (should be near 0)",
        diff_eq_term
    );

    Ok(())
}
