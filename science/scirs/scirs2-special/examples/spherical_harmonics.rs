use scirs2_core::Complex;
use scirs2_special::{sph_harm, sph_harm_complex};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spherical Harmonics Example");
    println!("===========================\n");

    // Real spherical harmonics examples
    println!("Real Spherical Harmonics (Y_l^m):");
    println!("--------------------------------");

    // Example 1: Y₀⁰ - spherically symmetric
    let y00: f64 = sph_harm(0, 0, PI / 2.0, 0.0)?;
    println!("Y₀⁰(π/2, 0) = {} (1/(2√π) ≈ {})", y00, 0.5 / f64::sqrt(PI));

    // Example 2: Y₁⁰ - varies with cos(θ)
    let y10_pole: f64 = sph_harm(1, 0, 0.0, 0.0)?;
    let y10_equator: f64 = sph_harm(1, 0, PI / 2.0, 0.0)?;
    let y10_south: f64 = sph_harm(1, 0, PI, 0.0)?;

    println!("Y₁⁰(0, 0) = {} (at north pole)", y10_pole);
    println!("Y₁⁰(π/2, 0) = {} (at equator)", y10_equator);
    println!("Y₁⁰(π, 0) = {} (at south pole)", y10_south);

    // Example 3: Y₂⁰ - second degree, varies with 3cos²(θ)-1
    let y20_pole: f64 = sph_harm(2, 0, 0.0, 0.0)?;
    let y20_mid: f64 = sph_harm(2, 0, PI / 4.0, 0.0)?;
    let y20_equator: f64 = sph_harm(2, 0, PI / 2.0, 0.0)?;

    println!("Y₂⁰(0, 0) = {} (at north pole)", y20_pole);
    println!("Y₂⁰(π/4, 0) = {} (at mid-latitude)", y20_mid);
    println!("Y₂⁰(π/2, 0) = {} (at equator)", y20_equator);

    // Example 4: Y₁¹ - varies with sin(θ)cos(φ)
    let y11_equator0: f64 = sph_harm(1, 1, PI / 2.0, 0.0)?;
    let y11_equator90: f64 = sph_harm(1, 1, PI / 2.0, PI / 2.0)?;

    println!("Y₁¹(π/2, 0) = {} (equator, φ=0)", y11_equator0);
    println!("Y₁¹(π/2, π/2) = {} (equator, φ=π/2)", y11_equator90);

    println!();

    // Complex spherical harmonics examples
    println!("Complex Spherical Harmonics (Y_l^m):");
    println!("-----------------------------------");

    // Example 1: Y₀⁰ - spherically symmetric, purely real
    let (y00_re, y00_im): (f64, f64) = sph_harm_complex(0, 0, PI / 2.0, 0.0)?;
    println!("Y₀⁰(π/2, 0) = {} + {}i", y00_re, y00_im);

    // Example 2: Y₁¹ - has both real and imaginary parts
    let (y11_re1, y11_im1): (f64, f64) = sph_harm_complex(1, 1, PI / 2.0, 0.0)?;
    let (y11_re2, y11_im2): (f64, f64) = sph_harm_complex(1, 1, PI / 2.0, PI / 4.0)?;

    println!("Y₁¹(π/2, 0) = {} + {}i", y11_re1, y11_im1);
    println!("Y₁¹(π/2, π/4) = {} + {}i", y11_re2, y11_im2);

    // Example 3: Y₁⁻¹ - negative m value
    let (y1n1_re1, y1n1_im1): (f64, f64) = sph_harm_complex(1, -1, PI / 2.0, 0.0)?;
    let (y1n1_re2, y1n1_im2): (f64, f64) = sph_harm_complex(1, -1, PI / 2.0, PI / 4.0)?;

    println!("Y₁⁻¹(π/2, 0) = {} + {}i", y1n1_re1, y1n1_im1);
    println!("Y₁⁻¹(π/2, π/4) = {} + {}i", y1n1_re2, y1n1_im2);

    // Example 4: Y₂² - higher degree and order
    let (y22_re, y22_im): (f64, f64) = sph_harm_complex(2, 2, PI / 4.0, PI / 3.0)?;
    println!("Y₂²(π/4, π/3) = {} + {}i", y22_re, y22_im);

    println!();

    // Demonstration of symmetry relations
    println!("Symmetry Properties:");
    println!("------------------");

    // Relation between Y_l^m and Y_l^(-m)
    let (y21_re, y21_im): (f64, f64) = sph_harm_complex(2, 1, PI / 3.0, PI / 5.0)?;
    let (y2n1_re, y2n1_im): (f64, f64) = sph_harm_complex(2, -1, PI / 3.0, PI / 5.0)?;

    println!("Y₂¹(π/3, π/5) = {} + {}i", y21_re, y21_im);
    println!("Y₂⁻¹(π/3, π/5) = {} + {}i", y2n1_re, y2n1_im);
    println!("Note: Y_l^(-m)(θ,φ) = (-1)^m [Y_l^m(θ,φ)]* (complex conjugate relation)");

    // Create complex numbers for the verification
    let y21 = Complex::new(y21_re, y21_im);
    let y2n1 = Complex::new(y2n1_re, y2n1_im);

    let expected_relation = (-1.0_f64).powi(1) * y21.conj();
    let relation_diff = (expected_relation - y2n1).norm();

    println!(
        "Verification: |(-1)^m [Y₂¹]* - Y₂⁻¹| = {:.2e}",
        relation_diff
    );

    Ok(())
}
