//! # Spheroidal Wave Functions: Interactive Tutorial
//!
//! This example provides an educational walkthrough of spheroidal wave functions,
//! their mathematical properties, physical applications, and computational aspects.
//!
//! ## Learning Objectives
//!
//! After running this tutorial, you will understand:
//! - The mathematical origin of spheroidal wave functions
//! - The relationship between prolate and oblate spheroidal functions
//! - How characteristic values depend on the spheroidal parameter
//! - Computational challenges and numerical strategies
//! - Physical applications in scattering and wave propagation

use scirs2_special::{
    obl_ang1, obl_cv, obl_cv_seq, obl_rad1, pro_ang1, pro_cv, pro_cv_seq, pro_rad1,
};
use std::f64::consts::PI;

/// Educational demonstration of characteristic value behavior
#[allow(dead_code)]
fn demonstrate_characteristic_values() {
    println!("\n📚 PART 1: CHARACTERISTIC VALUES");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 1.1 Basic Properties");
    println!("The characteristic values λₘₙ(c) are eigenvalues of the spheroidal wave equation.");
    println!("They determine the separation constants in prolate/oblate spheroidal coordinates.\n");

    // Demonstrate the c=0 limit (Legendre functions)
    println!("✨ Special case c=0 (Legendre limit):");
    println!("When c=0, spheroidal functions reduce to associated Legendre functions");
    println!("and λₘₙ(0) = n(n+1)");

    for n in 0..=4 {
        let lambda = pro_cv(0, n, 0.0).expect("Operation failed");
        let expected = n as f64 * (n as f64 + 1.0);
        println!("  λ₀,{n}(0) = {lambda:.1} = {n}×{} (Legendre)", n + 1);
        assert!((lambda - expected).abs() < 1e-12);
    }

    println!("\n🔹 1.2 Parameter Dependence");
    println!("Watch how characteristic values change with spheroidal parameter c:");

    let c_values = [0.0, 0.5, 1.0, 2.0, 5.0];
    let (m, n) = (0, 2);

    println!("  For λ₀,₂(c):");
    for &c in &c_values {
        let lambda_pro = pro_cv(m, n, c).expect("Operation failed");
        let lambda_obl = obl_cv(m, n, c).expect("Operation failed");
        println!("    c = {c:4.1}: λ_prolate = {lambda_pro:8.4}, λ_oblate = {lambda_obl:8.4}");
    }

    println!("\n🔹 1.3 Ordering Property");
    println!("Characteristic values satisfy λₘₙ(c) < λₘ,ₙ₊₁(c) for fixed m:");

    let c = 1.0;
    let m = 1;
    let sequence = pro_cv_seq(m, m + 4, c).expect("Operation failed");

    for (i, &lambda) in sequence.iter().enumerate() {
        let n = m + i as i32;
        println!("  λ₁,{n}({c}) = {lambda:8.4}");

        if i > 0 {
            assert!(lambda > sequence[i - 1], "Ordering property violated!");
        }
    }
}

/// Demonstrate perturbation theory for small c
#[allow(dead_code)]
fn demonstrate_perturbation_theory() {
    println!("\n📚 PART 2: PERTURBATION THEORY");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 2.1 Small c Expansion");
    println!("For small c, we can expand: λₘₙ(c) ≈ n(n+1) + c²αₘₙ + c⁴βₘₙ + ...");

    let (m, n) = (0, 2);
    let small_c_values = [0.01, 0.05, 0.1, 0.2, 0.3];

    println!("  Testing first-order perturbation theory for λ₀,₂(c):");
    println!("  Exact formula: λ₀,₂(c) ≈ 6 + c²/(2×7) = 6 + c²/14");

    for &c in &small_c_values {
        let exact = pro_cv(m, n, c).expect("Operation failed");
        let perturbation = 6.0 + c * c / 14.0;
        let error = (exact - perturbation).abs();
        let relative_error = error / exact * 100.0;

        println!("    c = {c:4.2}: exact = {exact:8.5}, pert = {perturbation:8.5}, error = {relative_error:5.2}%");
    }

    println!("\n🔹 2.2 Convergence Radius");
    println!("Perturbation series has limited convergence radius. Let's test breakdown:");

    let c_test_values = [0.5, 1.0, 1.5, 2.0];
    for &c in &c_test_values {
        let exact = pro_cv(0, 1, c).expect("Operation failed");
        let perturbation = 2.0 + c * c / 10.0; // Simple approximation
        let relative_error = (exact - perturbation).abs() / exact * 100.0;

        println!("    c = {c:4.1}: relative error = {relative_error:6.2}%");
        if relative_error > 20.0 {
            println!("      ⚠️  Perturbation theory breaking down!");
        }
    }
}

/// Demonstrate asymptotic behavior for large c
#[allow(dead_code)]
fn demonstrate_asymptotic_behavior() {
    println!("\n📚 PART 3: ASYMPTOTIC BEHAVIOR");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 3.1 Large c Asymptotics");
    println!("For large |c|: λₘₙ(c) ≈ -c²/4 + (2n+1)c + n(n+1) - m²/2");

    let large_c_values = [5.0, 10.0, 20.0, 50.0];
    let (m, n) = (1, 3);

    println!("  Testing asymptotic formula for λ₁,₃(c):");

    for &c in &large_c_values {
        let exact = pro_cv(m, n, c).expect("Operation failed");
        let asymptotic = -c * c / 4.0 + (2.0 * n as f64 + 1.0) * c + n as f64 * (n as f64 + 1.0)
            - (m as f64).powi(2) / 2.0;
        let relative_error = (exact - asymptotic).abs() / exact.abs() * 100.0;

        println!("    c = {c:5.1}: exact = {exact:10.3}, asymp = {asymptotic:10.3}, error = {relative_error:5.2}%");
    }

    println!("\n🔹 3.2 Asymptotic Accuracy");
    println!("Asymptotic approximation improves with increasing |c|:");

    let c_range: Vec<f64> = (10..=100).step_by(10).map(|i| i as f64).collect();
    for &c in &c_range {
        let exact = pro_cv(0, 2, c).expect("Operation failed");
        let asymptotic = -c * c / 4.0 + 5.0 * c + 6.0;
        let relative_error = (exact - asymptotic).abs() / exact.abs() * 100.0;

        if c % 20.0 == 0.0 {
            println!("    c = {c:3.0}: relative error = {relative_error:8.4}%");
        }
    }
}

/// Demonstrate angular and radial functions
#[allow(dead_code)]
fn demonstrate_spheroidal_functions() {
    println!("\n📚 PART 4: SPHEROIDAL FUNCTIONS");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 4.1 Angular Functions");
    println!("Angular functions Sₘₙ(c,η) are solutions regular at η = ±1");

    let c = 1.0;
    let eta_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    println!("  S₀,₁({c},η) values:");
    for &eta in &eta_points {
        match pro_ang1(0, 1, c, eta) {
            Ok((s_val, s_prime)) => {
                println!("    η = {eta:5.1}: S = {s_val:8.4}, S' = {s_prime:8.4}");
            }
            Err(_) => {
                println!("    η = {eta:5.1}: Computation failed");
            }
        }
    }

    println!("\n🔹 4.2 Radial Functions");
    println!("Radial functions Rₘₙ(c,ξ) describe behavior outside focal region");

    let xi_points = [1.1, 1.5, 2.0, 3.0, 5.0];

    println!("  R₀,₁⁽¹⁾({c},ξ) values:");
    for &xi in &xi_points {
        match pro_rad1(0, 1, c, xi) {
            Ok((r_val, r_prime)) => {
                println!("    ξ = {xi:4.1}: R = {r_val:10.4}, R' = {r_prime:10.4}");
            }
            Err(_) => {
                println!("    ξ = {xi:4.1}: Computation failed");
            }
        }
    }

    println!("\n🔹 4.3 Exponential Growth");
    println!("Radial functions typically grow exponentially for large ξ:");

    let xi_large = [2.0, 4.0, 6.0];
    let c_test = 2.0;

    for &xi in &xi_large {
        match pro_rad1(0, 0, c_test, xi) {
            Ok((r_val_, _)) => {
                println!("    R₀,₀⁽¹⁾({c_test},ξ={xi}) = {r_val_:12.4e}");
            }
            Err(_) => {
                println!("    Computation failed for ξ = {xi}");
            }
        }
    }
}

/// Compare prolate vs oblate spheroidal functions
#[allow(dead_code)]
fn compare_prolate_oblate() {
    println!("\n📚 PART 5: PROLATE vs OBLATE COMPARISON");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 5.1 Characteristic Value Differences");
    println!("Prolate and oblate functions have different characteristic values:");

    let c_values = [0.5, 1.0, 2.0, 3.0];
    let (m, n) = (0, 2);

    for &c in &c_values {
        let lambda_pro = pro_cv(m, n, c).expect("Operation failed");
        let lambda_obl = obl_cv(m, n, c).expect("Operation failed");
        let difference = lambda_pro - lambda_obl;

        println!("    c = {c:4.1}: λ_prolate = {lambda_pro:8.4}, λ_oblate = {lambda_obl:8.4}, diff = {difference:8.4}");
    }

    println!("\n🔹 5.2 Angular Function Comparison");
    println!("Angular functions differ between prolate and oblate cases:");

    let c = 1.5;
    let eta = 0.5;

    match (pro_ang1(0, 1, c, eta), obl_ang1(0, 1, c, eta)) {
        (Ok((s_pro_, _)), Ok((s_obl_, _))) => {
            println!("    At η = {eta}, c = {c}:");
            println!("      Prolate S₀,₁({c},{eta}) = {s_pro_:8.4}");
            println!("      Oblate  S₀,₁({c},{eta}) = {s_obl_:8.4}");
            println!("      Ratio = {:.4}", s_pro_ / s_obl_);
        }
        _ => println!("    Computation failed for angular functions"),
    }
}

/// Demonstrate physical applications
#[allow(dead_code)]
fn demonstrate_physical_applications() {
    println!("\n📚 PART 6: PHYSICAL APPLICATIONS");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 6.1 Electromagnetic Scattering");
    println!("Example: Plane wave scattering by a prolate spheroid");
    println!("  - Semi-major axis a = 2.0 wavelengths");
    println!("  - Eccentricity e = 0.8 (highly elongated)");
    println!("  - Wave number k = 2π/λ = 1.0");

    let a = 2.0; // Semi-major axis in wavelengths
    let e = 0.8; // Eccentricity
    let k = 1.0; // Wave number
    let c = k * a * e; // Spheroidal parameter

    println!("  Spheroidal parameter c = ka×e = {c:.2}");

    // Calculate some characteristic values for electromagnetic modes
    println!("  Characteristic values for different modes:");
    for n in 1..=3 {
        for m in 0..=1 {
            if m <= n {
                match pro_cv(m, n, c) {
                    Ok(lambda) => {
                        println!("    λ{m},{n}({c:.2}) = {lambda:8.4} (affects TM{m},{n} and TE{m},{n} modes)");
                    }
                    Err(_) => {
                        println!("    λ{m},{n}({c:.2}) = computation failed");
                    }
                }
            }
        }
    }

    println!("\n🔹 6.2 Quantum Mechanical Interpretation");
    println!("In quantum mechanics, spheroidal functions appear in:");
    println!("  - Diatomic molecules (Born-Oppenheimer approximation)");
    println!("  - Electrons in spheroidal quantum dots");
    println!("  - Nuclear physics (deformed nuclei)");

    let energy_scale = 1.0; // Arbitrary energy units
    let confinement_strength = 0.5;
    let c_quantum = confinement_strength;

    println!("  Energy eigenvalues (λ represents -2mE/ℏ²):");
    for n in 0..=2 {
        match pro_cv(0, n, c_quantum) {
            Ok(lambda) => {
                let energy = -lambda * energy_scale / 2.0;
                println!("    State |{n},0⟩: E{n} = {energy:8.4} × (ℏ²/2ma²)");
            }
            Err(_) => {
                println!("    State |{n},0⟩: computation failed");
            }
        }
    }

    println!("\n🔹 6.3 Acoustic Applications");
    println!("Sound scattering by prolate/oblate objects:");
    println!("  - Frequency = 1000 Hz");
    println!("  - Sound speed = 343 m/s");
    println!("  - Object size ~ 10 cm");

    let frequency = 1000.0; // Hz
    let sound_speed = 343.0; // m/s
    let wavelength = sound_speed / frequency; // meters
    let objectsize = 0.1; // meters
    let ka_acoustic = 2.0 * PI * objectsize / wavelength;

    println!("  Acoustic size parameter ka = {ka_acoustic:.2}");

    if ka_acoustic < 1.0 {
        println!("  → Rayleigh scattering regime (ka << 1)");
    } else if ka_acoustic < 5.0 {
        println!("  → Intermediate scattering regime");
    } else {
        println!("  → Geometric scattering regime (ka >> 1)");
    }
}

/// Demonstrate numerical challenges and solutions
#[allow(dead_code)]
fn demonstrate_numerical_challenges() {
    println!("\n📚 PART 7: NUMERICAL CONSIDERATIONS");
    println!("═══════════════════════════════════════════════════════════");

    println!("\n🔹 7.1 Parameter Range Limitations");
    println!("Testing accuracy across different parameter ranges:");

    // Test small c regime
    let small_c_test = [1e-6, 1e-4, 1e-2, 0.1];
    println!("  Small c regime (perturbation theory):");
    for &c in &small_c_test {
        match pro_cv(0, 1, c) {
            Ok(lambda) => {
                let expected = 2.0 + c * c / 10.0; // First-order approximation
                let error = (lambda - expected).abs();
                println!("    c = {c:8.1e}: λ = {lambda:10.6}, error = {error:8.1e}");
            }
            Err(_) => {
                println!("    c = {c:8.1e}: computation failed");
            }
        }
    }

    // Test large c regime
    let large_c_test = [10.0, 50.0, 100.0, 200.0];
    println!("\n  Large c regime (asymptotic methods):");
    for &c in &large_c_test {
        match pro_cv(0, 1, c) {
            Ok(lambda) => {
                println!("    c = {c:6.1}: λ = {lambda:12.4}");
            }
            Err(_) => {
                println!("    c = {c:6.1}: computation failed");
            }
        }
    }

    println!("\n🔹 7.2 High Quantum Number Challenges");
    println!("Accuracy decreases for high quantum numbers:");

    let c_test = 2.0;
    for n in 1..=10 {
        match pro_cv(0, n, c_test) {
            Ok(lambda) => {
                println!("    n = {n:2}: λ₀,{n}({c_test}) = {lambda:10.4}");
            }
            Err(_) => {
                println!("    n = {n:2}: computation failed");
            }
        }
    }

    println!("\n🔹 7.3 Computational Strategy Summary");
    println!("Our implementation uses multiple algorithms:");
    println!("  📋 c = 0: Exact (Legendre functions)");
    println!("  📋 |c| < 1: Perturbation series");
    println!("  📋 1 ≤ |c| < 10: Continued fractions");
    println!("  📋 |c| ≥ 10: Asymptotic expansions + WKB");
    println!("  📋 Very large |c|: May require arbitrary precision");
}

/// Main tutorial function
#[allow(dead_code)]
fn main() {
    println!("🌟 SPHEROIDAL WAVE FUNCTIONS: COMPREHENSIVE TUTORIAL");
    println!("════════════════════════════════════════════════════════════════");
    println!("This tutorial explores the fascinating world of spheroidal wave functions,");
    println!("their mathematical properties, and their applications in physics and engineering.");

    // Run all tutorial sections
    demonstrate_characteristic_values();
    demonstrate_perturbation_theory();
    demonstrate_asymptotic_behavior();
    demonstrate_spheroidal_functions();
    compare_prolate_oblate();
    demonstrate_physical_applications();
    demonstrate_numerical_challenges();

    println!("\n🎓 TUTORIAL COMPLETE!");
    println!("═══════════════════════════════════════════════════════════");
    println!("You've learned about:");
    println!("  ✅ Mathematical foundations of spheroidal wave functions");
    println!("  ✅ Computational methods for different parameter regimes");
    println!("  ✅ Physical applications in electromagnetics, quantum mechanics, and acoustics");
    println!("  ✅ Numerical challenges and implementation strategies");
    println!("\nFor more advanced topics, consider:");
    println!("  📖 Implementing higher-order corrections");
    println!("  📖 Exploring complex-valued parameters");
    println!("  📖 Studying connection formulas between different regions");
    println!("  📖 Investigating computational optimization techniques");

    println!("\n🔬 Happy computing with spheroidal wave functions! 🔬");
}
