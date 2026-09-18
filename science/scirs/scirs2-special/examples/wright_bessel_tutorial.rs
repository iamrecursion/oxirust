//! # Wright Bessel Functions: Mathematical Deep Dive
//!
//! This tutorial explores Wright Bessel functions W_ρ,β(z), their mathematical
//! properties, computational challenges, and applications in fractional calculus
//! and stochastic processes.
//!
//! ## Learning Outcomes
//!
//! - Understand the definition and properties of Wright Bessel functions
//! - Learn about their relationship to Mittag-Leffler functions
//! - Explore applications in fractional differential equations
//! - Appreciate computational challenges and numerical strategies

use scirs2_special::{wright_bessel, wright_bessel_zeros};
use std::f64::consts::{E, PI};

/// Demonstrate basic properties of Wright Bessel functions
#[allow(dead_code)]
fn demonstrate_basic_properties() {
    println!("\n📚 PART 1: WRIGHT BESSEL FUNCTION FUNDAMENTALS");
    println!("══════════════════════════════════════════════════════════════");

    println!("\n🔹 1.1 Mathematical Definition");
    println!("The Wright Bessel function W_ρ,β(z) is defined by the series:");
    println!("  W_ρ,β(z) = Σ_{{k=0}}^∞ z^k / [k! Γ(ρk + β)]");
    println!("where ρ > -1, β ∈ ℝ, and z ∈ ℂ");

    println!("\n🔹 1.2 Special Cases");
    println!("Several important functions are special cases of Wright Bessel functions:");

    // Case 1: ρ = 0 (exponential function)
    let z = 1.0;
    match wright_bessel(0.0, 1.0, z) {
        Ok(w_val) => {
            let exp_val = E.powf(z);
            println!("  W_0,1({z}) = {w_val:.6} ≈ e^{z} = {exp_val:.6} (exponential)");
        }
        Err(_) => println!("  W_0,1({z}): computation failed"),
    }

    // Case 2: ρ = 1, β = 1 (Bessel function relationship)
    println!("  W_1,1(z) relates to modified Bessel functions");

    // Case 3: ρ = 1/2 (related to error function)
    match wright_bessel(0.5, 1.0, 0.5) {
        Ok(w_val) => {
            println!("  W_1/2,1(0.5) = {w_val:.6} (appears in fractional diffusion)");
        }
        Err(_) => println!("  W_1/2,1(0.5): computation failed"),
    }

    println!("\n🔹 1.3 Parameter Sensitivity");
    println!("Wright Bessel functions are sensitive to parameter changes:");

    let z_test = 1.0;
    let rho_values = [0.1, 0.5, 1.0, 1.5, 2.0];

    for &rho in &rho_values {
        match wright_bessel(rho, 1.0, z_test) {
            Ok(w_val) => {
                println!("  W_{rho:.1},1({z_test}) = {w_val:10.6}");
            }
            Err(_) => {
                println!("  W_{rho:.1},1({z_test}): computation failed");
            }
        }
    }
}

/// Explore asymptotic behavior
#[allow(dead_code)]
fn demonstrate_asymptotic_behavior() {
    println!("\n📚 PART 2: ASYMPTOTIC BEHAVIOR");
    println!("══════════════════════════════════════════════════════════════");

    println!("\n🔹 2.1 Large z Asymptotics");
    println!("For large |z|, Wright Bessel functions exhibit exponential behavior:");
    println!("  W_ρ,β(z) ~ z^((1-β)/ρ) exp(ρz^(1/ρ)) / (ρ√(2π))  [ρ > 0]");

    let rho = 0.5;
    let beta = 1.0;
    let large_z_values = [1.0, 2.0, 5.0, 10.0];

    println!("  Testing W_{rho},{beta}(z) for increasing z:");
    for &z in &large_z_values {
        match wright_bessel(rho, beta, z) {
            Ok(w_val) => {
                // Rough asymptotic approximation
                let asymptotic = z.powf((1.0 - beta) / rho) * (rho * z.powf(1.0 / rho)).exp()
                    / (rho * (2.0 * PI).sqrt());
                let ratio = w_val / asymptotic;
                println!("    z = {z:4.1}: W = {w_val:12.4e}, asymp = {asymptotic:12.4e}, ratio = {ratio:.3}");
            }
            Err(_) => {
                println!("    z = {z:4.1}: computation failed");
            }
        }
    }

    println!("\n🔹 2.2 Small z Behavior");
    println!("For small z, the series truncates to leading terms:");
    println!("  W_ρ,β(z) ≈ 1/Γ(β) + z/Γ(ρ+β) + z²/[2!Γ(2ρ+β)] + ...");

    let small_z_values = [0.01, 0.1, 0.5];
    let (rho, beta) = (1.0, 1.0);

    println!("  W_{rho},{beta}(z) for small z:");
    for &z in &small_z_values {
        match wright_bessel(rho, beta, z) {
            Ok(w_val) => {
                // Leading term approximation: 1/Γ(β) = 1/Γ(1) = 1
                let leading = 1.0;
                // Note: Next term correction would require gamma function
                // let correction = z / gamma(rho + beta);
                println!("    z = {z:4.2}: W = {w_val:.6}, leading term = {leading:.6}");
            }
            Err(_) => {
                println!("    z = {z:4.2}: computation failed");
            }
        }
    }
}

/// Demonstrate zeros and oscillatory behavior
#[allow(dead_code)]
fn demonstrate_zeros() {
    println!("\n📚 PART 3: ZEROS AND OSCILLATORY BEHAVIOR");
    println!("══════════════════════════════════════════════════════════════");

    println!("\n🔹 3.1 Finding Zeros");
    println!("Wright Bessel functions can have zeros for certain parameter ranges.");
    println!("Finding zeros is important for:");
    println!("  - Boundary value problems in fractional calculus");
    println!("  - Spectral analysis of fractional operators");
    println!("  - Studying oscillatory solutions of fractional PDEs");

    let rho_test = 0.8;
    let beta_test = 0.5;
    let num_zeros = 3;

    println!("  Searching for zeros of W_{rho_test},{beta_test}(z):");

    match wright_bessel_zeros(rho_test, beta_test, num_zeros) {
        Ok(zeros) => {
            for (i, &zero) in zeros.iter().enumerate() {
                println!("    Zero #{}: z = {zero:.6}", i + 1);

                // Verify it's actually a zero
                match wright_bessel(rho_test, beta_test, zero) {
                    Ok(w_at_zero) => {
                        println!("      Verification: W({zero:.6}) = {w_at_zero:.2e}");
                    }
                    Err(_) => {
                        println!("      Verification failed");
                    }
                }
            }
        }
        Err(_) => {
            println!("    Zero-finding failed for these parameters");
            println!("    Note: Not all parameter combinations have zeros!");
        }
    }

    println!("\n🔹 3.2 Oscillatory Behavior");
    println!("For certain parameters, Wright Bessel functions can oscillate:");

    let z_range: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let mut sign_changes = 0;
    let mut prev_sign = 1;

    println!("  Scanning W_{rho_test},{beta_test}(z) for sign changes:");
    for &z in &z_range {
        if z > 0.0 {
            match wright_bessel(rho_test, beta_test, z) {
                Ok(w_val) => {
                    let current_sign = if w_val > 0.0 { 1 } else { -1 };
                    if current_sign != prev_sign {
                        sign_changes += 1;
                        println!("    Sign change #{sign_changes} near z ≈ {z:.1}");
                    }
                    prev_sign = current_sign;

                    if z % 2.0 == 0.0 {
                        println!("    z = {z:4.1}: W = {w_val:10.6}");
                    }
                }
                Err(_) => {
                    println!("    z = {z:4.1}: computation failed");
                    break;
                }
            }
        }
    }

    println!("  Total sign changes detected: {sign_changes}");
}

/// Explore connections to other special functions
#[allow(dead_code)]
fn demonstrate_function_relationships() {
    println!("\n📚 PART 4: RELATIONSHIPS TO OTHER FUNCTIONS");
    println!("══════════════════════════════════════════════════════════════");

    println!("\n🔹 4.1 Mittag-Leffler Function Connection");
    println!("Wright Bessel functions are related to Mittag-Leffler functions:");
    println!("  E_α,β(z) = z^((1-β)/α) W_α/α,β/α(z^(1/α))");

    // Demonstrate this relationship for simple cases
    let alpha = 2.0_f64;
    let beta = 1.0_f64;
    let z = 1.0_f64;

    let rho = 1.0 / alpha;
    let beta_wright = beta / alpha;
    let z_wright = z.powf(1.0 / alpha);

    match wright_bessel(rho, beta_wright, z_wright) {
        Ok(w_val) => {
            let scaling = z.powf((1.0 - beta) / alpha);
            let mittag_leffler_approx = scaling * w_val;
            println!("  E_{alpha},{beta}({z}) ≈ {mittag_leffler_approx:.6} (via Wright Bessel)");
        }
        Err(_) => {
            println!("  Relationship computation failed");
        }
    }

    println!("\n🔹 4.2 Bessel Function Limits");
    println!("Certain limits recover classical Bessel functions:");

    // The limit ρ → 1, β = ν + 1 gives modified Bessel functions
    let nu = 0.0; // Order of Bessel function
    let z_bessel = 2.0;
    let rho_limit = 1.0;
    let beta_limit = nu + 1.0;

    match wright_bessel(rho_limit, beta_limit, z_bessel) {
        Ok(w_val) => {
            println!("  W_1,{beta_limit}({z_bessel}) = {w_val:.6}");
            println!("  (This should relate to I_{nu}({z_bessel}) for ν = {nu})");
        }
        Err(_) => {
            println!("  Bessel limit computation failed");
        }
    }

    println!("\n🔹 4.3 Hypergeometric Connections");
    println!("Wright Bessel functions can be expressed using hypergeometric functions:");
    println!("  W_ρ,β(z) = z^((1-β)/ρ) ₀F₁(; ρ+β; z/ρ) / Γ(β)  [for certain cases]");

    // This relationship is complex to verify numerically without hypergeometric functions
    println!("  (Detailed verification requires hypergeometric function implementation)");
}

/// Applications in fractional calculus
#[allow(dead_code)]
fn demonstrate_applications() {
    println!("\n📚 PART 5: APPLICATIONS IN FRACTIONAL CALCULUS");
    println!("══════════════════════════════════════════════════════════════");

    println!("\n🔹 5.1 Fractional Diffusion Equation");
    println!("The fundamental solution of the fractional diffusion equation:");
    println!("  ∂^α u/∂t^α = D ∂²u/∂x²,  0 < α < 1");
    println!("involves Wright Bessel functions.");

    let alpha = 0.5_f64; // Fractional order
    let diffusivity = 1.0_f64;
    let time = 1.0_f64;
    let space = 1.0_f64;

    // The fundamental solution involves W_(-α/2, 1-α/2, ...)
    let rho = -alpha / 2.0;
    let beta = 1.0 - alpha / 2.0;

    // Argument involves |x|/(D*t^α)^(1/2)
    let argument = space / (diffusivity * time.powf(alpha)).sqrt();

    println!("  Parameters: α = {alpha}, D = {diffusivity}, t = {time}, x = {space}");
    println!("  Wright function parameters: ρ = {rho:.3}, β = {beta:.3}");

    match wright_bessel(rho, beta, argument) {
        Ok(w_val) => {
            // The actual fundamental solution has additional scaling
            let scaling = 1.0 / (2.0 * (PI * diffusivity).sqrt() * time.powf(alpha / 2.0));
            let solution = scaling * w_val;
            println!("  u({space},{time}) ≈ {solution:.6} (scaled solution)");
        }
        Err(_) => {
            println!("  Solution computation failed (ρ < 0 case is challenging)");
        }
    }

    println!("\n🔹 5.2 Fractional Oscillator");
    println!("Solutions to fractional oscillator equations:");
    println!("  D^α x(t) + ω²x(t) = 0");
    println!("can be expressed using Wright Bessel functions.");

    let alpha_osc = 1.5_f64; // Fractional order > 1
    let omega = 1.0_f64; // Frequency
    let time_osc = 2.0_f64;

    // Solution involves W_(-α, 1, ...)
    let rho_osc = -alpha_osc;
    let beta_osc = 1.0;
    let arg_osc = -(omega * time_osc).powf(alpha_osc);

    println!("  Oscillator: α = {alpha_osc}, ω = {omega}, t = {time_osc}");

    if rho_osc > -1.0 {
        match wright_bessel(rho_osc, beta_osc, arg_osc) {
            Ok(w_val) => {
                println!("  x({time_osc}) ∝ {w_val:.6} (Wright Bessel contribution)");
            }
            Err(_) => {
                println!("  Oscillator solution computation failed");
            }
        }
    } else {
        println!("  Parameters outside valid range (ρ ≤ -1)");
    }

    println!("\n🔹 5.3 Probability and Stochastic Processes");
    println!("Wright Bessel functions appear in:");
    println!("  - Densities of stable subordinators");
    println!("  - Time-changed Brownian motion");
    println!("  - Fractional Poisson processes");

    let stability_param = 0.8_f64; // α ∈ (0,1] for stable processes
    let time_prob = 1.0_f64;

    // Stable subordinator density involves W_(-α, 1-α, ...)
    let rho_stable = -stability_param;
    let beta_stable = 1.0 - stability_param;
    let x_stable = 0.5; // Value where we evaluate density

    println!("  Stable subordinator with α = {stability_param}:");

    if rho_stable > -1.0 {
        match wright_bessel(
            rho_stable,
            beta_stable,
            -x_stable / time_prob.powf(stability_param),
        ) {
            Ok(w_val) => {
                let density = w_val / time_prob.powf(1.0 - stability_param);
                println!("    Density at x = {x_stable}, t = {time_prob}: {density:.6}");
            }
            Err(_) => {
                println!("    Density computation failed");
            }
        }
    } else {
        println!("    Parameters outside convergent range");
    }
}

/// Computational challenges and strategies
#[allow(dead_code)]
fn demonstrate_computational_aspects() {
    println!("\n📚 PART 6: COMPUTATIONAL CHALLENGES");
    println!("══════════════════════════════════════════════════════════════");

    println!("\n🔹 6.1 Series Convergence");
    println!(
        "The defining series W_ρ,β(z) = Σ z^k/[k!Γ(ρk+β)] has different convergence properties:"
    );

    let test_cases = [
        (0.5, 1.0, 1.0, "Standard case"),
        (2.0, 1.0, 1.0, "Fast convergence (ρ > 1)"),
        (0.1, 1.0, 1.0, "Slow convergence (small ρ)"),
        (1.0, 0.1, 1.0, "Small β case"),
    ];

    for &(rho, beta, z, description) in &test_cases {
        println!("  {description}: ρ = {rho}, β = {beta}, z = {z}");
        match wright_bessel(rho, beta, z) {
            Ok(w_val) => {
                println!("    Result: {w_val:.8}");
            }
            Err(_) => {
                println!("    Computation failed");
            }
        }
    }

    println!("\n🔹 6.2 Numerical Stability Issues");
    println!("Challenging parameter regimes:");

    let difficult_cases = [
        (-0.9, 1.0, 1.0, "ρ near -1"),
        (0.1, -0.5, 1.0, "Negative β"),
        (1.0, 1.0, 10.0, "Large argument"),
        (0.5, 1.0, -2.0, "Negative argument"),
    ];

    for &(rho, beta, z, description) in &difficult_cases {
        println!("  {description}: ρ = {rho}, β = {beta}, z = {z}");
        if rho > -1.0 {
            match wright_bessel(rho, beta, z) {
                Ok(w_val) => {
                    println!("    Result: {w_val:.6}");
                }
                Err(e) => {
                    println!("    Error: {e}");
                }
            }
        } else {
            println!("    Skipped: ρ ≤ -1 (outside definition)");
        }
    }

    println!("\n🔹 6.3 Implementation Strategies");
    println!("Our implementation uses:");
    println!("  📋 Series summation with careful term evaluation");
    println!("  📋 Gamma function calls via scirs2-special");
    println!("  📋 Convergence monitoring and early termination");
    println!("  📋 Overflow/underflow protection");

    println!("\n  For production use, consider:");
    println!("  🔧 Asymptotic expansions for large |z|");
    println!("  🔧 Continued fractions for negative arguments");
    println!("  🔧 Arbitrary precision arithmetic for extreme cases");
    println!("  🔧 Precomputed lookup tables for common parameter sets");
}

/// Main tutorial runner
#[allow(dead_code)]
fn main() {
    println!("🌟 WRIGHT BESSEL FUNCTIONS: COMPREHENSIVE TUTORIAL");
    println!("════════════════════════════════════════════════════════════════");
    println!("Exploring the mathematical theory and computational aspects of");
    println!("Wright Bessel functions W_ρ,β(z) and their applications.");

    // Add missing import note
    println!("\n⚠️  Note: Some computations may fail if gamma function is not properly imported.");
    println!("This tutorial demonstrates the interface and expected behavior.");

    demonstrate_basic_properties();
    demonstrate_asymptotic_behavior();
    demonstrate_zeros();
    demonstrate_function_relationships();
    demonstrate_applications();
    demonstrate_computational_aspects();

    println!("\n🎓 WRIGHT BESSEL TUTORIAL COMPLETE!");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Key takeaways:");
    println!("  ✅ Wright Bessel functions generalize exponentials and Bessel functions");
    println!("  ✅ They appear naturally in fractional calculus and stochastic processes");
    println!("  ✅ Computational implementation requires careful numerical strategies");
    println!("  ✅ Parameter ranges significantly affect convergence and stability");

    println!("\nFurther exploration:");
    println!("  📚 Study complex-valued arguments and branch cuts");
    println!("  📚 Implement asymptotic expansions for extreme parameters");
    println!("  📚 Explore matrix-valued generalizations");
    println!("  📚 Connect to fractional differential operators");

    println!("\n🎯 Ready to tackle fractional calculus problems! 🎯");
}
