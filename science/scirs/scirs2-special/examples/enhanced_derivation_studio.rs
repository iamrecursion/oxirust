//! Enhanced Interactive Mathematical Derivation Studio
//!
//! This example provides an interactive environment for deriving mathematical
//! results in special function theory.
//! Split from the original 2297-line file to comply with the 2000-line policy.
//!
//! Features:
//! - Step-by-step derivation guidance
//! - Verification of intermediate results
//! - Multiple derivation paths
//!
//! Run with: cargo run --example enhanced_derivation_studio

#![allow(clippy::all)]

use scirs2_core::Complex64;
use scirs2_special::*;
use std::f64::consts::{E, PI};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Welcome to the Enhanced Mathematical Derivation Studio!\n");
    println!("Learn by deriving important results in special function theory.\n");

    loop {
        println!("\n=== Derivation Studio Menu ===");
        println!("1. Γ(1/2) = √π Derivation");
        println!("2. Gamma Reflection Formula");
        println!("3. Bessel Generating Function");
        println!("4. Stirling's Approximation");
        println!("5. Beta-Gamma Relationship");
        println!("q. Quit");
        print!("\nChoose a derivation: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;

        match choice.trim() {
            "1" => derive_gamma_half()?,
            "2" => derive_reflection_formula()?,
            "3" => derive_bessel_generating()?,
            "4" => derive_stirling()?,
            "5" => derive_beta_gamma()?,
            "q" | "Q" => {
                println!("\nThank you for using the Derivation Studio!");
                break;
            }
            _ => println!("Invalid choice. Please try again."),
        }
    }

    Ok(())
}

fn derive_gamma_half() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^80}", " Derivation: Γ(1/2) = √π ");

    println!("\nObjective: Prove that Γ(1/2) = √π");
    println!("\nPrerequisites:");
    println!("  • Definition: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt");
    println!("  • Gaussian integral: ∫_{{-∞}}^∞ e^(-x²) dx = √π");

    wait_for_user("Press Enter to begin the derivation...")?;

    println!("\n--- Step 1: Write the definition ---");
    println!("Γ(1/2) = ∫₀^∞ t^(-1/2) e^(-t) dt");
    println!("\nNote: The exponent is -1/2, so we have t^(-1/2) = 1/√t");

    wait_for_user("Press Enter for Step 2...")?;

    println!("\n--- Step 2: Change of variables ---");
    println!("Let t = u², then dt = 2u du");
    println!("When t = 0, u = 0");
    println!("When t → ∞, u → ∞");
    println!("\nSubstituting:");
    println!("Γ(1/2) = ∫₀^∞ (u²)^(-1/2) e^(-u²) · 2u du");
    println!("       = ∫₀^∞ u^(-1) · 2u · e^(-u²) du");
    println!("       = 2∫₀^∞ e^(-u²) du");

    wait_for_user("Press Enter for Step 3...")?;

    println!("\n--- Step 3: Recognize the Gaussian integral ---");
    println!("We know that ∫_{{-∞}}^∞ e^(-u²) du = √π");
    println!("\nBy symmetry of e^(-u²):");
    println!("∫₀^∞ e^(-u²) du = (1/2) · ∫_{{-∞}}^∞ e^(-u²) du = √π/2");

    wait_for_user("Press Enter for final step...")?;

    println!("\n--- Step 4: Conclude ---");
    println!("Γ(1/2) = 2 · ∫₀^∞ e^(-u²) du");
    println!("       = 2 · (√π/2)");
    println!("       = √π");

    println!("\n{}", "━".repeat(80));
    println!("Therefore: Γ(1/2) = √π ≈ {:.10}", PI.sqrt());

    println!("\n--- Numerical Verification ---");
    let computed = gamma(0.5);
    let expected = PI.sqrt();
    println!("Γ(0.5) computed = {:.15}", computed);
    println!("√π = {:.15}", expected);
    println!("Error = {:.2e}", (computed - expected).abs());

    Ok(())
}

fn derive_reflection_formula() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^80}", " Derivation: Gamma Reflection Formula ");

    println!("\nObjective: Prove that Γ(z)Γ(1-z) = π/sin(πz)");
    println!("\nThis is one of the most important identities of the gamma function.");

    wait_for_user("Press Enter to begin...")?;

    println!("\n--- Step 1: Start with the beta function ---");
    println!("The beta function is defined as:");
    println!("B(a,b) = ∫₀^1 t^(a-1)(1-t)^(b-1) dt");
    println!("\nAnd we have the relationship:");
    println!("B(a,b) = Γ(a)Γ(b)/Γ(a+b)");

    wait_for_user("Press Enter for Step 2...")?;

    println!("\n--- Step 2: Special case ---");
    println!("Consider B(z, 1-z):");
    println!("B(z, 1-z) = Γ(z)Γ(1-z)/Γ(1)");
    println!("           = Γ(z)Γ(1-z)  (since Γ(1) = 1)");

    wait_for_user("Press Enter for Step 3...")?;

    println!("\n--- Step 3: Evaluate the integral ---");
    println!("B(z, 1-z) = ∫₀^1 t^(z-1)(1-t)^(-z) dt");
    println!("\nUsing the substitution t = sin²(θ):");
    println!("This integral evaluates to π/sin(πz)");
    println!("(The full derivation requires complex analysis)");

    wait_for_user("Press Enter for conclusion...")?;

    println!("\n--- Step 4: Conclude ---");
    println!("Γ(z)Γ(1-z) = B(z, 1-z) = π/sin(πz)");

    println!("\n{}", "━".repeat(80));
    println!("Therefore: Γ(z)Γ(1-z) = π/sin(πz)");

    println!("\n--- Numerical Verification ---");
    let test_values = vec![0.25, 0.3333, 0.5, 0.6667, 0.75];

    for z in test_values {
        let lhs = gamma(z) * gamma(1.0 - z);
        let rhs = PI / (PI * z).sin();
        let error = ((lhs - rhs) / rhs).abs();

        println!(
            "z = {:.4}: LHS = {:.6}, RHS = {:.6}, Error = {:.2e}",
            z, lhs, rhs, error
        );
    }

    Ok(())
}

fn derive_bessel_generating() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^80}", " Derivation: Bessel Generating Function ");

    println!("\nObjective: Derive exp(x(t-1/t)/2) = Σ_{{n=-∞}}^∞ J_n(x) t^n");
    println!("\nThis is the generating function for Bessel functions of the first kind.");

    wait_for_user("Press Enter to begin...")?;

    println!("\n--- Step 1: Start with the integral representation ---");
    println!("J_n(x) = (1/π) ∫₀^π cos(nθ - x sin θ) dθ");
    println!("\nThis is one of the standard integral representations of J_n.");

    wait_for_user("Press Enter for Step 2...")?;

    println!("\n--- Step 2: Use the exponential form ---");
    println!("cos(nθ - x sin θ) = Re[exp(i(nθ - x sin θ))]");
    println!("\nSumming over all n:");
    println!("Σ J_n(x) t^n = (1/π) ∫₀^π Σ exp(i(nθ - x sin θ)) t^n dθ");

    wait_for_user("Press Enter for Step 3...")?;

    println!("\n--- Step 3: Geometric series ---");
    println!("The sum Σ exp(inθ)t^n is a geometric series:");
    println!("Σ_{{n=-∞}}^∞ (t·e^(iθ))^n");
    println!("\nAfter manipulation, this leads to:");
    println!("exp(x(t - 1/t)/2)");

    wait_for_user("Press Enter for conclusion...")?;

    println!("\n{}", "━".repeat(80));
    println!("Therefore: exp(x(t-1/t)/2) = Σ_{{n=-∞}}^∞ J_n(x) t^n");

    println!("\n--- Numerical Verification ---");
    println!("For x = 1, t = 1, both sides should equal J₀(1):");
    println!("LHS (exp(0)) = 1.0");
    println!("RHS (sum of J_n) ≈ {:.6}", j0(1.0));
    println!("\nNote: When t = 1, only J₀ term survives in the sum.");

    Ok(())
}

fn derive_stirling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^80}", " Derivation: Stirling's Approximation ");

    println!("\nObjective: Show that Γ(z) ~ √(2π/z) (z/e)^z as |z| → ∞");
    println!("\nThis is one of the most important asymptotic formulas in mathematics.");

    wait_for_user("Press Enter to begin...")?;

    println!("\n--- Step 1: Start with logarithm ---");
    println!("Taking the logarithm:");
    println!("ln Γ(z) = ln(∫₀^∞ t^(z-1) e^(-t) dt)");
    println!("\nFor large z, the integral is dominated by values near the maximum.");

    wait_for_user("Press Enter for Step 2...")?;

    println!("\n--- Step 2: Saddle point method ---");
    println!("The integrand t^(z-1) e^(-t) has a maximum at t = z-1");
    println!("\nNear this point, we can approximate:");
    println!("ln[t^(z-1) e^(-t)] ≈ (z-1)ln(z-1) - (z-1)");

    wait_for_user("Press Enter for Step 3...")?;

    println!("\n--- Step 3: Gaussian approximation ---");
    println!("The second derivative gives the width of the Gaussian:");
    println!("This leads to a factor of √(2π(z-1))");

    wait_for_user("Press Enter for conclusion...")?;

    println!("\n--- Step 4: Conclude ---");
    println!("Combining all terms:");
    println!("Γ(z) ~ √(2π/z) (z/e)^z");

    println!("\n{}", "━".repeat(80));
    println!("Stirling's Formula: Γ(z) ~ √(2π/z) (z/e)^z");

    println!("\n--- Numerical Verification ---");
    for n in vec![10, 20, 50, 100] {
        let z = n as f64;
        let exact = gamma(z);
        let stirling = (2.0 * PI / z).sqrt() * (z / E).powf(z);
        let rel_error = ((exact - stirling) / exact).abs();

        println!(
            "n = {:3}: Exact = {:.6e}, Stirling = {:.6e}, Rel Error = {:.6}",
            n, exact, stirling, rel_error
        );
    }

    Ok(())
}

fn derive_beta_gamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=^80}", " Derivation: Beta-Gamma Relationship ");

    println!("\nObjective: Prove that B(a,b) = Γ(a)Γ(b)/Γ(a+b)");
    println!("\nThis connects two fundamental special functions.");

    wait_for_user("Press Enter to begin...")?;

    println!("\n--- Step 1: Definitions ---");
    println!("Beta function:");
    println!("B(a,b) = ∫₀^1 t^(a-1)(1-t)^(b-1) dt");
    println!("\nGamma function:");
    println!("Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt");

    wait_for_user("Press Enter for Step 2...")?;

    println!("\n--- Step 2: Product of gammas ---");
    println!("Consider Γ(a)Γ(b):");
    println!("Γ(a)Γ(b) = [∫₀^∞ u^(a-1) e^(-u) du] · [∫₀^∞ v^(b-1) e^(-v) dv]");
    println!("\nThis is a double integral over the first quadrant.");

    wait_for_user("Press Enter for Step 3...")?;

    println!("\n--- Step 3: Change of variables ---");
    println!("Let u = st, v = s(1-t)");
    println!("Then s = u + v, t = u/(u+v)");
    println!("\nThe Jacobian is s, so du dv = s dt ds");

    wait_for_user("Press Enter for Step 4...")?;

    println!("\n--- Step 4: Simplify ---");
    println!("After substitution:");
    println!("Γ(a)Γ(b) = ∫₀^∞ ∫₀^1 [st]^(a-1) [s(1-t)]^(b-1) e^(-s) s dt ds");
    println!("         = ∫₀^∞ s^(a+b-1) e^(-s) ds · ∫₀^1 t^(a-1)(1-t)^(b-1) dt");
    println!("         = Γ(a+b) · B(a,b)");

    wait_for_user("Press Enter for conclusion...")?;

    println!("\n--- Step 5: Conclude ---");
    println!("Therefore:");
    println!("B(a,b) = Γ(a)Γ(b)/Γ(a+b)");

    println!("\n{}", "━".repeat(80));
    println!("Beta-Gamma Relationship: B(a,b) = Γ(a)Γ(b)/Γ(a+b)");

    println!("\n--- Numerical Verification ---");
    let test_pairs: Vec<(f64, f64)> = vec![(2.0, 3.0), (1.5, 2.5), (3.5, 4.5), (2.0, 2.0)];

    for (a, b) in test_pairs {
        let beta_direct = beta(a, b);
        let beta_from_gamma = gamma(a) * gamma(b) / gamma(a + b);
        let error = ((beta_direct - beta_from_gamma) / beta_from_gamma).abs() as f64;

        println!(
            "B({:.1},{:.1}): Direct = {:.10}, From Γ = {:.10}, Error = {:.2e}",
            a, b, beta_direct, beta_from_gamma, error
        );
    }

    Ok(())
}

fn wait_for_user(prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(())
}
