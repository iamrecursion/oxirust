//! Simplified Comprehensive Interactive Tutorial for Special Functions
//!
//! This example provides an interactive tutorial for learning special functions.
//! Split from the original 2927-line file to comply with the 2000-line policy.
//!
//! Features:
//! - Interactive gamma function exploration
//! - Step-by-step learning with feedback
//! - Real-time computation and verification
//!
//! Run with: cargo run --example comprehensive_interactive_tutorial

use scirs2_special::*;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Welcome to the Comprehensive Interactive Tutorial for Special Functions!\n");
    println!("Note: This is a simplified version. The full tutorial system has been");
    println!("refactored into multiple focused example files.\n");

    loop {
        println!("\n=== Main Menu ===");
        println!("1. Gamma Function Tutorial");
        println!("2. Bessel Function Tutorial");
        println!("3. Error Function Tutorial");
        println!("4. Quick Examples");
        println!("q. Quit");
        print!("\nChoose an option: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;

        match choice.trim() {
            "1" => gamma_tutorial()?,
            "2" => bessel_tutorial()?,
            "3" => error_function_tutorial()?,
            "4" => quick_examples()?,
            "q" | "Q" => {
                println!("\nThank you for using the tutorial!");
                break;
            }
            _ => println!("Invalid choice. Please try again."),
        }
    }

    Ok(())
}

fn gamma_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Gamma Function Tutorial ===");
    println!("\nThe gamma function Γ(z) is defined as:");
    println!("  Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt  for Re(z) > 0");
    println!("\nKey property: Γ(n) = (n-1)! for positive integers n");
    println!("Special value: Γ(1/2) = √π ≈ 1.772454");

    println!("\nLet's explore some values:");
    for n in 1..=5 {
        let val = gamma(n as f64);
        println!("  Γ({}) = {:.6} = {}!", n, val, (n - 1));
    }

    println!("\nComplex gamma function:");
    use scirs2_core::Complex64;
    let z = Complex64::new(1.5, 0.5);
    let result = gamma_complex(z);
    println!("  Γ(1.5 + 0.5i) = {:.6}", result);

    println!("\n--- Interactive Exercise ---");
    println!("What is Γ(4)?");
    println!("Hint: Γ(n) = (n-1)! for positive integers");
    print!("Your answer: ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    match answer.trim().parse::<f64>() {
        Ok(val) if (val - 6.0).abs() < 0.01 => {
            println!("Correct! Γ(4) = 3! = 6");
        }
        Ok(val) => {
            println!("Incorrect. You answered {:.2}, but Γ(4) = 3! = 6", val);
            println!("Remember: Γ(n) = (n-1)!, not n!");
        }
        Err(_) => println!("Invalid input. The correct answer is 6."),
    }

    Ok(())
}

fn bessel_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Bessel Function Tutorial ===");
    println!("\nBessel functions are solutions to Bessel's differential equation:");
    println!("  x²y'' + xy' + (x² - ν²)y = 0");
    println!("\nTypes:");
    println!("  J_ν(x): Bessel functions of the first kind");
    println!("  Y_ν(x): Bessel functions of the second kind");

    println!("\nLet's compute some values:");
    let x = 5.0;
    println!("For x = {}:", x);
    println!("  J₀({}) = {:.6}", x, j0(x));
    println!("  J₁({}) = {:.6}", x, j1(x));
    println!("  Y₀({}) = {:.6}", x, y0(x));
    println!("  Y₁({}) = {:.6}", x, y1(x));

    println!("\nZeros of J₀(x) (important in physics):");
    for i in 1..=5 {
        if let Ok(zero) = j0_zeros::<f64>(i) {
            println!("  α₀,{} = {:.6}", i, zero);
        }
    }

    println!("\n--- Interactive Exercise ---");
    println!("Bessel functions appear in many physical applications.");
    println!("For a circular drumhead, which function describes the radial vibration?");
    println!("a) J_n(x)");
    println!("b) Y_n(x)");
    println!("c) I_n(x)");
    print!("Your answer (a/b/c): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "a" {
        println!("Correct! J_n(x) describes the radial component of vibration.");
        println!("The zeros of J_n determine the resonant frequencies.");
    } else {
        println!("Incorrect. The correct answer is a) J_n(x).");
        println!("J_n is regular at the origin, unlike Y_n which is singular there.");
    }

    Ok(())
}

fn error_function_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Error Function Tutorial ===");
    println!("\nThe error function is defined as:");
    println!("  erf(x) = (2/√π) ∫₀^x e^(-t²) dt");
    println!("\nProperties:");
    println!("  • erf(0) = 0");
    println!("  • erf(∞) = 1");
    println!("  • erf(-x) = -erf(x)  (odd function)");
    println!("  • erfc(x) = 1 - erf(x)  (complementary error function)");

    println!("\nLet's explore some values:");
    let test_values = vec![0.0, 0.5, 1.0, 2.0, 3.0];
    for x in test_values {
        println!(
            "  erf({:.1}) = {:.6}, erfc({:.1}) = {:.6}",
            x,
            erf(x),
            x,
            erfc(x)
        );
    }

    println!("\nApplication: Normal Distribution");
    println!("The cumulative distribution function (CDF) of N(0,1) is:");
    println!("  Φ(x) = (1/2)[1 + erf(x/√2)]");

    let x = 1.0;
    let cdf = 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
    println!("  Φ({}) = {:.6}", x, cdf);
    println!("  (Probability that a standard normal variable ≤ {})", x);

    println!("\n--- Interactive Exercise ---");
    println!("As x → ∞, erf(x) approaches:");
    println!("a) 0");
    println!("b) 1");
    println!("c) π/2");
    println!("d) ∞");
    print!("Your answer (a/b/c/d): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "b" {
        println!("Correct! erf(∞) = 1");
        println!("The integral ∫₀^∞ e^(-t²) dt = √π/2, so (2/√π)·(√π/2) = 1");
    } else {
        println!("Incorrect. The correct answer is b) 1.");
        println!("The error function asymptotically approaches 1 as x → ∞.");
    }

    Ok(())
}

fn quick_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Quick Examples of Special Functions ===");

    println!("\n1. Gamma Function:");
    println!("   Γ(5.5) = {:.6}", gamma(5.5));
    println!("   ln Γ(100) = {:.6}", gammaln(100.0));

    println!("\n2. Beta Function:");
    println!("   B(2, 3) = {:.6}", beta(2.0, 3.0));
    println!("   (Relation: B(a,b) = Γ(a)Γ(b)/Γ(a+b))");

    println!("\n3. Bessel Functions:");
    println!("   J₀(1) = {:.6}", j0(1.0));
    println!("   J₁(1) = {:.6}", j1(1.0));
    let jv_val = jv::<f64>(2.5, 1.0);
    println!("   J_2.5(1) = {:.6}", jv_val);

    println!("\n4. Error Functions:");
    println!("   erf(1) = {:.6}", erf(1.0));
    println!("   erfc(1) = {:.6}", erfc(1.0));

    println!("\n5. Airy Functions:");
    let (ai, aip, bi, bip) = airye(1.0);
    println!("   Ai(1) = {:.6}, Ai'(1) = {:.6}", ai, aip);
    println!("   Bi(1) = {:.6}, Bi'(1) = {:.6}", bi, bip);

    println!("\n6. Elliptic Integrals:");
    let k_val = ellipk(0.5);
    println!("   K(0.5) = {:.6}", k_val);
    let e_val = ellipe(0.5);
    println!("   E(0.5) = {:.6}", e_val);

    println!("\n7. Zeta Function:");
    if let Ok(zeta_val) = zeta(2.0) {
        println!("   ζ(2) = {:.6} ≈ π²/6", zeta_val);
    }

    println!("\n8. Spherical Bessel Functions:");
    let j0_sph = spherical_jn(0, 1.0);
    println!("   j₀(1) = {:.6}", j0_sph);
    let j1_sph = spherical_jn(1, 1.0);
    println!("   j₁(1) = {:.6}", j1_sph);

    println!("\nPress Enter to continue...");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(())
}
