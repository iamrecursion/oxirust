//! Interactive Mathematics Laboratory for Special Functions
//!
//! This example provides a laboratory environment for exploring special functions.
//! Split from the original 2704-line file to comply with the 2000-line policy.
//!
//! Features:
//! - Mathematical expression evaluation
//! - Function exploration and visualization
//! - Computational experiments
//!
//! Run with: cargo run --example interactive_math_laboratory

#![allow(clippy::all)]

use scirs2_special::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Welcome to the Interactive Mathematics Laboratory!\n");
    println!("Explore special functions through computation and analysis.\n");

    let mut workspace: HashMap<String, f64> = HashMap::new();
    workspace.insert("pi".to_string(), PI);
    workspace.insert("e".to_string(), std::f64::consts::E);
    workspace.insert("sqrt_pi".to_string(), PI.sqrt());

    loop {
        println!("\n=== Math Laboratory Menu ===");
        println!("1. Function Explorer");
        println!("2. Theorem Verifier");
        println!("3. Numerical Experiments");
        println!("4. Function Relationships");
        println!("5. View Workspace Variables");
        println!("q. Quit");
        print!("\nChoose an option: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;

        match choice.trim() {
            "1" => function_explorer()?,
            "2" => theorem_verifier()?,
            "3" => numerical_experiments()?,
            "4" => function_relationships()?,
            "5" => view_workspace(&workspace),
            "q" | "Q" => {
                println!("\nClosing laboratory session. Goodbye!");
                break;
            }
            _ => println!("Invalid choice. Please try again."),
        }
    }

    Ok(())
}

fn function_explorer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Function Explorer ===");
    println!("Choose a function family to explore:");
    println!("1. Gamma and Related Functions");
    println!("2. Bessel Functions");
    println!("3. Error Functions");
    println!("4. Orthogonal Polynomials");
    print!("\nChoice: ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;

    match choice.trim() {
        "1" => explore_gamma()?,
        "2" => explore_bessel()?,
        "3" => explore_error_functions()?,
        "4" => explore_orthogonal_polynomials()?,
        _ => println!("Invalid choice."),
    }

    Ok(())
}

fn explore_gamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Gamma Function Explorer ---");
    print!("Enter a value to evaluate Γ(x): ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    match input.trim().parse::<f64>() {
        Ok(x) if x > 0.0 => {
            let result = gamma(x);
            println!("Γ({}) = {:.10}", x, result);
            println!("ln Γ({}) = {:.10}", x, gammaln(x));

            if x.fract() == 0.0 && x >= 1.0 && x <= 10.0 {
                let factorial = (1..x as i32).product::<i32>();
                println!("Verification: ({}−1)! = {}", x, factorial);
            }

            println!("\nRelated values:");
            println!("Γ({} + 1) = {:.10}", x, gamma(x + 1.0));
            println!(
                "Functional equation check: {}·Γ({}) = {:.10}",
                x,
                x,
                x * gamma(x)
            );
        }
        Ok(_) => println!("Error: Gamma function requires positive input"),
        Err(_) => println!("Error: Invalid input"),
    }

    Ok(())
}

fn explore_bessel() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Bessel Function Explorer ---");
    print!("Enter x value: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    match input.trim().parse::<f64>() {
        Ok(x) => {
            println!("\nBessel functions at x = {}:", x);
            println!("J₀({}) = {:.10}", x, j0(x));
            println!("J₁({}) = {:.10}", x, j1(x));
            println!("Y₀({}) = {:.10}", x, y0(x));
            println!("Y₁({}) = {:.10}", x, y1(x));

            println!("\nRecurrence relation check:");
            println!("J₁ should equal (J₀ + J₂)/2 at this point");

            let j2 = jv::<f64>(2.0, x);
            let recurrence = (j0(x) + j2) / 2.0;
            println!("From recurrence: {:.10}", recurrence);
            println!("Direct J₁: {:.10}", j1(x));
            println!("Difference: {:.2e}", (recurrence - j1(x)).abs());
        }
        Err(_) => println!("Error: Invalid input"),
    }

    Ok(())
}

fn explore_error_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Error Function Explorer ---");
    print!("Enter x value: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    match input.trim().parse::<f64>() {
        Ok(x) => {
            println!("\nError functions at x = {}:", x);
            println!("erf({}) = {:.10}", x, erf(x));
            println!("erfc({}) = {:.10}", x, erfc(x));
            println!("Sum: erf(x) + erfc(x) = {:.10}", erf(x) + erfc(x));
            println!("(Should be 1.0)");

            println!("\nSymmetry check:");
            println!("erf(-{}) = {:.10}", x, erf(-x));
            println!("-erf({}) = {:.10}", x, -erf(x));
            println!("(Should be equal for odd function)");
        }
        Err(_) => println!("Error: Invalid input"),
    }

    Ok(())
}

fn explore_orthogonal_polynomials() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Orthogonal Polynomials Explorer ---");
    print!("Enter n (polynomial degree): ");
    io::stdout().flush()?;

    let mut n_input = String::new();
    io::stdin().read_line(&mut n_input)?;

    let n: usize = n_input.trim().parse().unwrap_or(0);

    print!("Enter x value: ");
    io::stdout().flush()?;

    let mut x_input = String::new();
    io::stdin().read_line(&mut x_input)?;

    match x_input.trim().parse::<f64>() {
        Ok(x) => {
            println!("\nOrthogonal polynomials of degree {} at x = {}:", n, x);

            let legendre_val = legendre(n, x);
            println!("Legendre P_{}({}) = {:.10}", n, x, legendre_val);

            let cheby1_val = chebyshev(n, x, true);
            println!("Chebyshev T_{}({}) = {:.10}", n, x, cheby1_val);

            let cheby2_val = chebyshev(n, x, false);
            println!("Chebyshev U_{}({}) = {:.10}", n, x, cheby2_val);

            let hermite_val = hermite(n, x);
            println!("Hermite H_{}({}) = {:.10}", n, x, hermite_val);

            if x >= 0.0 {
                let laguerre_val = laguerre(n, x);
                println!("Laguerre L_{}({}) = {:.10}", n, x, laguerre_val);
            }
        }
        Err(_) => println!("Error: Invalid input"),
    }

    Ok(())
}

fn theorem_verifier() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Theorem Verifier ===");
    println!("Verify important mathematical identities:");
    println!("1. Gamma Reflection Formula: Γ(z)Γ(1-z) = π/sin(πz)");
    println!("2. Beta-Gamma Relation: B(a,b) = Γ(a)Γ(b)/Γ(a+b)");
    println!("3. Bessel Identity: J_{{-n}}(x) = (-1)^n J_n(x)");
    println!("4. Error Function Identity: erf(x) + erfc(x) = 1");
    print!("\nChoose theorem (1-4): ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;

    match choice.trim() {
        "1" => verify_gamma_reflection()?,
        "2" => verify_beta_gamma()?,
        "3" => verify_bessel_identity()?,
        "4" => verify_error_function_identity()?,
        _ => println!("Invalid choice."),
    }

    Ok(())
}

fn verify_gamma_reflection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nVerifying Γ(z)Γ(1-z) = π/sin(πz)");

    let test_values = vec![0.25, 0.3333, 0.5, 0.6, 0.75];

    for z in test_values {
        let lhs = gamma(z) * gamma(1.0 - z);
        let rhs = PI / (PI * z).sin();
        let error = ((lhs - rhs) / rhs).abs();

        println!(
            "z = {:.4}: LHS = {:.10}, RHS = {:.10}, Error = {:.2e}",
            z, lhs, rhs, error
        );
    }

    Ok(())
}

fn verify_beta_gamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nVerifying B(a,b) = Γ(a)Γ(b)/Γ(a+b)");

    let test_pairs: Vec<(f64, f64)> = vec![(2.0, 3.0), (1.5, 2.5), (3.0, 4.0)];

    for (a, b) in test_pairs {
        let beta_direct = beta(a, b);
        let beta_from_gamma = gamma(a) * gamma(b) / gamma(a + b);
        let error = ((beta_direct - beta_from_gamma) / beta_from_gamma).abs();

        println!("B({},{}) direct = {:.10}", a, b, beta_direct);
        println!("From Γ relation = {:.10}", beta_from_gamma);
        println!("Error = {:.2e}\n", error);
    }

    Ok(())
}

fn verify_bessel_identity() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nVerifying J_{{-n}}(x) = (-1)^n J_n(x) for integer n");

    let x = 2.0;
    for n in 1..=5 {
        let j_pos = jv::<f64>(n as f64, x);
        let j_neg = jv::<f64>(-(n as f64), x);
        let expected = if n % 2 == 0 { j_pos } else { -j_pos };
        let error = (j_neg - expected).abs();

        println!(
            "n = {}: J_{}({}) = {:.10}, J_{{-{}}}({}) = {:.10}, Error = {:.2e}",
            n, n, x, j_pos, n, x, j_neg, error
        );
    }

    Ok(())
}

fn verify_error_function_identity() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nVerifying erf(x) + erfc(x) = 1");

    let test_values: Vec<f64> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];

    for x in test_values {
        let sum = erf(x) + erfc(x);
        let error = (sum - 1.0).abs();

        println!(
            "x = {:.2}: erf + erfc = {:.15}, Error = {:.2e}",
            x, sum, error
        );
    }

    Ok(())
}

fn numerical_experiments() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Numerical Experiments ===");
    println!("1. Compare function evaluation methods");
    println!("2. Convergence rate analysis");
    println!("3. Precision limits exploration");
    print!("\nChoice: ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;

    match choice.trim() {
        "1" => compare_methods()?,
        "2" => convergence_analysis()?,
        "3" => precision_limits()?,
        _ => println!("Invalid choice."),
    }

    Ok(())
}

fn compare_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nComparing Γ(x) computation for large x:");
    println!("Direct vs Stirling approximation");

    for x in (10..=50).step_by(10) {
        let x_f = x as f64;
        let direct = gamma(x_f);
        let stirling = (2.0 * PI / x_f).sqrt() * (x_f / std::f64::consts::E).powf(x_f);
        let rel_error = ((direct - stirling) / direct).abs();

        println!(
            "x = {}: Direct = {:.6e}, Stirling = {:.6e}, Rel Error = {:.2e}",
            x, direct, stirling, rel_error
        );
    }

    Ok(())
}

fn convergence_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nAnalyzing series convergence for exp(x):");
    let x: f64 = 1.0;
    let target = x.exp();

    println!("Partial sums approaching e^{} = {}:", x, target);
    let mut sum = 0.0;
    let mut term = 1.0;

    for n in 0..=20 {
        sum += term;
        let error = (sum - target).abs();
        println!("n = {:2}: S_n = {:.15}, Error = {:.2e}", n, sum, error);
        term *= x / ((n + 1) as f64);
    }

    Ok(())
}

fn precision_limits() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nExploring numerical precision limits:");
    println!("Testing erf(x) + erfc(x) = 1 for various x:");

    let test_values: Vec<f64> = vec![1e-10, 1e-5, 1.0, 10.0, 20.0, 30.0];

    for x in test_values {
        let sum = erf(x) + erfc(x);
        let deviation = (sum - 1.0).abs();
        println!(
            "x = {:8.2e}: sum = {:.17}, deviation = {:.2e}",
            x, sum, deviation
        );
    }

    Ok(())
}

fn function_relationships() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Function Relationships ===");
    println!("Exploring connections between special functions");

    println!("\n1. Beta and Gamma:");
    println!("   B(a,b) = Γ(a)Γ(b)/Γ(a+b)");
    let a = 2.5;
    let b = 3.5;
    println!("   B({},{}) = {:.10}", a, b, beta(a, b));
    println!(
        "   Γ({})Γ({})/Γ({}) = {:.10}",
        a,
        b,
        a + b,
        gamma(a) * gamma(b) / gamma(a + b)
    );

    println!("\n2. Error function and Normal CDF:");
    println!("   Φ(x) = (1/2)[1 + erf(x/√2)]");
    let x = 1.0;
    let cdf = 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
    println!("   Φ({}) = {:.10}", x, cdf);

    println!("\n3. Spherical Bessel and Regular Bessel:");
    println!("   j_n(x) = √(π/2x) J_(n+1/2)(x)");
    let x = 1.0;
    let j0_sph = spherical_jn(0, x);
    let j_half = jv::<f64>(0.5, x);
    let from_bessel = (PI / (2.0 * x)).sqrt() * j_half;
    println!("   j_0({}) = {:.10}", x, j0_sph);
    println!("   From J_1/2: {:.10}", from_bessel);
    println!("   Error: {:.2e}", (j0_sph - from_bessel).abs());

    Ok(())
}

fn view_workspace(workspace: &HashMap<String, f64>) {
    println!("\n=== Workspace Variables ===");
    for (name, value) in workspace.iter() {
        println!("{:15} = {:.15}", name, value);
    }
}
