//! Cross-validation demonstration against reference implementations
//!
//! This example shows how to validate special functions against
//! multiple reference implementations like SciPy, GSL, and MPFR.

use scirs2_special::{
    bessel::{j0, y0},
    beta,
    cross_validation::{CrossValidator, PythonValidator},
    digamma, erf, erfc, gamma,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Special Functions Cross-Validation Demo");
    println!("======================================\n");

    // Run basic cross-validation
    basic_validation()?;

    // Demonstrate Python validation
    python_validation()?;

    // Show comprehensive validation
    comprehensive_validation()?;

    Ok(())
}

#[allow(dead_code)]
fn basic_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Cross-Validation");
    println!("------------------------\n");

    let mut validator = CrossValidator::new();
    validator.load_test_cases()?;

    // Validate gamma function
    println!("Validating gamma function:");
    let gamma_summary = validator.validate_function("gamma", |args| gamma(args[0]));
    print_summary(&gamma_summary);

    // Validate Bessel J0
    println!("\nValidating Bessel J0 function:");
    let j0_summary = validator.validate_function("bessel_j0", |args| j0(args[0]));
    print_summary(&j0_summary);

    // Validate error function
    println!("\nValidating error function:");
    let erf_summary = validator.validate_function("erf", |args| erf(args[0]));
    print_summary(&erf_summary);

    Ok(())
}

#[allow(dead_code)]
fn python_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n2. Python/SciPy Validation");
    println!("--------------------------\n");

    // Check if Python is available
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("Python3 not available, skipping Python validation");
        return Ok(());
    }

    let py_validator = PythonValidator::new();

    // Test cases
    let test_cases = vec![
        ("gamma", vec![5.0]),
        ("gamma", vec![0.5]),
        ("beta", vec![2.0, 3.0]),
        ("j0", vec![1.0]),
        ("j0", vec![10.0]),
        ("erf", vec![1.0]),
    ];

    println!("Comparing with SciPy reference values:\n");
    println!("Function | Input | Our Value | SciPy Value | Rel Error");
    println!("---------|-------|-----------|-------------|----------");

    for (func_name, args) in test_cases {
        let our_value = match func_name {
            "gamma" => gamma(args[0]),
            "beta" => beta(args[0], args[1]),
            "j0" => j0(args[0]),
            "erf" => erf(args[0]),
            _ => continue,
        };

        match py_validator.compute_reference(func_name, &args) {
            Ok(scipy_value) => {
                let rel_error = ((our_value - scipy_value) / scipy_value).abs();
                println!(
                    "{:8} | {:?} | {:.6e} | {:.6e} | {:.2e}",
                    func_name, args, our_value, scipy_value, rel_error
                );
            }
            Err(e) => {
                println!("{:8} | {:?} | Error: {}", func_name, args, e);
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn comprehensive_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n3. Comprehensive Validation Report");
    println!("----------------------------------\n");

    let mut validator = CrossValidator::new();
    validator.load_test_cases()?;

    // Validate multiple functions
    let functions = vec![
        (
            "gamma",
            Box::new(|args: &[f64]| gamma(args[0])) as Box<dyn Fn(&[f64]) -> f64>,
        ),
        ("digamma", Box::new(|args: &[f64]| digamma(args[0]))),
        ("j0", Box::new(|args: &[f64]| j0(args[0]))),
        ("y0", Box::new(|args: &[f64]| y0(args[0]))),
        ("erf", Box::new(|args: &[f64]| erf(args[0]))),
        ("erfc", Box::new(|args: &[f64]| erfc(args[0]))),
    ];

    for (name, func) in functions {
        let summary = validator.validate_function(name, func);
        if summary.total_tests > 0 {
            print_detailed_summary(&summary);
        }
    }

    // Generate and save report
    let report = validator.generate_report();
    std::fs::write("CROSS_VALIDATION_REPORT.md", report)?;
    println!("\nDetailed report saved to CROSS_VALIDATION_REPORT.md");

    Ok(())
}

#[allow(dead_code)]
fn print_summary(summary: &scirs2_special::cross_validation::ValidationSummary) {
    println!("  Total tests: {}", summary.total_tests);
    println!(
        "  Passed: {} ({:.1}%)",
        summary.passed,
        100.0 * summary.passed as f64 / summary.total_tests as f64
    );
    println!("  Failed: {}", summary.failed);
    println!("  Max error: {:.2e}", summary.max_error);
    println!("  Mean error: {:.2e}", summary.mean_error);
    println!("  Max ULP error: {}", summary.max_ulp_error);
}

#[allow(dead_code)]
fn print_detailed_summary(summary: &scirs2_special::cross_validation::ValidationSummary) {
    println!("\n{} Validation:", summary.function);
    println!(
        "  Tests: {} total, {} passed, {} failed",
        summary.total_tests, summary.passed, summary.failed
    );

    if summary.failed > 0 {
        println!("  Failed cases:");
        for (i, result) in summary.failed_cases.iter().take(3).enumerate() {
            println!(
                "    {}: inputs={:?}, expected={:.6e}, computed={:.6e}, rel_err={:.2e}",
                i + 1,
                result.test_case.inputs,
                result.test_case.expected,
                result.computed,
                result.relative_error
            );
        }
        if summary.failed > 3 {
            println!("    ... and {} more", summary.failed - 3);
        }
    }

    println!("  Error statistics:");
    println!("    Max error: {:.2e}", summary.max_error);
    println!("    Mean error: {:.2e}", summary.mean_error);
    println!("    Max ULP error: {}", summary.max_ulp_error);
}
