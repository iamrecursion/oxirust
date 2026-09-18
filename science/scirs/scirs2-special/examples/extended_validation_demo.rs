//! Extended validation demonstration
//!
//! This example runs comprehensive validation tests for the newly implemented
//! SciPy compatibility functions to ensure they meet mathematical properties
//! and numerical accuracy requirements.

use scirs2_special::extended_scipy_validation::run_extended_validation_tests;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2-Special Extended Validation Demo");
    println!("=====================================\n");

    println!("This demo runs comprehensive validation tests for:");
    println!(
        "• Exponentially scaled Bessel functions (j0e, j1e, y0e, y1e, i0e, i1e, k0e, k1e, etc.)"
    );
    println!("• Dawson's integral function (dawsn)");
    println!("• Polygamma function (polygamma)");
    println!("• Numerical stability and accuracy validation");
    println!("• Comparison against reference values\n");

    match run_extended_validation_tests() {
        Ok(()) => {
            println!("\n🎉 All validation tests completed successfully!");
            println!("The newly implemented functions are mathematically correct and numerically stable.");
        }
        Err(e) => {
            println!("\n❌ Validation failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
