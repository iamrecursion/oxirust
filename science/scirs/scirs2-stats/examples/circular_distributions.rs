//! Circular distributions example
//!
//! This example demonstrates the von Mises and wrapped Cauchy distributions
//! which are specialized for circular data (angles, directions, etc.).

use scirs2_stats::distributions::circular::{VonMises, WrappedCauchy};
use scirs2_stats::traits::CircularDistribution;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Circular Distributions Example");
    println!("==============================\n");

    // Von Mises distribution example
    println!("1. 🎯 Von Mises Distribution");
    let vm = VonMises::new(PI / 4.0, 2.0)?;

    println!("   Parameters: μ = {:.3}, κ = {:.3}", vm.mu, vm.kappa);
    println!("   Mean direction: {:.3}", vm.circular_mean());
    println!("   Concentration: {:.3}", vm.concentration());
    println!("   Circular variance: {:.3}", vm.circular_variance());

    // Calculate PDF and CDF at some points
    let test_points = [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];
    println!("   Test points:");
    for &x in &test_points {
        println!(
            "     x = {:.3}: PDF = {:.4}, CDF = {:.4}",
            x,
            vm.pdf(x),
            vm.cdf(x)
        );
    }

    // Generate some samples
    println!("   Random samples:");
    for i in 0..5 {
        let sample = vm.rvs_single().expect("Operation failed");
        println!("     Sample {}: {:.3}", i + 1, sample);
    }

    println!();

    // Wrapped Cauchy distribution example
    println!("2. 🌀 Wrapped Cauchy Distribution");
    let wc = WrappedCauchy::new(0.0, 0.6)?;

    println!("   Parameters: μ = {:.3}, γ = {:.3}", wc.mu, wc.gamma);
    println!("   Mean direction: {:.3}", wc.circular_mean());
    println!("   Concentration: {:.3}", wc.concentration());
    println!("   Circular variance: {:.3}", wc.circular_variance());

    // Calculate PDF and CDF at some points
    println!("   Test points:");
    for &x in &test_points {
        println!(
            "     x = {:.3}: PDF = {:.4}, CDF = {:.4}",
            x,
            wc.pdf(x),
            wc.cdf(x)
        );
    }

    // Generate some samples
    println!("   Random samples:");
    for i in 0..5 {
        let sample = wc.rvs_single().expect("Operation failed");
        println!("     Sample {}: {:.3}", i + 1, sample);
    }

    println!();

    // Compare different concentrations
    println!("3. 📊 Concentration Comparison");
    let vm1 = VonMises::new(0.0, 1.0)?;
    let vm2 = VonMises::new(0.0, 5.0)?;
    let vm3 = VonMises::new(0.0, 20.0)?;
    let wc1 = WrappedCauchy::new(0.0, 0.3)?;
    let wc2 = WrappedCauchy::new(0.0, 0.7)?;

    println!("   Von Mises distributions:");
    println!(
        "     κ = 1.0:  Circular var = {:.3}",
        vm1.circular_variance()
    );
    println!(
        "     κ = 5.0:  Circular var = {:.3}",
        vm2.circular_variance()
    );
    println!(
        "     κ = 20.0: Circular var = {:.3}",
        vm3.circular_variance()
    );

    println!("   Wrapped Cauchy distributions:");
    println!(
        "     γ = 0.3: Circular var = {:.3}",
        wc1.circular_variance()
    );
    println!(
        "     γ = 0.7: Circular var = {:.3}",
        wc2.circular_variance()
    );

    println!("\n✅ Circular distributions example completed successfully!");
    Ok(())
}
