use scirs2_stats::distributions::exponential::Exponential;
use scirs2_stats::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use statrs::statistics::Statistics;
use std::f64;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Exponential Distribution Example");
    println!("==============================\n");

    // Create a standard exponential distribution (rate=1)
    let exp = Exponential::new(1.0f64, 0.0)?;

    println!("1. Basic Exponential Distribution (rate=1.0)");
    println!("   Mean: {}", exp.mean());
    println!("   Variance: {}", exp.var());
    println!("   Standard deviation: {}\n", exp.std());

    // Using direct methods
    println!("2. PDF and CDF at various points (direct method):");
    println!("   PDF(0.0) = {:.6}", exp.pdf(0.0));
    println!("   PDF(0.5) = {:.6}", exp.pdf(0.5));
    println!("   PDF(1.0) = {:.6}", exp.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", exp.pdf(2.0));
    println!("   CDF(0.5) = {:.6}", exp.cdf(0.5));
    println!("   CDF(1.0) = {:.6}", exp.cdf(1.0));
    println!("   CDF(2.0) = {:.6}\n", exp.cdf(2.0));

    // Using the trait interface (ContinuousDistribution)
    let dist: &dyn ContinuousDistribution<f64> = &exp;

    println!("3. PDF and CDF at various points (trait method):");
    println!("   PDF(0.5) = {:.6}", dist.pdf(0.5));
    println!("   PDF(1.0) = {:.6}", dist.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", dist.pdf(2.0));
    println!("   CDF(0.5) = {:.6}", dist.cdf(0.5));
    println!("   CDF(1.0) = {:.6}", dist.cdf(1.0));
    println!("   CDF(2.0) = {:.6}\n", dist.cdf(2.0));

    // Survival function and hazard function (using ContinuousCDF trait)
    let dist_cdf: &dyn ContinuousCDF<f64> = &exp;
    println!("4. Additional functions available through the ContinuousCDF trait:");
    println!("   Survival function SF(1.0) = {:.6}", dist_cdf.sf(1.0));
    println!("   Hazard function h(1.0) = {:.6}", dist_cdf.hazard(1.0));
    println!(
        "   Cumulative hazard function H(1.0) = {:.6}\n",
        dist_cdf.cumhazard(1.0)
    );

    // Quantile function (inverse CDF)
    println!("5. Quantile functions:");
    println!("   Median (p=0.50) = {:.6}", dist.ppf(0.50)?);
    println!("   95th percentile (p=0.95) = {:.6}", dist.ppf(0.95)?);
    println!("   99th percentile (p=0.99) = {:.6}", dist.ppf(0.99)?);
    println!(
        "   Inverse survival function (p=0.75) = {:.6}\n",
        dist_cdf.isf(0.75)?
    );

    // Random sampling
    println!("6. Random sampling:");
    let samples = dist.rvs(5)?;
    println!("   5 random samples: {:?}\n", samples);

    // Different parameterizations
    println!("7. Different parameterizations:");
    let exp_rate2 = Exponential::new(2.0f64, 0.0)?;
    println!("   Exponential with rate=2.0:");
    println!("   Mean: {}", exp_rate2.mean());
    println!("   Variance: {}", exp_rate2.var());
    println!("   Entropy: {:.6}\n", exp_rate2.entropy());

    // Creating from scale parameter
    let exp_scale = Exponential::from_scale(2.0f64, 0.0)?;
    println!("8. Parameterization using scale=2.0 (rate=0.5):");
    println!("   Mean: {}", exp_scale.mean());
    println!("   Variance: {}", exp_scale.var());
    println!("   Standard deviation: {}\n", exp_scale.std());

    // Shifted exponential
    let shifted = Exponential::new(1.0f64, 1.0)?;
    println!("9. Shifted exponential (rate=1.0, loc=1.0):");
    println!("   Mean: {}", shifted.mean());
    println!("   Median: {:.6}", shifted.ppf(0.5)?);
    println!("   PDF(1.0) = {:.6}", shifted.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", shifted.pdf(2.0));
    println!("   CDF(2.0) = {:.6}", shifted.cdf(2.0));

    Ok(())
}
