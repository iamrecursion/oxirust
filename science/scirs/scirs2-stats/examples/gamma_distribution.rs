use scirs2_stats::distributions::gamma::Gamma;
use scirs2_stats::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use statrs::statistics::Statistics;
use std::f64;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Gamma Distribution Example");
    println!("==========================\n");

    // Create a standard gamma distribution with shape=2.0, scale=1.0
    let gamma = Gamma::new(2.0f64, 1.0, 0.0)?;

    println!("1. Basic Gamma Distribution (shape=2.0, scale=1.0)");
    println!("   Mean: {}", gamma.mean());
    println!("   Variance: {}", gamma.var());
    println!("   Standard deviation: {}\n", gamma.std());

    // Using direct methods
    println!("2. PDF and CDF at various points (direct method):");
    println!("   PDF(0.5) = {:.6}", gamma.pdf(0.5));
    println!("   PDF(1.0) = {:.6}", gamma.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", gamma.pdf(2.0));
    println!("   CDF(0.5) = {:.6}", gamma.cdf(0.5));
    println!("   CDF(1.0) = {:.6}", gamma.cdf(1.0));
    println!("   CDF(2.0) = {:.6}\n", gamma.cdf(2.0));

    // Using the trait interface (ContinuousDistribution)
    let dist: &dyn ContinuousDistribution<f64> = &gamma;

    println!("3. PDF and CDF at various points (trait method):");
    println!("   PDF(0.5) = {:.6}", dist.pdf(0.5));
    println!("   PDF(1.0) = {:.6}", dist.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", dist.pdf(2.0));
    println!("   CDF(0.5) = {:.6}", dist.cdf(0.5));
    println!("   CDF(1.0) = {:.6}", dist.cdf(1.0));
    println!("   CDF(2.0) = {:.6}\n", dist.cdf(2.0));

    // Survival function and hazard function (using ContinuousCDF trait)
    let dist_cdf: &dyn ContinuousCDF<f64> = &gamma;
    println!("4. Additional functions available through the ContinuousCDF trait:");
    println!("   Survival function SF(1.0) = {:.6}", dist_cdf.sf(1.0));
    println!("   Hazard function h(1.0) = {:.6}", dist_cdf.hazard(1.0));
    println!(
        "   Cumulative hazard function H(1.0) = {:.6}\n",
        dist_cdf.cumhazard(1.0)
    );

    // Quantile function (inverse CDF)
    println!("5. Quantile functions:");
    println!("   Quantile (p=0.25) = {:.6}", dist.ppf(0.25)?);
    println!("   Quantile (p=0.50) = {:.6}", dist.ppf(0.50)?);
    println!("   Quantile (p=0.75) = {:.6}", dist.ppf(0.75)?);
    println!(
        "   Inverse survival function (p=0.75) = {:.6}\n",
        dist_cdf.isf(0.75)?
    );

    // Random sampling
    println!("6. Random sampling:");
    let samples = dist.rvs(5)?;
    println!("   5 random samples: {:?}\n", samples);

    // Chi-square distribution (special case of gamma)
    // Chi-square with k degrees of freedom is Gamma(k/2, 2)
    println!("7. Special case: Chi-square distribution (4 degrees of freedom)");
    let chi_square_4 = Gamma::new(2.0f64, 2.0, 0.0)?;

    println!("   Mean: {} (should be 4)", chi_square_4.mean());
    println!("   Variance: {} (should be 8)", chi_square_4.var());
    println!(
        "   Standard deviation: {:.6} (should be 2.828)\n",
        chi_square_4.std()
    );

    // Exponential distribution (special case of gamma)
    // Gamma(1, 1/lambda) is an Exponential(lambda) distribution
    println!("8. Special case: Exponential distribution (rate=2)");
    let exponential_2 = Gamma::new(1.0f64, 0.5, 0.0)?;

    println!("   Mean: {} (should be 0.5)", exponential_2.mean());
    println!("   Variance: {} (should be 0.25)", exponential_2.var());
    println!(
        "   Standard deviation: {:.6} (should be 0.5)",
        exponential_2.std()
    );

    Ok(())
}
